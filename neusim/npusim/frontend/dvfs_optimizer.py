"""Request-level DVFS optimization for NeuSim.

Ported from tracked trace-util ``power_management_config_lib.py`` at commit
``e26dcd5033ea90c893c63ad962f4d0e37b83fa8e``, starting at
``_is_floatish``. Power-gating configuration remains in NeuSim's dedicated
config module. The millisecond region modules were subsequently recovered from
trace-util commit ``8ad6961b2a266e91ebb1162c7e2c5df61d10b1a4`` and ported
separately.
"""

import csv
import heapq
import os
import pickle
import random
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from copy import deepcopy
from itertools import chain, combinations

import numpy as np
from absl import logging

import neusim.npusim.frontend.Operator as Operator
from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.models.ModelConfig import ModelConfig
from neusim.configs.power_gating.PowerGatingConfig import PowerGatingConfig
from neusim.npusim.backend.dvfs_custom_policy import couple_compute_domains
from neusim.npusim.backend.dvfs_policy_lib import get_dvfs_config
from neusim.npusim.backend.dvfs_power_getter import (
    _min_power_point,
    get_all_dvfs_configs_for_component,
    get_all_dvfs_configs_for_op,
    iter_all_dvfs_configs_for_op,
)
from neusim.npusim.frontend.dvfs_candidate_batch import (
    MS_CANDIDATE_BATCH_SIZE_ENV,
    ORDERED_RAY_CANDIDATE_BATCH_MODE,
    SERIAL_CANDIDATE_MODE,
    analyze_operator_energy_candidates,
    configured_ms_candidate_batch_size,
)
from neusim.npusim.frontend.dvfs_candidate_batch import (
    analyze_operator_energy_batch as _analyze_ms_operator_energy_batch,
)
from neusim.npusim.frontend.Operator import (
    ComponentDVFSConfig,
    DVFSConfig,
    DVFSPolicy,
)

PAPER_MS_INTERVAL_NS = 5_000_000.0
"""Frequency-adjustment interval used by the MICRO26 paper experiments."""

DEFAULT_PARETO_ANALYSIS_BATCH_SIZE = 128
"""Candidates analyzed sequentially by one exhaustive-search Ray task."""

MAX_DEFAULT_PARETO_INFLIGHT_BATCHES = 24
"""Default Ray task window cap; available CPU slots may lower this value."""

MAX_PARETO_DUMP_MERGE_FAN_IN = 64
"""Maximum sorted spool files opened by one external merge pass."""

DEFAULT_DVFS_GA_EXACT_BATCH_SIZE = 32
"""Population rows gathered at once by the bit-exact scalar GA evaluator."""

DVFS_GA_EXACT_BATCH_SIZE_ENV = "DVFS_GA_EXACT_BATCH_SIZE"
DVFS_GA_SCALAR_EXACT_EXECUTION_MODE = "scalar_exact_batched_ltr"
DVFS_GA_VECTORIZED_EXECUTION_MODE = "vectorized_non_bit_exact"


def _is_floatish(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_global_dvfs_config_helper(
    dvfs_config: str | DVFSConfig | DVFSPolicy | None = None,
) -> DVFSConfig:
    """
    Parse a DVFS policy string into a DVFSConfig.

    Supported string forms (tokens joined by '_'):
      - "<policy>"                     e.g. "Custom", "Ideal", "None"
      - "<policy>_<perf_degrad>"       e.g. "Custom_0.05" (perf degradation 5%)
      - "<policy>_<domain_mode>"       e.g. "Custom_dom3", "Custom_dom4_savu"
      - "<policy>_<domain_mode>_<perf_degrad>"  e.g. "Custom_dom3_0.05"
      - "<ms_policy>_<perf_degrad>_<interval_ns>"
        e.g. "DVFSCms_0.1_1000000" (10% degradation, 1 ms interval)
    A domain mode (any token starting with "dom") may itself span multiple
    tokens (e.g. "dom4_savu"). The first numeric token is the performance
    degradation. For DVFSCms and CustomAllms, the second numeric token is the
    frequency-adjustment interval in ns. Domain-mode placement is flexible.
    """
    if not dvfs_config:
        dvfs_config = DVFSConfig()
    elif isinstance(dvfs_config, DVFSPolicy):
        dvfs_config = DVFSConfig(policy=dvfs_config)
    elif isinstance(dvfs_config, str):
        if dvfs_config != "None":
            tokens = dvfs_config.split("_")
            policy = DVFSPolicy.from_str(tokens[0])
            perf_degrad_factor = 0.0
            frequency_adjustment_interval_ns = 1_000_000.0
            domain_mode = "dom5"
            numeric_values: list[float] = []
            rest = tokens[1:]
            i = 0
            while i < len(rest):
                tok = rest[i]
                if tok.lower().startswith("dom"):
                    # A domain mode can span multiple tokens (e.g. "dom4_savu").
                    mode_parts = [tok]
                    j = i + 1
                    while j < len(rest) and not _is_floatish(rest[j]):
                        mode_parts.append(rest[j])
                        j += 1
                    domain_mode = "_".join(mode_parts).lower()
                    i = j
                elif _is_floatish(tok):
                    numeric_values.append(float(tok))
                    i += 1
                else:
                    i += 1

            if numeric_values:
                perf_degrad_factor = numeric_values[0]
            if (
                policy in (DVFSPolicy.DVFS_C_ms, DVFSPolicy.CUSTOM_ALL_ms)
                and len(numeric_values) >= 2
            ):
                frequency_adjustment_interval_ns = numeric_values[1]

            dvfs_config = DVFSConfig(
                policy=policy,
                performance_degradation_percentage=perf_degrad_factor,
                frequency_adjustment_interval_ns=frequency_adjustment_interval_ns,
                custom_compute_domain_mode=domain_mode,
            )
        else:
            dvfs_config = DVFSConfig(policy=DVFSPolicy.NONE)

    return dvfs_config


def configure_dvfs_for_op(
    op: Operator.Operator,
    config: ChipConfig,
    dvfs_config: DVFSConfig,
) -> Operator.Operator:
    """
    Initialize per-component DVFSConfig on `op` based on a JSON DVFS policy.
    """
    dvfs_configs = get_dvfs_config(op, config, dvfs_config)

    op.dvfs_sa = dvfs_configs["sa"]
    op.dvfs_vu = dvfs_configs["vu"]
    op.dvfs_sram = dvfs_configs["sram"]
    op.dvfs_hbm_mc = dvfs_configs["hbm_mc"]
    op.dvfs_hbm_die = dvfs_configs["hbm_die"]
    op.dvfs_hbm_io = dvfs_configs["hbm_io"]
    op.dvfs_ici_mc = dvfs_configs["ici_mc"]
    op.dvfs_ici_phy = dvfs_configs["ici_phy"]

    return op


def _positive_environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer; got {raw!r}")
    return value


def _evaluate_ga_population_scalar_reference(
    population: np.ndarray,
    pareto_time: np.ndarray,
    pareto_energy: np.ndarray,
    orig_total_time: float,
    baseline_total_energy: float,
    perf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Literal scalar GA fitness path retained for tests and debugging only."""
    population_size, num_genes = population.shape
    scores = np.zeros(population_size)
    times = np.zeros(population_size)
    energies = np.zeros(population_size)
    for individual_index in range(population_size):
        total_time = 0.0
        total_energy = 0.0
        for gene_index in range(num_genes):
            pareto_index = population[individual_index, gene_index]
            total_time += pareto_time[gene_index, pareto_index]
            total_energy += pareto_energy[gene_index, pareto_index]

        if total_energy <= 0 or np.isinf(total_energy) or total_time <= 0:
            score = 0.0
        else:
            perf_ratio = orig_total_time / total_time
            energy_ratio = baseline_total_energy / total_energy
            score = perf_ratio * energy_ratio**3
            if perf_ratio >= perf_threshold:
                score *= 10.0
        scores[individual_index] = score
        times[individual_index] = total_time
        energies[individual_index] = total_energy
    return scores, times, energies


def _evaluate_ga_population_scalar_exact(
    population: np.ndarray,
    pareto_time: np.ndarray,
    pareto_energy: np.ndarray,
    orig_total_time: float,
    baseline_total_energy: float,
    perf_threshold: float,
    *,
    batch_size: int = DEFAULT_DVFS_GA_EXACT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch GA gathers while retaining scalar left-to-right arithmetic.

    The explicit leading ``+0.0`` makes ``np.add.accumulate`` execute the same
    additions, in the same order, as the recovered scalar loop. Keeping the
    score arithmetic scalar is intentional: vector division and power differ
    by one ULP on some NumPy/SIMD builds. Advanced indexing materializes a
    gathered batch alongside ``work``, so peak temporary storage is roughly
    ``batch_size * (2 * num_genes + 1)`` float64 values.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive; got {batch_size!r}")

    population_size, num_genes = population.shape
    gene_rows = np.arange(num_genes)
    scores = np.zeros(population_size)
    times = np.zeros(population_size)
    energies = np.zeros(population_size)
    for start in range(0, population_size, batch_size):
        stop = min(start + batch_size, population_size)
        population_batch = population[start:stop]
        work = np.empty((stop - start, num_genes + 1), dtype=np.float64)
        work[:, 0] = 0.0
        work[:, 1:] = pareto_time[gene_rows[None, :], population_batch]
        np.add.accumulate(work, axis=1, out=work)
        times[start:stop] = work[:, -1]

        work[:, 0] = 0.0
        work[:, 1:] = pareto_energy[gene_rows[None, :], population_batch]
        np.add.accumulate(work, axis=1, out=work)
        energies[start:stop] = work[:, -1]

    for individual_index in range(population_size):
        total_time = times[individual_index]
        total_energy = energies[individual_index]
        if total_energy <= 0 or np.isinf(total_energy) or total_time <= 0:
            score = 0.0
        else:
            perf_ratio = orig_total_time / total_time
            energy_ratio = baseline_total_energy / total_energy
            score = perf_ratio * energy_ratio**3
            if perf_ratio >= perf_threshold:
                score *= 10.0
        scores[individual_index] = score
    return scores, times, energies


def _fill_ga_offspring_scalar_reference(
    population: np.ndarray,
    new_population: np.ndarray,
    probabilities: np.ndarray,
    num_pareto_points: list[int] | np.ndarray,
    rng: random.Random,
    np_rng: np.random.RandomState,
    *,
    crossover_prob: float,
    mutation_prob: float,
    elitism_count: int,
) -> None:
    """Literal scalar offspring loop retained for tests and debugging only."""
    population_size, num_genes = population.shape
    offspring_index = elitism_count
    while offspring_index < population_size:
        parent1_index, parent2_index = np_rng.choice(
            population_size, size=2, replace=False, p=probabilities
        )
        child1 = population[parent1_index].copy()
        child2 = population[parent2_index].copy()
        if rng.random() < crossover_prob:
            crossover_point = rng.randint(1, num_genes - 1)
            child1[crossover_point:], child2[crossover_point:] = (
                child2[crossover_point:].copy(),
                child1[crossover_point:].copy(),
            )
        for gene_index in range(num_genes):
            if rng.random() < mutation_prob:
                step = rng.choice([-1, 1])
                child1[gene_index] = max(
                    0,
                    min(
                        num_pareto_points[gene_index] - 1,
                        child1[gene_index] + step,
                    ),
                )
            if rng.random() < mutation_prob:
                step = rng.choice([-1, 1])
                child2[gene_index] = max(
                    0,
                    min(
                        num_pareto_points[gene_index] - 1,
                        child2[gene_index] + step,
                    ),
                )
        new_population[offspring_index] = child1
        offspring_index += 1
        if offspring_index < population_size:
            new_population[offspring_index] = child2
            offspring_index += 1


def _apply_collected_ga_mutations(
    child: np.ndarray,
    indices: list[int],
    steps: list[int],
    upper_bounds: np.ndarray,
) -> None:
    """Apply already-drawn independent integer mutations as one array update."""
    if not indices:
        return
    mutation_indices = np.asarray(indices, dtype=np.intp)
    mutation_steps = np.asarray(steps, dtype=np.int64)
    values = child[mutation_indices].astype(np.int64, copy=True)
    values += mutation_steps
    int64_upper_bounds = np.asarray(upper_bounds, dtype=np.int64)
    np.clip(values, 0, int64_upper_bounds[mutation_indices], out=values)
    child[mutation_indices] = values


def _fill_ga_offspring_scalar_exact(
    population: np.ndarray,
    new_population: np.ndarray,
    probabilities: np.ndarray,
    num_pareto_points: list[int] | np.ndarray,
    rng: random.Random,
    np_rng: np.random.RandomState,
    *,
    crossover_prob: float,
    mutation_prob: float,
    elitism_count: int,
) -> None:
    """Preserve every scalar RNG branch while deferring mutation application."""
    population_size, num_genes = population.shape
    upper_bounds = np.asarray(num_pareto_points, dtype=np.int64) - 1
    choose_parents = np_rng.choice
    random_draw = rng.random
    random_crossover_point = rng.randint
    random_bits = rng.getrandbits
    gene_indices = range(num_genes)
    offspring_index = elitism_count
    while offspring_index < population_size:
        parent1_index, parent2_index = choose_parents(
            population_size, size=2, replace=False, p=probabilities
        )
        child1 = population[parent1_index].copy()
        child2 = population[parent2_index].copy()
        if random_draw() < crossover_prob:
            crossover_point = random_crossover_point(1, num_genes - 1)
            child1[crossover_point:], child2[crossover_point:] = (
                child2[crossover_point:].copy(),
                child1[crossover_point:].copy(),
            )

        child1_indices: list[int] = []
        child1_steps: list[int] = []
        child2_indices: list[int] = []
        child2_steps: list[int] = []
        append_child1_index = child1_indices.append
        append_child1_step = child1_steps.append
        append_child2_index = child2_indices.append
        append_child2_step = child2_steps.append
        for gene_index in gene_indices:
            if random_draw() < mutation_prob:
                append_child1_index(gene_index)
                # choice([-1, 1]) calls _randbelow(2), which draws two bits
                # and rejects values 2 and 3. Spell out those same draws so
                # MT19937 state, rejection behavior, and direction stay
                # identical without the choice/randbelow wrapper dispatch.
                direction = random_bits(2)
                while direction >= 2:
                    direction = random_bits(2)
                append_child1_step(-1 if direction == 0 else 1)
            if random_draw() < mutation_prob:
                append_child2_index(gene_index)
                direction = random_bits(2)
                while direction >= 2:
                    direction = random_bits(2)
                append_child2_step(-1 if direction == 0 else 1)
        _apply_collected_ga_mutations(
            child1,
            child1_indices,
            child1_steps,
            upper_bounds,
        )
        _apply_collected_ga_mutations(
            child2,
            child2_indices,
            child2_steps,
            upper_bounds,
        )
        new_population[offspring_index] = child1
        offspring_index += 1
        if offspring_index < population_size:
            new_population[offspring_index] = child2
            offspring_index += 1


def _configured_operator_candidates(
    op: Operator.Operator,
    configs: Iterable[dict[str, ComponentDVFSConfig]],
) -> Iterator[Operator.Operator]:
    """Lazily copy one operator for each component-level DVFS configuration."""
    for candidate_config in configs:
        # Keep component objects independent because energy analysis annotates
        # regulator efficiency on each selected configuration.
        yield op.model_copy(
            update={
                "stats": op.stats.model_copy(),
                "dvfs_sa": candidate_config["sa"].model_copy(),
                "dvfs_vu": candidate_config["vu"].model_copy(),
                "dvfs_sram": candidate_config["sram"].model_copy(),
                "dvfs_hbm_mc": candidate_config["hbm_mc"].model_copy(),
                "dvfs_hbm_die": candidate_config["hbm_die"].model_copy(),
                "dvfs_hbm_io": candidate_config["hbm_io"].model_copy(),
                "dvfs_ici_mc": candidate_config["ici_mc"].model_copy(),
                "dvfs_ici_phy": candidate_config["ici_phy"].model_copy(),
            }
        )


def _next_candidate_batch(
    candidates: Iterator[Operator.Operator],
    batch_size: int,
) -> list[Operator.Operator]:
    batch: list[Operator.Operator] = []
    for _ in range(batch_size):
        try:
            batch.append(next(candidates))
        except StopIteration:
            break
    return batch


def _analyze_operator_energy_batch(
    candidates: list[Operator.Operator],
    config: ModelConfig,
    pg_config: "str | PowerGatingConfig | None",
    dvfs_config: DVFSConfig,
    ignore_vr_power_loss: bool,
) -> list[Operator.Operator]:
    """Analyze a candidate list serially inside one bounded Ray task."""
    from neusim.npusim.frontend.power_analysis_lib import analyze_operator_energy

    return [
        analyze_operator_energy(
            candidate,
            config,
            pg_config,
            dvfs_config,
            False,
            ignore_vr_power_loss,
        )
        for candidate in candidates
    ]


def _analyze_operator_config_batch(
    candidate_configs: list[dict[str, ComponentDVFSConfig]],
    op: Operator.Operator,
    config: ModelConfig,
    pg_config: "str | PowerGatingConfig | None",
    dvfs_config: DVFSConfig,
    ignore_vr_power_loss: bool,
) -> list[Operator.Operator]:
    """Construct and analyze candidates together inside one batch worker."""
    return _analyze_operator_energy_batch(
        list(_configured_operator_candidates(op, candidate_configs)),
        config,
        pg_config,
        dvfs_config,
        ignore_vr_power_loss,
    )


def _iter_bounded_analyzed_candidate_batches(
    candidates: Iterable[Operator.Operator | dict[str, ComponentDVFSConfig]],
    config: ModelConfig,
    pg_config: "str | PowerGatingConfig | None",
    dvfs_config: DVFSConfig,
    ignore_vr_power_loss: bool,
    *,
    serial: bool,
    run_stats: dict[str, int],
    candidate_factory_op: Operator.Operator | None = None,
) -> Iterator[tuple[int, list[Operator.Operator]]]:
    """Analyze candidates in order with a bounded Ray batch-task window."""
    batch_size = _positive_environment_integer(
        "DVFS_PARETO_BATCH_SIZE",
        DEFAULT_PARETO_ANALYSIS_BATCH_SIZE,
    )
    run_stats.update(
        {
            "candidate_batch_size": batch_size,
            "submitted_batches": 0,
            "max_inflight_batches": 0,
            "max_inflight_candidates": 0,
        }
    )
    candidate_iterator = iter(candidates)
    if candidate_factory_op is None:
        analyze_batch = _analyze_operator_energy_batch
        analyze_args = (config, pg_config, dvfs_config, ignore_vr_power_loss)
    else:
        analyze_batch = _analyze_operator_config_batch
        analyze_args = (
            candidate_factory_op,
            config,
            pg_config,
            dvfs_config,
            ignore_vr_power_loss,
        )

    if serial:
        ordinal = 0
        while batch := _next_candidate_batch(candidate_iterator, batch_size):
            run_stats["submitted_batches"] += 1
            run_stats["max_inflight_batches"] = 1
            run_stats["max_inflight_candidates"] = max(
                run_stats["max_inflight_candidates"],
                len(batch),
            )
            analyzed = analyze_batch(batch, *analyze_args)
            yield ordinal, analyzed
            ordinal += len(analyzed)
        run_stats["inflight_batch_limit"] = 1
        run_stats["cancelled_pending_batches"] = 0
        return

    import ray

    available_cpus = max(1, int(ray.available_resources().get("CPU", 1)))
    default_window = min(MAX_DEFAULT_PARETO_INFLIGHT_BATCHES, available_cpus)
    window_size = _positive_environment_integer(
        "DVFS_PARETO_MAX_INFLIGHT_BATCHES",
        default_window,
    )
    run_stats["inflight_batch_limit"] = window_size
    analyze_remote = ray.remote(analyze_batch)
    pending: dict[object, tuple[int, int]] = {}
    completed: dict[int, tuple[int, list[Operator.Operator]]] = {}
    next_ordinal = 0
    next_yield_ordinal = 0
    exhausted = False
    inflight_candidates = 0

    cancelled_pending_batches = 0

    def submit_one() -> bool:
        nonlocal exhausted, inflight_candidates, next_ordinal
        batch = _next_candidate_batch(candidate_iterator, batch_size)
        if not batch:
            exhausted = True
            return False
        batch_ordinal = next_ordinal
        next_ordinal += len(batch)
        future = analyze_remote.remote(batch, *analyze_args)
        pending[future] = (batch_ordinal, len(batch))
        inflight_candidates += len(batch)
        run_stats["submitted_batches"] += 1
        run_stats["max_inflight_batches"] = max(
            run_stats["max_inflight_batches"],
            len(pending),
        )
        run_stats["max_inflight_candidates"] = max(
            run_stats["max_inflight_candidates"],
            inflight_candidates,
        )
        return True

    try:
        while pending or completed or not exhausted:
            # Keep a full worker window, including before returning a completed
            # batch to the caller. Bound out-of-order local results to one
            # additional window so a single straggler cannot grow memory.
            while (
                len(pending) < window_size
                and len(completed) < window_size
                and not exhausted
            ):
                submit_one()

            if next_yield_ordinal in completed:
                expected_count, analyzed = completed.pop(next_yield_ordinal)
                batch_ordinal = next_yield_ordinal
                next_yield_ordinal += expected_count
                yield batch_ordinal, analyzed
                continue

            if not pending:
                if completed:
                    raise RuntimeError(
                        "bounded Pareto analysis lost canonical batch order"
                    )
                continue

            wait_refs = list(pending)
            if len(completed) >= window_size:
                wait_refs = [
                    future
                    for future, (ordinal, _) in pending.items()
                    if ordinal == next_yield_ordinal
                ]
                if not wait_refs:
                    raise RuntimeError(
                        "bounded Pareto analysis cannot locate the next batch"
                    )
            ready, _ = ray.wait(wait_refs, num_returns=1)
            future = ready[0]
            batch_ordinal, expected_count = pending.pop(future)
            analyzed = ray.get(future)
            inflight_candidates -= expected_count
            if len(analyzed) != expected_count:
                raise RuntimeError(
                    "bounded Pareto analysis returned an unexpected batch size: "
                    f"expected {expected_count}, got {len(analyzed)}"
                )
            completed[batch_ordinal] = (expected_count, analyzed)
    finally:
        cancel = getattr(ray, "cancel", None)
        for future in pending:
            if cancel is None:
                continue
            try:
                cancel(future, force=False)
                cancelled_pending_batches += 1
            except Exception as error:
                logging.warning("Could not cancel pending Pareto batch: %s", error)
        run_stats["cancelled_pending_batches"] = cancelled_pending_batches


def _indexed_pareto_sort_key(
    indexed_op: tuple[int, Operator.Operator],
    policy: DVFSPolicy,
) -> tuple[float, float, int]:
    ordinal, candidate = indexed_op
    metric = (
        candidate.stats.total_power_W
        if policy in (DVFSPolicy.DVFS_C, DVFSPolicy.DVFS_C_ms)
        else candidate.stats.total_energy_J
    )
    return candidate.stats.execution_time_ns, metric, ordinal


def _extract_indexed_pareto_front(
    indexed_ops: Iterable[tuple[int, Operator.Operator]],
    policy: DVFSPolicy,
) -> list[tuple[int, Operator.Operator]]:
    """Apply the legacy stable Pareto ordering with explicit ordinal ties."""
    ordered = sorted(
        indexed_ops, key=lambda item: _indexed_pareto_sort_key(item, policy)
    )
    if policy in (DVFSPolicy.DVFS_C, DVFSPolicy.DVFS_C_ms):
        result: list[tuple[int, Operator.Operator]] = []
        seen_times: set[float] = set()
        for indexed_op in ordered:
            execution_time = indexed_op[1].stats.execution_time_ns
            if execution_time not in seen_times:
                seen_times.add(execution_time)
                result.append(indexed_op)
        return result

    result = []
    for indexed_op in ordered:
        if (
            not result
            or indexed_op[1].stats.total_energy_J < result[-1][1].stats.total_energy_J
        ):
            result.append(indexed_op)
    return result


def _write_analyzed_dump_run(
    indexed_ops: list[tuple[int, Operator.Operator]],
    policy: DVFSPolicy,
    path: str,
) -> None:
    ordered = sorted(
        indexed_ops, key=lambda item: _indexed_pareto_sort_key(item, policy)
    )
    with open(path, "wb") as output:
        for indexed_op in ordered:
            pickle.dump(
                (
                    _indexed_pareto_sort_key(indexed_op, policy),
                    indexed_op[1].to_csv_dict(),
                ),
                output,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def _read_analyzed_dump_run(path: str) -> Iterator[tuple[tuple, dict]]:
    with open(path, "rb") as source:
        while True:
            try:
                yield pickle.load(source)
            except EOFError:
                return


def _merge_dump_runs_to_pickle(run_paths: list[str], output_path: str) -> None:
    streams = [_read_analyzed_dump_run(path) for path in run_paths]
    merged = heapq.merge(*streams, key=lambda record: record[0])
    with open(output_path, "wb") as output:
        for record in merged:
            pickle.dump(record, output, protocol=pickle.HIGHEST_PROTOCOL)


def _collapse_analyzed_dump_runs(
    run_paths: list[str],
    *,
    max_fan_in: int = MAX_PARETO_DUMP_MERGE_FAN_IN,
) -> list[str]:
    """Hierarchically merge spool runs without exceeding descriptor limits."""
    if max_fan_in < 2:
        raise ValueError("max_fan_in must be at least two")
    current_paths = list(run_paths)
    level = 0
    while len(current_paths) > max_fan_in:
        next_paths = []
        for group_index, start in enumerate(range(0, len(current_paths), max_fan_in)):
            group = current_paths[start : start + max_fan_in]
            if len(group) == 1:
                next_paths.append(group[0])
                continue
            merged_path = os.path.join(
                os.path.dirname(group[0]),
                f"merged-{level:04d}-{group_index:08d}.pickle",
            )
            _merge_dump_runs_to_pickle(group, merged_path)
            next_paths.append(merged_path)
            for path in group:
                os.unlink(path)
        current_paths = next_paths
        level += 1
    return current_paths


def _merge_analyzed_dump_runs(run_paths: list[str], output_path: str) -> None:
    """Atomically merge sorted batch runs into the legacy analyzed-point CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    collapsed_paths = _collapse_analyzed_dump_runs(run_paths)
    streams = [_read_analyzed_dump_run(path) for path in collapsed_paths]
    merged = heapq.merge(*streams, key=lambda record: record[0])
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=os.path.dirname(output_path),
            prefix=".pareto-points-",
            suffix=".csv.tmp",
            delete=False,
        ) as output:
            temporary_path = output.name
            writer = None
            for _, row in merged:
                if writer is None:
                    writer = csv.DictWriter(output, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
            if writer is None:
                raise RuntimeError("cannot dump an empty Pareto candidate set")
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def generate_pareto_energy_latency_points_for_op_exhaustive_search(
    op: Operator.Operator,
    config: ModelConfig,
    dvfs_config: DVFSConfig,
    dump_pareto_points_to_file: bool = False,
    total_exe_time_ns: int | None = None,
    ignore_vr_power_loss: bool = False,
    pg_config: "str | PowerGatingConfig | None" = None,
) -> list[Operator.Operator]:
    """
    Generate pareto energy-latency points for an operator by exhaustive sweep of all DVFS configs.
    Returns a list of Operator.Operator with different DVFS settings and their corresponding energy and latency.
    The pareto points are sorted by execution_time_ns ascending and total_energy_J descending.
    Do not modify the input operator.
    - dump_pareto_points_to_file: if True, dump the pareto points to a CSV file (for debugging/analysis purposes).
        File path is config.output_file_path/../pareto_points/{op.name}_dvfs_pareto_points.csv
    """
    _dvfs_serial = os.environ.get("DVFS_PARETO_SERIAL") == "1"
    if dvfs_config.policy == DVFSPolicy.IDEAL:
        iter_all_dvfs_configs_for_op.last_stats = {}
        all_dvfs_configs: Iterable[
            dict[str, ComponentDVFSConfig]
        ] = iter_all_dvfs_configs_for_op(
            op,
            dvfs_config.policy,
            dvfs_config.performance_degradation_percentage,
            total_exe_time_ns,
        )
    else:
        # Preserve the materialized enumeration used by non-Ideal compatibility
        # paths, including explicit exhaustive DVFS-C diagnostics.
        all_dvfs_configs = get_all_dvfs_configs_for_op(
            op,
            dvfs_config.policy,
            dvfs_config.performance_degradation_percentage,
            total_exe_time_ns,
        )

    # For DVFS_C, force HBM and ICI (both MC and PHY) to peak V/f
    if dvfs_config.policy in (DVFSPolicy.DVFS_C, DVFSPolicy.DVFS_C_ms):
        peak = ComponentDVFSConfig(
            policy=DVFSPolicy.NONE,
            voltage_V=0.7,
            frequency_GHz=1.7,
            voltage_regulator_scaling_time_ns=0,
        )
        for cfg in all_dvfs_configs:
            cfg["hbm_mc"] = peak
            cfg["hbm_die"] = peak
            cfg["hbm_io"] = peak
            cfg["ici_mc"] = peak
            cfg["ici_phy"] = peak

        # DVFS-C couples SA/VU/SRAM onto one physical rail. Sweep the SA
        # frequency, but choose the maximum voltage required by any member's
        # own V/f table at that shared frequency.
        for cfg in all_dvfs_configs:
            cfg["vu"] = cfg["sa"].model_copy()
            cfg["sram"] = cfg["sa"].model_copy()
            couple_compute_domains(cfg, "dom3")

        seen: set[tuple] = set()
        unique_configs = []
        for cfg in all_dvfs_configs:
            key = (cfg["sa"].voltage_V, cfg["sa"].frequency_GHz)
            if key not in seen:
                seen.add(key)
                unique_configs.append(cfg)
        all_dvfs_configs = unique_configs

    # Peek without materializing the lazy Ideal Cartesian product. If the
    # backend produces no candidate, retain the legacy default fallback.
    config_iterator = iter(all_dvfs_configs)
    try:
        first_config = next(config_iterator)
        config_iterator = chain((first_config,), config_iterator)
    except StopIteration:
        default_cfg = get_dvfs_config(op, config, dvfs_config)
        config_iterator = iter((default_cfg,))

    batch_stats: dict[str, int] = {}
    indexed_frontier: list[tuple[int, Operator.Operator]] = []
    num_analyzed_candidates = 0
    max_frontier_size = 0
    dump_directory = (
        tempfile.TemporaryDirectory(prefix="neusim-pareto-runs-")
        if dump_pareto_points_to_file
        else None
    )
    dump_run_paths: list[str] = []
    try:
        analyzed_batches = _iter_bounded_analyzed_candidate_batches(
            config_iterator,
            config,
            pg_config,
            dvfs_config,
            ignore_vr_power_loss,
            serial=_dvfs_serial,
            run_stats=batch_stats,
            candidate_factory_op=op,
        )
        for batch_ordinal, analyzed_batch in analyzed_batches:
            indexed_batch = [
                (batch_ordinal + offset, candidate)
                for offset, candidate in enumerate(analyzed_batch)
            ]
            indexed_frontier = _extract_indexed_pareto_front(
                chain(indexed_frontier, indexed_batch),
                dvfs_config.policy,
            )
            num_analyzed_candidates += len(indexed_batch)
            max_frontier_size = max(max_frontier_size, len(indexed_frontier))
            if dump_directory is not None:
                run_path = os.path.join(
                    dump_directory.name,
                    f"run-{len(dump_run_paths):08d}.pickle",
                )
                _write_analyzed_dump_run(indexed_batch, dvfs_config.policy, run_path)
                dump_run_paths.append(run_path)
            del indexed_batch
            del analyzed_batch

        if dump_directory is not None:
            pareto_points_file = os.path.join(
                os.path.dirname(config.output_file_path),
                "pareto_points",
                f"{op.name}_dvfs_pareto_points.csv",
            )
            _merge_analyzed_dump_runs(dump_run_paths, pareto_points_file)
    finally:
        if dump_directory is not None:
            dump_directory.cleanup()

    config_reduction = (
        deepcopy(getattr(iter_all_dvfs_configs_for_op, "last_stats", {}))
        if dvfs_config.policy == DVFSPolicy.IDEAL
        else None
    )
    run_stats: dict = {
        **batch_stats,
        "analysis_scheduler": (
            "serial_batched" if _dvfs_serial else "bounded_ray_batch_tasks"
        ),
        "config_enumeration": (
            "lazy_cartesian"
            if dvfs_config.policy == DVFSPolicy.IDEAL
            else "materialized"
        ),
        "num_analyzed_candidates": num_analyzed_candidates,
        "max_frontier_size": max_frontier_size,
        "config_reduction": config_reduction,
    }
    generate_pareto_energy_latency_points_for_op_exhaustive_search.last_run_stats = (
        run_stats
    )
    logging.info(
        "Exhaustive Pareto for %s analyzed %d candidates using %s; "
        "batch=%d, max in-flight candidates=%d, frontier=%d",
        op.name,
        num_analyzed_candidates,
        run_stats["analysis_scheduler"],
        batch_stats["candidate_batch_size"],
        batch_stats["max_inflight_candidates"],
        len(indexed_frontier),
    )
    return [candidate for _, candidate in indexed_frontier]


def generate_pareto_energy_latency_points_for_op_greedy_search(
    op: Operator.Operator,
    config: ModelConfig,
    dvfs_config: DVFSConfig,
    dump_pareto_points_to_file: bool = False,
    total_exe_time_ns: int | None = None,
    ignore_vr_power_loss: bool = False,
    pg_config: "str | PowerGatingConfig | None" = None,
) -> list[Operator.Operator]:
    """
    Generate pareto energy-latency points for an operator using a greedy algorithm.
    Starts from the min-energy (lowest freq) point and greedily steps toward higher performance,
    picking the component freq increment that gives the best energy/time tradeoff at each step.

    Returns a list of Operator.Operator on the Pareto front, sorted by execution_time_ns ascending
    and total_energy_J descending (same contract as the exhaustive search).
    """
    from neusim.npusim.frontend.power_analysis_lib import analyze_operator_energy

    # Only sweep DVFS-able components; PHY sub-components always run at fixed voltage.
    # For DVFS_C / DVFS_C_NO_PARETO: SA/VU/SRAM are coupled into a single "compute" domain
    # with shared V/f, and HBM/ICI are fixed at peak (compute-only DVFS).
    couple_compute = dvfs_config.policy in (
        DVFSPolicy.DVFS_C,
        DVFSPolicy.DVFS_C_NO_PARETO,
        DVFSPolicy.DVFS_C_ms,
    )
    if couple_compute:
        comp_names = ("compute",)  # HBM/ICI fixed at peak for DVFS_C
    else:
        comp_names = ("sa", "vu", "sram", "hbm_mc", "ici_mc")
    # Fixed config for PHY sub-components (not DVFS-able)
    phy_fixed_config = ComponentDVFSConfig(
        policy=DVFSPolicy.NONE,
        voltage_V=0.7,
        frequency_GHz=1.7,
        voltage_regulator_scaling_time_ns=0,
    )

    def _get_comp_time_ns(comp_name: str) -> float:
        if comp_name == "sa":
            return op.stats.sa_time_ns
        elif comp_name == "vu":
            return op.stats.vu_time_ns
        elif comp_name == "sram":
            return op.stats.vmem_time_ns
        elif comp_name == "compute":
            # Coupled compute domain: bottleneck is the max of SA/VU/SRAM times
            return max(op.stats.sa_time_ns, op.stats.vu_time_ns, op.stats.vmem_time_ns)
        elif comp_name == "hbm_mc":
            return op.stats.memory_time_ns
        elif comp_name == "ici_mc":
            return op.stats.ici_time_ns
        else:
            raise ValueError(f"Unsupported component name: {comp_name!r}")

    # 1. Compute max allowed exe time (same logic as get_all_dvfs_configs_for_op)
    if total_exe_time_ns is None:
        total_exe_time_ns = op.stats.execution_time_ns
    max_slack_time_ns = (
        total_exe_time_ns * dvfs_config.performance_degradation_percentage
    )
    max_allowed_exe_time_ns = (
        op.stats.execution_time_ns + max_slack_time_ns / op.stats.count
    )

    # 2. Get per-component frequency lists (sorted ascending by freq)
    comp_freq_lists: dict[str, list[ComponentDVFSConfig]] = {}
    for comp_name in comp_names:
        comp_time_ns = _get_comp_time_ns(comp_name)
        # "compute" sweeps the SA frequency lattice. _build_op applies
        # the worst-case SA/VU/SRAM voltage at that shared frequency.
        lookup_name = "sa" if comp_name == "compute" else comp_name
        if comp_time_ns > 0:
            comp_freq_lists[comp_name] = get_all_dvfs_configs_for_component(
                lookup_name, dvfs_config.policy
            )
        else:
            # Unused component: fix at min power point
            min_pp = _min_power_point(lookup_name)
            comp_freq_lists[comp_name] = [
                ComponentDVFSConfig(
                    policy=dvfs_config.policy,
                    voltage_V=min_pp.voltage_V,
                    frequency_GHz=0.05,
                    voltage_regulator_scaling_time_ns=20,
                )
            ]

    # 3. Compute min freq index per component
    #    For each component, find the lowest freq such that scaled time <= max_allowed_exe_time_ns.
    #    Component time scales as: scaled_time = orig_time * (base_freq / dvfs_freq)
    #    So we need: orig_time * (base_freq / dvfs_freq) <= max_allowed_exe_time_ns
    #    => dvfs_freq >= orig_time * base_freq / max_allowed_exe_time_ns
    from neusim.npusim.backend.dvfs_power_getter import (
        SA_POINTS,
        SRAM_POINTS,
        VU_POINTS,
        # HBM_MC_POINTS, HBM_PHY_POINTS, ICI_MC_POINTS, ICI_PHY_POINTS,
        _baseline_freq_ghz,
    )

    base_freq_map = {
        "sa": _baseline_freq_ghz(SA_POINTS),
        "vu": _baseline_freq_ghz(VU_POINTS),
        "sram": _baseline_freq_ghz(SRAM_POINTS),
        "compute": _baseline_freq_ghz(
            SA_POINTS
        ),  # coupled compute domain uses SA baseline
        # hardcoded for now since DVFS points for these are defined by BW, not freq
        "hbm_mc": 1.7,  #  _baseline_freq_ghz(HBM_MC_POINTS),
        "ici_mc": 1.7,  #  _baseline_freq_ghz(ICI_MC_POINTS),
    }

    current_indices: dict[str, int] = {}
    for comp_name in comp_names:
        comp_time_ns = _get_comp_time_ns(comp_name)
        freq_list = comp_freq_lists[comp_name]

        if comp_time_ns == 0 or len(freq_list) <= 1:
            current_indices[comp_name] = 0
            continue

        # Find min freq that keeps scaled time within max_allowed_exe_time_ns
        base_freq = base_freq_map[comp_name]
        min_freq_required = comp_time_ns * base_freq / max_allowed_exe_time_ns

        # Find the lowest index whose freq >= min_freq_required
        min_idx = 0
        for i, cfg in enumerate(freq_list):
            if cfg.frequency_GHz >= min_freq_required:
                min_idx = i
                break
        else:
            # All freqs below threshold; use the highest
            min_idx = len(freq_list) - 1

        current_indices[comp_name] = min_idx

    # Pre-compute CUSTOM_ALL static data (used by _build_op via closure)
    _is_custom_all = dvfs_config.policy in (
        DVFSPolicy.CUSTOM_ALL,
        DVFSPolicy.CUSTOM_ALL_ms,
    )
    if _is_custom_all:
        from neusim.npusim.backend.dvfs_power_getter import (
            HBM_DIE_POINTS,
            HBM_IO_POINTS,
            _baseline_bw_hbm,
        )

        _hbm_base_bw = _baseline_bw_hbm()
        _die_bws = sorted(
            [p for p in HBM_DIE_POINTS if abs(p.voltage_V - 1.2) < 0.02],
            key=lambda p: p.bandwidth_GBs,
        )
        _io_bws = sorted(HBM_IO_POINTS, key=lambda p: p.bandwidth_GBs)

    # Helper to build an operator with DVFS configs from freq indices.
    # Each dvfs field gets its own ComponentDVFSConfig copy to avoid aliasing
    # bugs where add_op_dvfs_exe_time_overhead() overwrites shared configs'
    # voltage_conversion_power_efficiency_percent.
    def _build_op(indices: dict[str, int]) -> Operator.Operator:
        _op = op.model_copy()
        _op.stats = op.stats.model_copy()
        for comp_name in comp_names:
            cfg = comp_freq_lists[comp_name][indices[comp_name]]
            if comp_name == "compute":
                compute_plan = {
                    "sa": cfg.model_copy(),
                    "vu": cfg.model_copy(),
                    "sram": cfg.model_copy(),
                }
                couple_compute_domains(compute_plan, "dom3")
                _op.dvfs_sa = compute_plan["sa"]
                _op.dvfs_vu = compute_plan["vu"]
                _op.dvfs_sram = compute_plan["sram"]
            elif comp_name == "sa":
                _op.dvfs_sa = cfg.model_copy()
            elif comp_name == "vu":
                _op.dvfs_vu = cfg.model_copy()
            elif comp_name == "sram":
                _op.dvfs_sram = cfg.model_copy()
            elif comp_name == "hbm_mc":
                _op.dvfs_hbm_mc = cfg.model_copy()
            elif comp_name == "ici_mc":
                _op.dvfs_ici_mc = cfg.model_copy()
        # For DVFS_C: fix HBM/ICI at peak (not swept)
        if couple_compute:
            _op.dvfs_hbm_mc = phy_fixed_config.model_copy()
            _op.dvfs_ici_mc = phy_fixed_config.model_copy()
        # PHY sub-components
        _op.dvfs_ici_phy = phy_fixed_config.model_copy()
        if _is_custom_all:
            # Couple Die/IO to MC bandwidth
            mc_freq = _op.dvfs_hbm_mc.frequency_GHz or 1.7
            target_bw = _hbm_base_bw * (mc_freq / 1.7)

            die_point = _die_bws[-1]
            for p in _die_bws:
                if p.bandwidth_GBs >= target_bw - 1:
                    die_point = p
                    break
            _op.dvfs_hbm_die = ComponentDVFSConfig(
                policy=DVFSPolicy.CUSTOM,
                voltage_V=die_point.voltage_V,
                frequency_GHz=mc_freq,
                voltage_regulator_scaling_time_ns=0,
            )

            io_point = _io_bws[-1]
            for p in _io_bws:
                if p.bandwidth_GBs >= target_bw - 1:
                    io_point = p
                    break
            _op.dvfs_hbm_io = ComponentDVFSConfig(
                policy=DVFSPolicy.CUSTOM,
                voltage_V=io_point.voltage_V,
                frequency_GHz=mc_freq,
                voltage_regulator_scaling_time_ns=0,
            )
        else:
            _op.dvfs_hbm_die = phy_fixed_config.model_copy()
            _op.dvfs_hbm_io = phy_fixed_config.model_copy()
        return _op

    def _evaluate_batch(indices_list: list[dict[str, int]]) -> list[Operator.Operator]:
        """Evaluate multiple DVFS configs locally (no Ray overhead)."""
        return [
            analyze_operator_energy(
                _build_op(indices),
                config,
                pg_config,
                dvfs_config,
                False,
                ignore_vr_power_loss,
            )
            for indices in indices_list
        ]

    # 4. Start from min-freq combination
    all_evaluated: list[Operator.Operator] = []
    current_op = _evaluate_batch([current_indices])[0]
    all_evaluated.append(current_op)

    # Counters for paper table: how many times the per-op loop runs and how
    # many candidate DVFS plans get evaluated per op.
    op_iteration = 0
    greedy_total_candidates = 1  # initial point

    # 5. Greedy iteration
    #    At each step, try all possible "frontier" moves: increment any non-empty
    #    subset of components that haven't reached peak freq (up to 2^5 - 1 = 31
    #    candidates). This handles co-bottleneck situations where incrementing a
    #    single component doesn't reduce exe_time because another component still
    #    dominates.
    while True:
        op_iteration += 1
        # Collect components that can still be incremented
        incrementable = []
        for comp_name in comp_names:
            if current_indices[comp_name] < len(comp_freq_lists[comp_name]) - 1:
                incrementable.append(comp_name)

        if not incrementable:
            break

        # Build all candidate index sets (all non-empty subsets of incrementable)
        candidate_indices_list: list[dict[str, int]] = []
        for r in range(1, len(incrementable) + 1):
            for subset in combinations(incrementable, r):
                trial_indices = dict(current_indices)
                for comp_name in subset:
                    trial_indices[comp_name] = current_indices[comp_name] + 1
                candidate_indices_list.append(trial_indices)

        greedy_total_candidates += len(candidate_indices_list)

        # Evaluate all candidates in parallel via Ray
        candidate_ops = _evaluate_batch(candidate_indices_list)

        best_candidate = None
        best_candidate_indices = None
        best_ratio = None  # Δenergy / Δtime; prefer most negative energy per positive time reduction

        for candidate_op, trial_indices in zip(
            candidate_ops, candidate_indices_list, strict=False
        ):
            delta_time = (
                candidate_op.stats.execution_time_ns
                - current_op.stats.execution_time_ns
            )
            delta_energy = (
                candidate_op.stats.total_energy_J - current_op.stats.total_energy_J
            )

            # We want to move toward lower exe time (delta_time < 0) and/or lower energy (delta_energy < 0)
            if delta_time < 0 and delta_energy <= 0:
                # Free improvement: lower time AND lower (or same) energy. Always prefer.
                # Among free improvements, pick the one with the largest |delta_time|
                ratio = float(
                    "-inf"
                )  # sentinel: always wins over non-free improvements
                if best_ratio is not None and best_ratio == float("-inf"):
                    if best_candidate is not None and delta_time < (
                        best_candidate.stats.execution_time_ns
                        - current_op.stats.execution_time_ns
                    ):
                        best_candidate = candidate_op
                        best_candidate_indices = trial_indices
                        best_ratio = ratio
                else:
                    best_candidate = candidate_op
                    best_candidate_indices = trial_indices
                    best_ratio = ratio
            elif delta_time < 0 and delta_energy > 0:
                # Trade: more energy for less time. Pick best ratio (smallest energy increase per time decrease).
                if best_ratio == float("-inf"):
                    continue  # free improvement exists, skip trades
                ratio = delta_energy / abs(delta_time)
                if best_ratio is None or ratio < best_ratio:
                    best_candidate = candidate_op
                    best_candidate_indices = trial_indices
                    best_ratio = ratio
            elif delta_time == 0 and delta_energy < 0:
                # Same time, less energy: always good (treat as free improvement)
                ratio = float("-inf")
                if best_ratio is not None and best_ratio == float("-inf"):
                    if best_candidate is not None and delta_energy < (
                        best_candidate.stats.total_energy_J
                        - current_op.stats.total_energy_J
                    ):
                        best_candidate = candidate_op
                        best_candidate_indices = trial_indices
                        best_ratio = ratio
                else:
                    best_candidate = candidate_op
                    best_candidate_indices = trial_indices
                    best_ratio = ratio
            # else: delta_time >= 0 and delta_energy >= 0 -> skip (no benefit)

        if best_candidate is None:
            break

        current_indices = best_candidate_indices
        current_op = best_candidate
        all_evaluated.append(current_op)

    # 6. Post-process: extract the true Pareto front from collected points
    pareto_ops: list[Operator.Operator] = []
    all_evaluated = sorted(
        all_evaluated, key=lambda x: (x.stats.execution_time_ns, x.stats.total_energy_J)
    )
    for op_candidate in all_evaluated:
        if len(pareto_ops) == 0:
            pareto_ops.append(op_candidate)
        else:
            if op_candidate.stats.total_energy_J < pareto_ops[-1].stats.total_energy_J:
                pareto_ops.append(op_candidate)

    # 7. Optional CSV dump
    if dump_pareto_points_to_file:
        op_dicts = [o.to_csv_dict() for o in all_evaluated]
        pareto_points_dir = os.path.join(
            os.path.dirname(config.output_file_path),
            "pareto_points",
        )
        os.makedirs(pareto_points_dir, exist_ok=True)
        pareto_points_file = os.path.join(
            pareto_points_dir,
            f"{op.name}_dvfs_pareto_points.csv",
        )
        with open(pareto_points_file, "w") as f:
            writer = csv.DictWriter(f, fieldnames=op_dicts[0].keys())
            writer.writeheader()
            writer.writerows(op_dicts)
        # Also dump per-op greedy search counts for the paper table
        import json as _json

        search_stats_file = os.path.join(
            pareto_points_dir,
            f"{op.name}_search_stats.json",
        )
        with open(search_stats_file, "w") as f:
            _json.dump(
                {
                    "op_name": op.name,
                    "op_description": getattr(op, "description", ""),
                    "policy": str(dvfs_config.policy.value),
                    "perf_degrad_pct": float(
                        dvfs_config.performance_degradation_percentage
                    ),
                    "op_iteration": op_iteration,
                    "greedy_total_candidates_evaluated": greedy_total_candidates,
                    "num_pareto_points_kept": len(all_evaluated),
                },
                f,
                indent=2,
            )

    return pareto_ops


def generate_pareto_energy_latency_points_for_op(
    op: Operator.Operator,
    config: ModelConfig,
    dvfs_config: DVFSConfig,
    dump_pareto_points_to_file: bool = False,
    total_exe_time_ns: int | None = None,
    ignore_vr_power_loss: bool = False,
    algorithm: str = "auto",
    pg_config: "str | PowerGatingConfig | None" = None,
) -> list[Operator.Operator]:
    """
    Generate pareto energy-latency points for an operator by sweeping DVFS settings.
    Returns a list of Operator.Operator with different DVFS settings and their corresponding energy and latency.
    The pareto points are sorted by execution_time_ns ascending and total_energy_J descending.
    Do not modify the input operator.

    Args:
        algorithm: "auto" (default), "greedy", or "exhaustive".
            "auto" selects exhaustive for IDEAL policy (exact over the retained,
            bounded candidate set) and greedy otherwise (fast).
        pg_config: Power gating config to use when evaluating energy for each candidate DVFS point.
            If None, defaults to "NoPG" (no power gating).
    """
    if algorithm == "auto":
        algorithm = "exhaustive" if dvfs_config.policy == DVFSPolicy.IDEAL else "greedy"

    if algorithm == "greedy":
        return generate_pareto_energy_latency_points_for_op_greedy_search(
            op,
            config,
            dvfs_config,
            dump_pareto_points_to_file,
            total_exe_time_ns,
            ignore_vr_power_loss,
            pg_config,
        )
    elif algorithm == "exhaustive":
        return generate_pareto_energy_latency_points_for_op_exhaustive_search(
            op,
            config,
            dvfs_config,
            dump_pareto_points_to_file,
            total_exe_time_ns,
            ignore_vr_power_loss,
            pg_config,
        )
    else:
        raise ValueError(
            f"Unsupported algorithm: {algorithm!r}. Use 'auto', 'greedy', or 'exhaustive'."
        )


def generate_pareto_energy_latency_points_for_all_ops(
    ops: list[Operator.Operator],
    config: ModelConfig,
    dvfs_config: DVFSConfig,
    dump_pareto_points_to_file: bool = False,
    ignore_vr_power_loss: bool = False,
    algorithm: str = "auto",
    pg_config: "str | PowerGatingConfig | None" = None,
) -> list[list[Operator.Operator]]:
    """
    Generate pareto energy-latency points for all operators.
    Returns a list of list of Operator.Operator.
    Do not modify the input operators.

    Args:
        algorithm: "auto" (default), "greedy", or "exhaustive".
            "auto" selects exhaustive for IDEAL policy and greedy otherwise.
        pg_config: Power gating config to use when evaluating energy for each candidate DVFS point.
            If None, defaults to "NoPG" (no power gating).
    """
    # When DVFS_PARETO_SERIAL=1, generate per-op pareto points serially instead
    # of fanning out one Ray task per op. This is used by batch drivers (e.g.
    # the offline DVFS sweep) that already parallelize across many points via
    # ray.data: nesting Ray tasks there deadlocks, because the outer tasks hold
    # all CPUs while blocked on ray.get of inner tasks that can never schedule.
    import os

    total_exe_time_ns = sum(op.stats.execution_time_ns * op.stats.count for op in ops)
    effective_exhaustive = algorithm == "exhaustive" or (
        algorithm == "auto" and dvfs_config.policy == DVFSPolicy.IDEAL
    )
    serial_environment = os.environ.get("DVFS_PARETO_SERIAL") == "1"
    if serial_environment or effective_exhaustive:
        pareto_ops = []
        operator_searches = []
        for op in ops:
            points = generate_pareto_energy_latency_points_for_op(
                op,
                config,
                dvfs_config,
                dump_pareto_points_to_file,
                total_exe_time_ns,
                ignore_vr_power_loss,
                algorithm,
                pg_config,
            )
            pareto_ops.append(points)
            if effective_exhaustive:
                operator_searches.append(
                    deepcopy(
                        getattr(
                            generate_pareto_energy_latency_points_for_op_exhaustive_search,
                            "last_run_stats",
                            {},
                        )
                    )
                )
        generate_pareto_energy_latency_points_for_all_ops.last_run_stats = {
            "outer_scheduler": (
                "sequential_ideal_operators"
                if effective_exhaustive and dvfs_config.policy == DVFSPolicy.IDEAL
                else "sequential_exhaustive_operators"
                if effective_exhaustive
                else "serial_environment"
            ),
            "nested_ray_fanout": False,
            "operator_searches": operator_searches,
            "total_analyzed_candidates": sum(
                int(search.get("num_analyzed_candidates", 0))
                for search in operator_searches
            ),
            "total_batch_tasks": sum(
                int(search.get("submitted_batches", 0)) for search in operator_searches
            ),
        }
        return pareto_ops

    import ray

    generate_pareto_energy_latency_points_for_op_remote = ray.remote(
        generate_pareto_energy_latency_points_for_op
    )

    ray_futures = [
        generate_pareto_energy_latency_points_for_op_remote.remote(
            op,
            config,
            dvfs_config,
            dump_pareto_points_to_file,
            total_exe_time_ns,
            ignore_vr_power_loss,
            algorithm,
            pg_config,  # type: ignore
        )
        for op in ops
    ]
    pareto_ops = ray.get(ray_futures)
    generate_pareto_energy_latency_points_for_all_ops.last_run_stats = {
        "outer_scheduler": "ray_per_operator",
        "nested_ray_fanout": algorithm == "exhaustive",
        "operator_searches": [],
    }
    return pareto_ops


def configure_dvfs_c_with_degradation(
    ops: list[Operator.Operator],
    config: ModelConfig,
    dvfs_config: DVFSConfig,
    dump_pareto_points_to_file: bool = False,
    # GA hyperparameters
    population_size: int = 200,
    max_generations: int = 300,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.03,
    elitism_count: int = 5,
    seed: int = 42,
    pg_config: "str | PowerGatingConfig | None" = None,
    _precomputed_points: list[list[Operator.Operator]] | None = None,
    _seed_individual: np.ndarray | None = None,
    _seed_population: np.ndarray | None = None,
    timing_result: dict | None = None,
) -> list[Operator.Operator]:
    """
    DVFS_C-specific algorithm for finding optimal DVFS configuration
    using a genetic algorithm.

    Only compute-domain components (SA, VU, SRAM) are swept; HBM and ICI are
    fixed at peak V/f. Each individual in the population is a vector of
    pareto-point indices (one per expanded operator).

    Fitness: When within perf budget, score = energy_ratio^2 * 10 (purely energy-focused).
    Near-threshold uses a ramp bonus; far over-budget gets a 0.1x penalty.

    Args:
        ops: list of operators to configure DVFS for.
        config: model configuration.
        dvfs_config: global DVFS configuration (policy must be DVFS_C).
        dump_pareto_points_to_file: if True, dump pareto points to CSV for analysis.
        population_size: number of individuals in the GA population.
        max_generations: maximum number of GA generations.
        crossover_prob: probability of crossover (P_c).
        mutation_prob: probability of mutation per gene (P_m).
        elitism_count: number of top individuals carried over unchanged.
        seed: random seed for reproducibility.

    Returns:
        list of operators with DVFS configurations applied.
    """
    logging.set_verbosity(logging.INFO)
    logging.info(
        "Using DVFS_C genetic algorithm with performance_degradation_percentage=%.4f",
        dvfs_config.performance_degradation_percentage,
    )
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    # Optional numpy-vectorized inner loop (same GA, ~50-100x faster on high
    # gene-count MoE workloads). Off by default → identical behavior to the
    # original per-gene Python loops; set DVFS_GA_VECTORIZED=1 to enable.
    # Not bit-identical to the scalar path (different RNG draw structure), but the
    # same algorithm/operators — validate energy against known results before use.
    _ga_vectorized = os.environ.get("DVFS_GA_VECTORIZED") == "1"
    _ga_execution_mode = (
        DVFS_GA_VECTORIZED_EXECUTION_MODE
        if _ga_vectorized
        else DVFS_GA_SCALAR_EXACT_EXECUTION_MODE
    )
    _ga_exact_batch_size = (
        None
        if _ga_vectorized
        else _positive_environment_integer(
            DVFS_GA_EXACT_BATCH_SIZE_ENV,
            DEFAULT_DVFS_GA_EXACT_BATCH_SIZE,
        )
    )

    # 1. Generate pareto points with a wide per-op budget so the GA has a rich search space.
    #    The real global budget is enforced by the GA fitness function, not the per-op config generation.
    #    HBM/ICI forced to peak inside generate_pareto_energy_latency_points_for_op for DVFS_C.
    _t_pareto_start = time.time()
    if _precomputed_points is not None:
        # Reuse points generated by the caller (e.g. configure_dvfs_c_no_pareto_all_budgets)
        pareto_ops = _precomputed_points
    else:
        wide_dvfs_config = DVFSConfig(
            policy=dvfs_config.policy,
            performance_degradation_percentage=1.0,  # 100% budget for config generation
        )
        pareto_ops = generate_pareto_energy_latency_points_for_all_ops(
            ops,
            config,
            wide_dvfs_config,
            dump_pareto_points_to_file,
            pg_config=pg_config,
        )

        # 1b. Inject the 0%-degradation Ideal config as an extra pareto point for each op.
        #     This guarantees the GA can always match or beat the non-GA 0% path.
        from neusim.npusim.frontend.power_analysis_lib import analyze_operator_energy

        zero_dvfs_config = DVFSConfig(
            policy=dvfs_config.policy,
            performance_degradation_percentage=0.0,
        )
        for i, op in enumerate(ops):
            ideal_op = configure_dvfs_for_op(deepcopy(op), config, zero_dvfs_config)
            ideal_op = analyze_operator_energy(
                ideal_op, config, pg_config, zero_dvfs_config, False, False
            )
            # Insert at front so index 0 is always the Ideal baseline
            pareto_ops[i].insert(0, ideal_op)
    _t_pareto_end = time.time()

    # 2. Expand operators into instances (same logic as the greedy in configure_dvfs_for_ops).
    #    For LLM decode: each op is expanded by num_layers (count / output_seqlen),
    #    with each instance weighted by output_seqlen.
    #    This allows different instances of the same operator to get different V/f configs.
    has_decode = any("decode" in op.description.lower() for op in ops)
    has_prefill = any("prefill" in op.description.lower() for op in ops)
    is_llm_decode = has_decode and not has_prefill
    if is_llm_decode:
        from neusim.configs.models.LLMConfig import LLMConfig

        assert isinstance(config, LLMConfig)
        instance_weight = config.output_seqlen
    else:
        instance_weight = 1

    expanded_indices: list[int] = []  # expanded index -> original op index
    expanded_pareto_ops: list[list[Operator.Operator]] = []
    for i, op in enumerate(ops):
        assert (
            op.stats.count % instance_weight == 0
        ), f"Op '{op.description}' count {op.stats.count} is not divisible by instance_weight {instance_weight}"
        expansion_factor = op.stats.count // instance_weight
        for _ in range(expansion_factor):
            expanded_indices.append(i)
            expanded_pareto_ops.append(pareto_ops[i])

    num_expanded = len(expanded_indices)
    num_pareto_points = [len(expanded_pareto_ops[i]) for i in range(num_expanded)]

    orig_total_time_ns = sum(op.stats.execution_time_ns * op.stats.count for op in ops)
    perf_threshold = 1.0 / (1.0 + dvfs_config.performance_degradation_percentage)

    # 4. Pre-extract execution_time, energy, and power arrays for fast fitness evaluation.
    # Each value is scaled by instance_weight (contribution of one expanded instance).
    max_pareto = max(num_pareto_points)
    pareto_time = np.full((num_expanded, max_pareto), np.inf)
    pareto_energy = np.full((num_expanded, max_pareto), np.inf)
    pareto_power = np.full((num_expanded, max_pareto), np.inf)
    for j in range(num_expanded):
        for k in range(num_pareto_points[j]):
            pareto_time[j, k] = (
                expanded_pareto_ops[j][k].stats.execution_time_ns * instance_weight
            )
            pareto_energy[j, k] = (
                expanded_pareto_ops[j][k].stats.total_energy_J * instance_weight
            )
            pareto_power[j, k] = expanded_pareto_ops[j][k].stats.total_power_W

    baseline_total_energy_J = sum(pareto_energy[j, 0] for j in range(num_expanded))

    _t_search_start = time.time()

    # 5. GA search
    # Compute the zero-degradation baseline: for each expanded instance, pick the pareto point
    # with the lowest energy (this is what configure_dvfs_for_op with DVFS_C at 0% would pick).
    # This guarantees the GA starts from at least the zero-degradation optimum.
    zero_degrad_individual = np.zeros(num_expanded, dtype=np.int32)
    for j in range(num_expanded):
        best_k = 0
        best_e = pareto_energy[j, 0]
        for k in range(1, num_pareto_points[j]):
            # Only consider points that don't increase execution time beyond baseline
            if pareto_time[j, k] <= pareto_time[j, 0] and pareto_energy[j, k] < best_e:
                best_e = pareto_energy[j, k]
                best_k = k
        zero_degrad_individual[j] = best_k

    # Initialize GA population biased toward low-power (high pareto index = slow but low power).
    # The GA will evolve toward the performance threshold from the low-power side.
    logging.info(
        "Zero-degradation baseline: energy=%.4f J, time=%.2f ms",
        sum(pareto_energy[j, zero_degrad_individual[j]] for j in range(num_expanded)),
        sum(pareto_time[j, zero_degrad_individual[j]] for j in range(num_expanded))
        / 1e6,
    )

    population = np.zeros((population_size, num_expanded), dtype=np.int32)

    if _seed_population is not None and _seed_population.shape == (
        population_size,
        num_expanded,
    ):
        # Warm start: use entire previous population
        population[:] = _seed_population
        # Ensure key individuals are still present
        population[0] = zero_degrad_individual.copy()
        if _seed_individual is not None:
            population[1] = _seed_individual.copy()
    else:
        # Cold start: initialize from scratch
        seed_idx = 0
        population[seed_idx] = zero_degrad_individual.copy()
        seed_idx += 1
        population[seed_idx] = [num_pareto_points[j] - 1 for j in range(num_expanded)]
        seed_idx += 1
        population[seed_idx] = 0
        seed_idx += 1
        if _seed_individual is not None:
            population[seed_idx] = _seed_individual.copy()
            seed_idx += 1

        is_no_pareto = dvfs_config.policy == DVFSPolicy.DVFS_C_NO_PARETO
        for i in range(seed_idx, population_size):
            for j in range(num_expanded):
                if is_no_pareto:
                    lo = 0
                else:
                    lo = num_pareto_points[j] // 2
                hi = num_pareto_points[j] - 1
                population[i, j] = rng.randint(lo, hi) if hi > lo else lo

    best_individual = zero_degrad_individual.copy()
    best_score = -np.inf
    best_time_ns = 0.0
    best_energy_J = np.inf

    # Track GA evolution history for visualization (every generation)
    ga_history: list[dict] = []
    history_interval = 1  # record every generation

    # Vectorized whole-population fitness (numpy gather+sum), same formula as
    # _get_individual_fitness. Only used when _ga_vectorized.
    _rows = np.arange(num_expanded)
    _hi = np.array(
        [num_pareto_points[j] - 1 for j in range(num_expanded)], dtype=np.int64
    )

    def _eval_population(pop: np.ndarray):
        t = pareto_time[_rows[None, :], pop].sum(axis=1)
        e = pareto_energy[_rows[None, :], pop].sum(axis=1)
        valid = (e > 0) & np.isfinite(e) & (t > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = np.where(t > 0, orig_total_time_ns / t, 0.0)
            er = np.where(e > 0, baseline_total_energy_J / e, 0.0)
            sc = pr * er**3
        sc = np.where(pr >= perf_threshold, sc * 10.0, sc)
        sc = np.where(valid, sc, 0.0)
        return sc, t, e

    for gen in range(max_generations):
        # Evaluate fitness for all individuals
        if _ga_vectorized:
            scores, all_times, all_energies = _eval_population(population)
            gi = int(np.argmax(scores))
            if scores[gi] > best_score:
                best_score = float(scores[gi])
                best_individual = population[gi].copy()
                best_time_ns = float(all_times[gi])
                best_energy_J = float(all_energies[gi])
        else:
            scores, all_times, all_energies = _evaluate_ga_population_scalar_exact(
                population,
                pareto_time,
                pareto_energy,
                orig_total_time_ns,
                baseline_total_energy_J,
                perf_threshold,
                batch_size=_ga_exact_batch_size,
            )
            for i in range(population_size):
                score = scores[i]
                t_ns = all_times[i]
                e_J = all_energies[i]
                if score > best_score:
                    best_score = score
                    best_individual = population[i].copy()
                    best_time_ns = t_ns
                    best_energy_J = e_J
        # Record best individual's absolute values for visualization
        if gen % history_interval == 0 or gen == max_generations - 1:
            ga_history.append(
                {
                    "step": gen,
                    "total_time_ns": float(best_time_ns),
                    "total_energy_J": float(best_energy_J),
                }
            )

        if gen % 100 == 0:
            logging.info(
                "GA gen %d: best_score=%.6f, best_energy=%.4f J, "
                "best_time=%.2f ms, perf_ratio=%.4f",
                gen,
                best_score,
                best_energy_J,
                best_time_ns / 1e6,
                orig_total_time_ns / best_time_ns if best_time_ns > 0 else 0,
            )

        # Selection probabilities (proportional to score)
        score_sum = scores.sum()
        if score_sum <= 0:
            probs = np.ones(population_size) / population_size
        else:
            probs = scores / score_sum

        # Build next generation
        new_population = np.zeros_like(population)

        # Elitism: carry over top individuals
        elite_indices = np.argsort(scores)[-elitism_count:]
        for ei, idx in enumerate(elite_indices):
            new_population[ei] = population[idx]

        # Fill rest via selection, crossover, mutation
        if _ga_vectorized:
            n_off = population_size - elitism_count
            if n_off > 0:
                # Roulette parents, single-point crossover, per-gene ±1 mutation —
                # same operators as the scalar path, batched over all offspring.
                p1 = np_rng.choice(population_size, size=n_off, p=probs)
                p2 = np_rng.choice(population_size, size=n_off, p=probs)
                children = population[p1].copy()
                mates = population[p2]
                if num_expanded > 1:
                    do_cx = np_rng.rand(n_off) < crossover_prob
                    ks = np_rng.randint(1, num_expanded, size=n_off)
                    tail = (_rows[None, :] >= ks[:, None]) & do_cx[:, None]
                    children = np.where(tail, mates, children)
                mut = np_rng.rand(n_off, num_expanded) < mutation_prob
                steps = np_rng.choice(np.array([-1, 1]), size=(n_off, num_expanded))
                children = children + mut * steps
                np.clip(children, 0, _hi[None, :], out=children)
                new_population[elitism_count:] = children
        else:
            _fill_ga_offspring_scalar_exact(
                population,
                new_population,
                probs,
                num_pareto_points,
                rng,
                np_rng,
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                elitism_count=elitism_count,
            )
        population = new_population

    # Final evaluation
    if _ga_vectorized:
        scores, all_times, all_energies = _eval_population(population)
        gi = int(np.argmax(scores))
        if scores[gi] > best_score:
            best_score = float(scores[gi])
            best_individual = population[gi].copy()
            best_time_ns = float(all_times[gi])
            best_energy_J = float(all_energies[gi])
    else:
        scores, all_times, all_energies = _evaluate_ga_population_scalar_exact(
            population,
            pareto_time,
            pareto_energy,
            orig_total_time_ns,
            baseline_total_energy_J,
            perf_threshold,
            batch_size=_ga_exact_batch_size,
        )
        for i in range(population_size):
            score = scores[i]
            t_ns = all_times[i]
            e_J = all_energies[i]
            if score > best_score:
                best_score = score
                best_individual = population[i].copy()
                best_time_ns = t_ns
                best_energy_J = e_J
    logging.info(
        "GA finished: best_score=%.6f, best_energy=%.4f J, "
        "best_time=%.2f ms, perf_ratio=%.4f",
        best_score,
        best_energy_J,
        best_time_ns / 1e6,
        orig_total_time_ns / best_time_ns if best_time_ns > 0 else 0,
    )

    # Save search trace for visualization
    search_trace = {
        "algorithm": "ga",
        "policy": str(dvfs_config.policy.value),
        "perf_degrad_pct": float(dvfs_config.performance_degradation_percentage),
        "orig_total_time_ns": float(orig_total_time_ns),
        "ga_execution_mode": _ga_execution_mode,
        "ga_exact_batch_size": _ga_exact_batch_size,
        "ga_exact_batch_size_env": DVFS_GA_EXACT_BATCH_SIZE_ENV,
        "trace": ga_history,
    }
    _save_search_trace(config, dvfs_config, ops, search_trace)

    # 6. Assemble result ops — group expanded instances by (orig_op, pareto_index)
    #    and split ops that have instances with different V/f configs.
    selected_ops_indices = best_individual.tolist()

    from collections import Counter

    group_counts: Counter = Counter()
    for j, orig_idx in enumerate(expanded_indices):
        group_counts[(orig_idx, selected_ops_indices[j])] += 1

    result_ops = []
    for (orig_idx, pareto_idx), cnt in sorted(group_counts.items()):
        new_op = ops[orig_idx].model_copy(deep=True)
        new_op.stats.count = cnt * instance_weight
        selected = pareto_ops[orig_idx][pareto_idx]
        new_op.dvfs_sa = selected.dvfs_sa.model_copy()
        new_op.dvfs_vu = selected.dvfs_vu.model_copy()
        new_op.dvfs_sram = selected.dvfs_sram.model_copy()
        new_op.dvfs_hbm_mc = selected.dvfs_hbm_mc.model_copy()
        new_op.dvfs_hbm_die = selected.dvfs_hbm_die.model_copy()
        new_op.dvfs_hbm_io = selected.dvfs_hbm_io.model_copy()
        new_op.dvfs_ici_mc = selected.dvfs_ici_mc.model_copy()
        new_op.dvfs_ici_phy = selected.dvfs_ici_phy.model_copy()
        result_ops.append(new_op)

    # Store GA's best energy for callers that need it (e.g. monotonicity checks)
    # without re-analyzing energy on the result ops.
    configure_dvfs_c_with_degradation.last_best_energy_J = best_energy_J
    configure_dvfs_c_with_degradation.last_best_individual = best_individual.copy()
    configure_dvfs_c_with_degradation.last_population = population.copy()

    _t_search_end = time.time()
    if timing_result is not None:
        timing_result["ga_execution_mode"] = _ga_execution_mode
        timing_result["ga_exact_batch_size"] = _ga_exact_batch_size
        timing_result["ga_exact_batch_size_env"] = DVFS_GA_EXACT_BATCH_SIZE_ENV
        timing_result["pareto_generation_seconds"] = round(
            _t_pareto_end - _t_pareto_start, 4
        )
        if _precomputed_points is None:
            timing_result["pareto_generation"] = deepcopy(
                getattr(
                    generate_pareto_energy_latency_points_for_all_ops,
                    "last_run_stats",
                    {},
                )
            )
        timing_result["inter_op_search_seconds"] = round(
            _t_search_end - _t_search_start, 4
        )
        timing_result["avg_pareto_points"] = round(
            sum(len(p) for p in pareto_ops) / len(pareto_ops), 2
        )
        timing_result["num_expanded_ops"] = num_expanded
        timing_result["inter_op_search_iterations"] = max_generations

    ops[:] = result_ops
    return ops


def _save_search_trace(
    config: ModelConfig,
    dvfs_config: DVFSConfig,
    ops: list[Operator.Operator],
    trace_data: dict,
) -> None:
    """Save a search trace JSON alongside trace CSV output.

    Saved as ``{output_dir}/search_traces/{policy}_{pct}_{phase}.json``.
    The notebook loads these by constructing the same path from the sweep config.
    """
    try:
        import json as _json

        out_path = getattr(config, "output_file_path", "") or ""
        if not out_path:
            logging.warning(
                "_save_search_trace: no output_file_path on config; skipping."
            )
            return

        # Detect phase from output path, op descriptions, or op counts
        phase = "unknown"
        if "prefill" in out_path.lower():
            phase = "prefill"
        elif "decode" in out_path.lower():
            phase = "decode"
        else:
            # Check op descriptions for phase keywords
            all_descs = " ".join(
                op.description.lower()
                for op in ops
                if hasattr(op, "description") and op.description
            )
            logging.info(
                "_save_search_trace: phase detection from descriptions: '%s'",
                all_descs[:200],
            )
            if "prefill" in all_descs:
                phase = "prefill"
            elif "decode" in all_descs or "servingdecode" in all_descs:
                phase = "decode"
            else:
                max_count = max((op.stats.count for op in ops), default=0)
                phase = "decode" if max_count > 200 else "prefill"

        trace_dir = os.path.join(os.path.dirname(out_path), "search_traces")
        os.makedirs(trace_dir, exist_ok=True)
        policy_name = dvfs_config.policy.value
        pct = dvfs_config.performance_degradation_percentage
        fname = f"{policy_name}_{pct:.2f}_{phase}.json"
        trace_path = os.path.join(trace_dir, fname)
        with open(trace_path, "w") as f:
            _json.dump(trace_data, f)
        logging.info("Search trace saved to %s", trace_path)
    except Exception as e:
        logging.warning("Failed to save search trace: %s", e)


def configure_dvfs_c_no_pareto_all_budgets(
    ops: list[Operator.Operator],
    config: ModelConfig,
    budgets: list[float],
    dump_pareto_points_to_file: bool = False,
    pg_config: "str | PowerGatingConfig | None" = None,
    *,
    population_size: int = 1000,
    max_generations: int = 500,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.15,
    elitism_count: int = 50,
    seed: int = 42,
    ga_policy: DVFSPolicy = DVFSPolicy.DVFS_C_NO_PARETO,
) -> dict[float, list[Operator.Operator]]:
    """Run a compute-only GA for multiple budgets, generating points only once.

    ``DVFS_C`` reuses the regular optimizer's 100%-wide Pareto envelope and
    preserves the original raw trace time as the GA baseline.
    ``DVFS_C_NO_PARETO`` retains its exhaustive SA-point envelope and adjusted
    zero-degradation baseline semantics.

    Budgets are processed from smallest to largest.  Monotonicity is enforced:
    if a larger budget produces worse energy, the best previous result is kept.

    Returns:
        dict mapping budget percentage → list of configured operators.
    """
    from neusim.npusim.frontend.power_analysis_lib import analyze_operator_energy

    if not budgets:
        raise ValueError("budgets must contain at least one value")
    sorted_budgets = sorted({float(value) for value in budgets})
    if any(not np.isfinite(value) or value < 0 for value in sorted_budgets):
        raise ValueError("budgets must be finite, non-negative fractions")
    if ga_policy not in (DVFSPolicy.DVFS_C, DVFSPolicy.DVFS_C_NO_PARETO):
        raise ValueError("ga_policy must be DVFS_C or DVFS_C_NO_PARETO")
    if population_size < 3 or max_generations < 1:
        raise ValueError("population_size must be >= 3 and max_generations >= 1")
    if elitism_count < 1 or elitism_count >= population_size:
        raise ValueError("elitism_count must be in [1, population_size)")
    results: dict[float, list[Operator.Operator]] = {}
    raw_baseline_time_ns = sum(
        op.stats.execution_time_ns * op.stats.count for op in ops
    )
    # Plan "compilation time" bookkeeping, exposed via function attribute:
    # point_gen_s is shared across budgets, ga_s is per budget.
    plan_timings: dict = {
        "point_gen_s": 0.0,
        "ga_s": {},
        "ga_details": {},
        "budgets": list(sorted_budgets),
        "num_ops": len(ops),
        "num_vf_points_per_op": 0,
        "total_s": 0.0,
        "ga_policy": ga_policy.value,
        "raw_baseline_time_ns": float(raw_baseline_time_ns),
        "candidate_generation": (
            "regular_100pct_pareto"
            if ga_policy == DVFSPolicy.DVFS_C
            else "all_sa_vf_points"
        ),
        "baseline_semantics": (
            "original_raw_trace_time"
            if ga_policy == DVFSPolicy.DVFS_C
            else "zero_degradation_candidate_time"
        ),
        "zero_degradation_baseline_injection": (
            "unconditional"
            if ga_policy == DVFSPolicy.DVFS_C
            else "insert_if_execution_time_differs"
        ),
    }
    configure_dvfs_c_no_pareto_all_budgets.last_timings = plan_timings
    _point_gen_start_s = time.perf_counter()

    zero_dvfs_config = DVFSConfig(
        policy=ga_policy,
        performance_degradation_percentage=0.0,
    )
    all_points: list[list[Operator.Operator]]
    if ga_policy == DVFSPolicy.DVFS_C:
        # Match configure_dvfs_c_with_degradation: generate its regular
        # 100%-wide Pareto envelope once and inject the 0%-degradation point.
        wide_dvfs_config = DVFSConfig(
            policy=ga_policy,
            performance_degradation_percentage=1.0,
        )
        all_points = generate_pareto_energy_latency_points_for_all_ops(
            ops,
            config,
            wide_dvfs_config,
            dump_pareto_points_to_file,
            pg_config=pg_config,
        )
        for i, op in enumerate(ops):
            ideal_op = configure_dvfs_for_op(deepcopy(op), config, zero_dvfs_config)
            ideal_op = analyze_operator_energy(
                ideal_op,
                config,
                pg_config,
                zero_dvfs_config,
                False,
                False,
            )
            all_points[i].insert(0, ideal_op)
        logging.info(
            "DVFS_C: generated a shared 100%% Pareto envelope for %d ops",
            len(ops),
        )
    else:
        # Generate every SA V/f point, preserving the original no-Pareto path.
        _dvfs_serial = os.environ.get("DVFS_PARETO_SERIAL") == "1"
        from neusim.npusim.backend.dvfs_power_getter import SA_POINTS
        from neusim.npusim.frontend.power_analysis_lib import (
            analyze_operator_energy as _analyze_op_energy_single,
        )

        if not _dvfs_serial:
            import ray

            _analyze_remote = ray.remote(_analyze_op_energy_single)

        peak = ComponentDVFSConfig(
            policy=DVFSPolicy.NONE,
            voltage_V=0.7,
            frequency_GHz=1.7,
            voltage_regulator_scaling_time_ns=0,
        )
        dvfs_for_eval = DVFSConfig(
            policy=ga_policy, performance_degradation_percentage=0.0
        )

        all_points = []
        for op in ops:
            ray_futures = []
            for pt in SA_POINTS:
                compute_cfg = ComponentDVFSConfig(
                    policy=ga_policy,
                    voltage_V=pt.voltage_V,
                    frequency_GHz=pt.frequency_GHz,
                    voltage_regulator_scaling_time_ns=20,
                )
                compute_plan = {
                    name: compute_cfg.model_copy() for name in ("sa", "vu", "sram")
                }
                couple_compute_domains(compute_plan, "dom3")
                _op = op.model_copy()
                _op.stats = op.stats.model_copy()
                _op.dvfs_sa = compute_plan["sa"]
                _op.dvfs_vu = compute_plan["vu"]
                _op.dvfs_sram = compute_plan["sram"]
                _op.dvfs_hbm_mc = peak.model_copy()
                _op.dvfs_hbm_die = peak.model_copy()
                _op.dvfs_hbm_io = peak.model_copy()
                _op.dvfs_ici_mc = peak.model_copy()
                _op.dvfs_ici_phy = peak.model_copy()
                if _dvfs_serial:
                    ray_futures.append(
                        _analyze_op_energy_single(
                            _op, config, pg_config, dvfs_for_eval, False, False
                        )
                    )
                else:
                    ray_futures.append(
                        _analyze_remote.remote(
                            _op, config, pg_config, dvfs_for_eval, False, False
                        )
                    )
            op_points = ray_futures if _dvfs_serial else ray.get(ray_futures)
            op_points.sort(key=lambda x: x.stats.execution_time_ns)
            all_points.append(op_points)

        logging.info(
            "DVFS_C_NO_PARETO: generated %d V/f points per op (%d ops)",
            len(SA_POINTS),
            len(ops),
        )

        for i, op in enumerate(ops):
            ideal_op = configure_dvfs_for_op(deepcopy(op), config, zero_dvfs_config)
            ideal_op = analyze_operator_energy(
                ideal_op,
                config,
                pg_config,
                zero_dvfs_config,
                False,
                False,
            )
            # Only insert if not already at index 0 (same V/f as peak).
            if (
                all_points[i][0].stats.execution_time_ns
                != ideal_op.stats.execution_time_ns
            ):
                all_points[i].insert(0, ideal_op)

        # Preserve the no-Pareto helper's adjusted-baseline semantics.
        for i, op in enumerate(ops):
            op.stats.execution_time_ns = all_points[i][0].stats.execution_time_ns

    plan_timings["point_gen_s"] = time.perf_counter() - _point_gen_start_s
    plan_timings["num_vf_points_per_op"] = max(
        (len(points) for points in all_points), default=0
    )

    # --- Run GA for each budget (including 0%), enforce monotonicity ---------
    best_energy = float("inf")
    best_ops: list[Operator.Operator] | None = None
    prev_best_individual = None
    prev_population = None

    for budget_pct in sorted_budgets:
        _ga_start_s = time.perf_counter()
        step_dvfs = DVFSConfig(
            policy=ga_policy,
            performance_degradation_percentage=budget_pct,
        )
        budget_timing: dict = {}
        candidate_ops = configure_dvfs_c_with_degradation(
            [op.model_copy(deep=True) for op in ops],
            config,
            step_dvfs,
            dump_pareto_points_to_file,
            population_size=population_size,
            max_generations=max_generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            elitism_count=elitism_count,
            seed=seed,
            pg_config=pg_config,
            _precomputed_points=all_points,
            _seed_individual=prev_best_individual,
            _seed_population=prev_population,
            timing_result=budget_timing,
        )
        prev_best_individual = configure_dvfs_c_with_degradation.last_best_individual
        prev_population = configure_dvfs_c_with_degradation.last_population
        plan_timings["ga_s"][budget_pct] = time.perf_counter() - _ga_start_s
        plan_timings["ga_details"][budget_pct] = budget_timing
        # Analyze candidate energy accurately for monotonicity comparison
        analyzed_candidate = [op.model_copy(deep=True) for op in candidate_ops]
        for op in analyzed_candidate:
            analyze_operator_energy(op, config, pg_config, step_dvfs, False, False)
        candidate_energy = sum(
            op.stats.total_energy_J * op.stats.count for op in analyzed_candidate
        )

        if candidate_energy < best_energy:
            best_energy = candidate_energy
            best_ops = candidate_ops

        # Always store the best seen so far (monotonicity guarantee)
        results[budget_pct] = [op.model_copy(deep=True) for op in best_ops]
        logging.info(
            "DVFS_C_NO_PARETO budget=%.4f: candidate=%.4f J, best=%.4f J",
            budget_pct,
            candidate_energy,
            best_energy,
        )

    plan_timings["total_s"] = plan_timings["point_gen_s"] + sum(
        plan_timings["ga_s"].values()
    )
    return results


def _ms_precompute(ops, config, interval_ns, pg_config):
    """Budget-independent data for DVFS_C_ms: regions (compute->HFC, memory/ICI->LFC,
    count-aware), per-(region,layer) gene layout, and per-region time/energy at each
    compute V/f summed over the region's ops (Σ max(components) = correct sequential
    time). Evaluated once, reused across budgets."""
    from neusim.npusim.backend.dvfs_power_getter import SA_POINTS
    from neusim.npusim.frontend.dvfs_region_merge import build_regions

    K = len(SA_POINTS)
    PEAK_K = K - 1
    has_decode = any("decode" in op.description.lower() for op in ops)
    has_prefill = any("prefill" in op.description.lower() for op in ops)
    is_llm_decode = has_decode and not has_prefill
    instance_weight = config.output_seqlen if is_llm_decode else 1

    regions = build_regions(ops, interval_ns)
    num_regions = len(regions)

    dvfs_serial = os.environ.get("DVFS_PARETO_SERIAL") == "1"
    remote_candidate_batch = None
    if not dvfs_serial:
        import ray

        remote_candidate_batch = ray.remote(_analyze_ms_operator_energy_batch)
    candidate_batching = {
        "candidate_evaluation_mode": (
            SERIAL_CANDIDATE_MODE if dvfs_serial else ORDERED_RAY_CANDIDATE_BATCH_MODE
        ),
        "candidate_batch_size": configured_ms_candidate_batch_size(),
        "candidate_batch_size_env": MS_CANDIDATE_BATCH_SIZE_ENV,
        "candidate_count": 0,
        "submitted_candidate_tasks": 0,
        "candidate_result_order": "operator_major_vf_point_major",
    }
    peak = ComponentDVFSConfig(
        policy=DVFSPolicy.NONE,
        voltage_V=0.7,
        frequency_GHz=1.7,
        voltage_regulator_scaling_time_ns=0,
    )
    dvfs_for_eval = DVFSConfig(
        policy=DVFSPolicy.DVFS_C_ms, performance_degradation_percentage=0.0
    )

    def _compute_cfg(pt):
        raw = ComponentDVFSConfig(
            policy=DVFSPolicy.DVFS_C_ms,
            voltage_V=pt.voltage_V,
            frequency_GHz=pt.frequency_GHz,
            voltage_regulator_scaling_time_ns=20,
        )
        plan = {name: raw.model_copy() for name in ("sa", "vu", "sram")}
        couple_compute_domains(plan, "dom3")
        return plan["sa"]

    n_ops = len(ops)
    time_arr = np.zeros((n_ops, K))
    energy_arr = np.zeros((n_ops, K))
    for i, op in enumerate(ops):
        jobs = []
        for pt in SA_POINTS:
            cfg = _compute_cfg(pt)
            _op = op.model_copy()
            _op.stats = op.stats.model_copy()
            _op.dvfs_sa = cfg.model_copy()
            _op.dvfs_vu = cfg.model_copy()
            _op.dvfs_sram = cfg.model_copy()
            _op.dvfs_hbm_mc = peak.model_copy()
            _op.dvfs_hbm_die = peak.model_copy()
            _op.dvfs_hbm_io = peak.model_copy()
            _op.dvfs_ici_mc = peak.model_copy()
            _op.dvfs_ici_phy = peak.model_copy()
            jobs.append(_op)
        evaluated, operator_batching = analyze_operator_energy_candidates(
            jobs,
            config,
            pg_config,
            dvfs_for_eval,
            serial=dvfs_serial,
            remote_batch=remote_candidate_batch,
        )
        candidate_batching["candidate_count"] += operator_batching["candidate_count"]
        candidate_batching["submitted_candidate_tasks"] += operator_batching[
            "submitted_candidate_tasks"
        ]
        for k, evaluated_op in enumerate(evaluated):
            time_arr[i, k] = evaluated_op.stats.execution_time_ns
            energy_arr[i, k] = evaluated_op.stats.total_energy_J

    rot = np.zeros((num_regions, K))
    roe = np.zeros((num_regions, K))
    region_layers = np.zeros(num_regions, dtype=np.int64)
    for r, region in enumerate(regions):
        idxs = list(region.op_indices)
        rot[r] = time_arr[idxs].sum(axis=0)
        roe[r] = energy_arr[idxs].sum(axis=0)
        count_r = region.repeat_count
        assert all(ops[index].stats.count == count_r for index in idxs)
        assert (
            count_r % instance_weight == 0
        ), f"count {count_r} not divisible by iw {instance_weight}"
        region_layers[r] = count_r // instance_weight

    region_ids = np.repeat(np.arange(num_regions), region_layers)
    n_pos = int(region_ids.size)
    rot_pos = rot[region_ids]
    roe_pos = roe[region_ids]
    iw = float(instance_weight)
    orig_time = float(iw * rot_pos[:, PEAK_K].sum())
    base_energy = float(iw * roe_pos[:, PEAK_K].sum())

    zero_per_region = np.full(num_regions, PEAK_K, dtype=np.int32)
    for r in range(num_regions):
        peak_t = rot[r, PEAK_K]
        best_k, best_e = PEAK_K, roe[r, PEAK_K]
        for k in range(K):
            if rot[r, k] <= peak_t and roe[r, k] < best_e:
                best_e, best_k = roe[r, k], k
        zero_per_region[r] = best_k
    zero_degrad = zero_per_region[region_ids]
    pos_by_region = [[] for _ in range(num_regions)]
    for p, r in enumerate(region_ids.tolist()):
        pos_by_region[r].append(p)
    return dict(
        regions=regions,
        num_regions=num_regions,
        region_ids=region_ids,
        n_pos=n_pos,
        rot_pos=rot_pos,
        roe_pos=roe_pos,
        iw=iw,
        K=K,
        PEAK_K=PEAK_K,
        orig_time=orig_time,
        base_energy=base_energy,
        zero_degrad=zero_degrad,
        instance_weight=instance_weight,
        pos_by_region=pos_by_region,
        SA_POINTS=SA_POINTS,
        peak=peak,
        compute_cfg=_compute_cfg,
        candidate_batching=candidate_batching,
    )


def _ms_run_ga(
    pre,
    perf_threshold,
    np_rng,
    population_size=300,
    max_generations=250,
    crossover_prob=0.8,
    mutation_prob=0.05,
    elitism_count=10,
    seed_population=None,
    seed_individual=None,
):
    """Vectorized per-(region,layer) GA over compute V/f. Returns (best_ind, final_pop, T, E).
    Warm-starts from seed_population when given (checkpointing across budgets)."""
    K = pre["K"]
    n_pos = pre["n_pos"]
    rot_pos = pre["rot_pos"]
    roe_pos = pre["roe_pos"]
    iw = pre["iw"]
    orig_time = pre["orig_time"]
    base_energy = pre["base_energy"]
    pos_ax = np.arange(n_pos)[None, :]

    def eval_pop(pop):
        T = iw * rot_pos[pos_ax, pop].sum(axis=1)
        E = iw * roe_pos[pos_ax, pop].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            perf = orig_time / T
            esave = base_energy / E
            score = perf * esave**3
            score = np.where(perf >= perf_threshold, score * 10.0, score)
        score = np.where((T <= 0) | (E <= 0) | ~np.isfinite(score), 0.0, score)
        return score, T, E

    P = population_size
    if seed_population is not None and seed_population.shape == (P, n_pos):
        pop = seed_population.copy().astype(np.int32)
        pop[0] = pre["zero_degrad"]
        if seed_individual is not None:
            pop[1] = seed_individual
    else:
        pop = np_rng.randint(0, K, size=(P, n_pos)).astype(np.int32)
        pop[0] = pre["zero_degrad"]
        pop[1] = pre["PEAK_K"]
        pop[2] = 0
        if seed_individual is not None:
            pop[3] = seed_individual

    best_ind = pop[0].copy()
    best_score, best_T, best_E = -np.inf, orig_time, base_energy
    for _gen in range(max_generations):
        score, T, E = eval_pop(pop)
        bi = int(score.argmax())
        if score[bi] > best_score:
            best_score, best_ind, best_T, best_E = (
                score[bi],
                pop[bi].copy(),
                T[bi],
                E[bi],
            )
        ssum = score.sum()
        probs = score / ssum if ssum > 0 else np.full(P, 1.0 / P)
        new = np.empty_like(pop)
        elite = np.argsort(score)[-elitism_count:]
        new[:elitism_count] = pop[elite]
        n_off = P - elitism_count
        par = np_rng.choice(P, size=(n_off, 2), p=probs)
        child = pop[par[:, 0]].copy()
        p2 = pop[par[:, 1]]
        if n_pos > 1:
            do_cx = np_rng.random(n_off) < crossover_prob
            pts = np_rng.randint(1, n_pos, size=n_off)
            take = (np.arange(n_pos)[None, :] >= pts[:, None]) & do_cx[:, None]
            child = np.where(take, p2, child)
        mut = np_rng.random((n_off, n_pos)) < mutation_prob
        steps = np_rng.choice(np.array([-1, 1]), size=(n_off, n_pos))
        child = np.where(mut, np.clip(child + steps, 0, K - 1), child)
        new[elitism_count:] = child
        pop = new

    score, T, E = eval_pop(pop)
    bi = int(score.argmax())
    if score[bi] > best_score:
        best_score, best_ind, best_T, best_E = score[bi], pop[bi].copy(), T[bi], E[bi]
    return best_ind, pop, float(best_T), float(best_E)


def _ms_apply_back(ops, pre, best_ind):
    """Split each op by its region's per-layer frequencies into output ops (re-analyzed
    per-op downstream, so the time/energy is exact)."""
    regions = pre["regions"]
    pos_by_region = pre["pos_by_region"]
    iw_i = pre["instance_weight"]
    SA_POINTS = pre["SA_POINTS"]
    peak = pre["peak"]
    _compute_cfg = pre["compute_cfg"]
    result = []
    for r, region in enumerate(regions):
        freq_counts = Counter(int(best_ind[p]) for p in pos_by_region[r])
        for oi in region.op_indices:
            op = ops[oi]
            for k, nlayers in sorted(freq_counts.items()):
                cfg = _compute_cfg(SA_POINTS[k])
                no = op.model_copy(deep=True)
                no.stats = op.stats.model_copy(deep=True)
                no.stats.count = nlayers * iw_i
                no.dvfs_sa = cfg.model_copy()
                no.dvfs_vu = cfg.model_copy()
                no.dvfs_sram = cfg.model_copy()
                no.dvfs_hbm_mc = peak.model_copy()
                no.dvfs_hbm_die = peak.model_copy()
                no.dvfs_hbm_io = peak.model_copy()
                no.dvfs_ici_mc = peak.model_copy()
                no.dvfs_ici_phy = peak.model_copy()
                result.append(no)
    return result


def configure_dvfs_c_ms_with_regions(
    ops: list[Operator.Operator],
    config: ModelConfig,
    dvfs_config: DVFSConfig,
    dump_pareto_points_to_file: bool = False,
    population_size: int = 300,
    max_generations: int = 250,
    seed: int = 42,
    pg_config: "str | PowerGatingConfig | None" = None,
) -> list[Operator.Operator]:
    """Single-budget DVFS_C_ms (regions + per-(region,layer) compute-V/f GA). For the
    monotonic, checkpointed multi-budget sweep use configure_dvfs_c_ms_all_budgets."""
    logging.set_verbosity(logging.INFO)
    pre = _ms_precompute(
        ops, config, dvfs_config.frequency_adjustment_interval_ns, pg_config
    )
    np_rng = np.random.RandomState(seed)
    pt = 1.0 / (1.0 + dvfs_config.performance_degradation_percentage)
    best_ind, _pop, bT, bE = _ms_run_ga(
        pre, pt, np_rng, population_size, max_generations
    )
    logging.info(
        "DVFS_C_ms: %d regions %d genes energy=%.4f J (base %.4f) perf_ratio=%.4f budget=%.4f",
        pre["num_regions"],
        pre["n_pos"],
        bE,
        pre["base_energy"],
        pre["orig_time"] / bT if bT > 0 else 0.0,
        dvfs_config.performance_degradation_percentage,
    )
    ops[:] = _ms_apply_back(ops, pre, best_ind)
    return ops


def configure_dvfs_c_ms_all_budgets(
    ops,
    config,
    budgets,
    interval_ns,
    dump_pareto_points_to_file=False,
    pg_config=None,
    population_size=300,
    max_generations=250,
    seed=42,
):
    """Multi-budget DVFS_C_ms WITH checkpointing (like configure_dvfs_c_no_pareto_all_budgets):
    build regions+per-op tables ONCE, sweep budgets ascending, warm-start each budget's GA
    population from the previous budget, and enforce monotonicity (a larger budget never
    yields worse energy). Returns {budget: configured_ops}."""
    logging.set_verbosity(logging.INFO)
    precompute_start_s = time.perf_counter()
    pre = _ms_precompute(ops, config, interval_ns, pg_config)
    precompute_s = time.perf_counter() - precompute_start_s
    np_rng = np.random.RandomState(seed)
    results = {}
    timings = {
        "point_gen_s": precompute_s,
        "ga_s": {},
        "budgets": list(sorted(budgets)),
        "num_regions": pre["num_regions"],
        "num_genes": pre["n_pos"],
        "num_vf_points_per_region": pre["K"],
        "total_s": 0.0,
        **pre["candidate_batching"],
    }
    configure_dvfs_c_ms_all_budgets.last_timings = timings
    best_energy = float("inf")
    best_ind_global = pre["zero_degrad"].copy()
    prev_pop = None
    prev_ind = None
    for budget in sorted(budgets):
        ga_start_s = time.perf_counter()
        pt = 1.0 / (1.0 + budget)
        best_ind, pop, bT, bE = _ms_run_ga(
            pre,
            pt,
            np_rng,
            population_size,
            max_generations,
            seed_population=prev_pop,
            seed_individual=prev_ind,
        )
        timings["ga_s"][budget] = time.perf_counter() - ga_start_s
        prev_pop, prev_ind = pop, best_ind
        if bE < best_energy:
            best_energy = bE
            best_ind_global = best_ind
        logging.info(
            "DVFS_C_ms interval=%.0f budget=%.4f: energy=%.4f J best=%.4f J (regions=%d genes=%d)",
            interval_ns,
            budget,
            bE,
            best_energy,
            pre["num_regions"],
            pre["n_pos"],
        )
        results[budget] = _ms_apply_back(ops, pre, best_ind_global)
    timings["total_s"] = timings["point_gen_s"] + sum(timings["ga_s"].values())
    return results


# Above this expanded-op count, configure_dvfs_for_ops switches the per-op budget
# allocation from the per-instance greedy to the equivalent (much faster) group-level
# allocator. Tunable for testing / verification.
_GROUPED_GREEDY_GATE = 2000


def _build_weighted_pareto_arrays(
    pareto_ops: list[list[Operator.Operator]],
    instance_weight: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extract one weighted energy/time array per original operator."""
    energies = [
        np.array([point.stats.total_energy_J for point in points]) * instance_weight
        for points in pareto_ops
    ]
    times = [
        np.array([point.stats.execution_time_ns for point in points]) * instance_weight
        for points in pareto_ops
    ]
    assert len(energies) == len(pareto_ops) == len(times)
    return energies, times


def _greedy_allocate_grouped(
    ops: list[Operator.Operator],
    instance_weight: int,
    pareto_ops: list[list[Operator.Operator]],
    perf_degradation_percentage: float,
) -> tuple[Counter, int]:
    """Group-level greedy budget allocation for high expanded-op-count workloads.

    Tracks the number of expanded instances at each (op, pareto-index) — O(num_ops *
    num_pareto) state — instead of a per-instance array of size num_expanded, and
    batch-advances all instances that share the chosen move at once. Each step is then
    O(#active groups) rather than O(num_expanded), so MoE workloads with 10k-40k expanded
    instances finish in seconds instead of tens of minutes.

    Equivalence to the per-instance greedy in configure_dvfs_for_ops is EXACT when both of
    these hold (they do for every real NPU trace, and are checked by the guard below):
      * all pareto Δtimes are integers — pareto execution_time_ns is a max of ceil-scaled
        integer component times, so this always holds; it guarantees the batch size
        int(slack // dt) agrees with the per-instance repeated-addition slack test, with no
        floating-point boundary off-by-one, and
      * no front has two chords with an exactly-equal energy/time slope — irregular
        ~17-significant-digit float energies make this a measure-zero coincidence, so real
        fronts are tie-free and strictly convex.
    If either is violated (e.g. a future timing/energy model), batching can strand budget
    slack that the per-instance greedy would instead spend by chaining a single instance
    deeper, diverging by at most ~1 instance per affected group — negligible at 10k-40k
    expanded instances, but up to ~14% on adversarial tiny-count collinear fronts. (Verified
    0.000000% on all real DeepSeek MoE cells; divergence is real only under exact within-front
    derivative ties or non-integer Δtimes — hence the guard below warns rather than silently
    drifting.)

    Returns (group_counts, num_steps) where group_counts maps
    (orig_op_index, pareto_index) -> instance count (in expanded units).
    """
    n = len(ops)
    e = [
        np.array([p.stats.total_energy_J for p in pareto_ops[i]], dtype=float)
        * instance_weight
        for i in range(n)
    ]
    t = [
        np.array([p.stats.execution_time_ns for p in pareto_ops[i]], dtype=float)
        * instance_weight
        for i in range(n)
    ]
    expf = [ops[i].stats.count // instance_weight for i in range(n)]

    # Exactness guard (see docstring): this group-level allocator reproduces the per-instance
    # greedy EXACTLY only when pareto Δtimes are integers (else int(slack // dt) can disagree
    # with the per-instance repeated-addition slack test by a floating-point boundary) and no
    # single front has two chords with an exactly-equal slope (a within-front derivative tie can
    # make batching strand slack the per-instance greedy would chain deeper). Both always hold
    # on real NPU traces (integer ceil-scaled times; irregular-float energies), so this is silent
    # in practice; it exists so a future timing/energy model that breaks the equivalence is
    # caught loudly instead of drifting silently. O(num_ops * num_pareto^2), trivially cheap.
    def _has_repeated_chord_slope(ti: np.ndarray, ei: np.ndarray) -> bool:
        slopes = []
        for a in range(len(ti)):
            for b in range(a + 1, len(ti)):
                d_t = float(ti[b]) - float(ti[a])
                d_e = float(ei[a]) - float(ei[b])
                if d_t > 0.0 and d_e > 0.0:
                    slopes.append(
                        d_e / d_t
                    )  # instance_weight cancels; equals the raw front slope
        return len(set(slopes)) < len(slopes)

    _nonint_time = any(not float(x).is_integer() for ti in t for x in ti)
    _repeated_slope = any(_has_repeated_chord_slope(t[i], e[i]) for i in range(n))
    if _nonint_time or _repeated_slope:
        _reasons = []
        if _nonint_time:
            _reasons.append("non-integer pareto Δtimes")
        if _repeated_slope:
            _reasons.append("repeated within-front chord slope")
        logging.warning(
            "_greedy_allocate_grouped: grouped/per-instance exactness is not guaranteed for this "
            "workload (%s); the selected DVFS allocation may differ from the per-instance greedy "
            "by ~1 instance per affected group. Real NPU traces never trigger this — investigate "
            "the timing/energy model if you see this.",
            "; ".join(_reasons),
        )

    orig_total_time_ns = sum(
        ops[i].stats.execution_time_ns * ops[i].stats.count for i in range(n)
    )
    allowed = orig_total_time_ns * (1.0 + perf_degradation_percentage)

    gc: Counter = Counter()
    total_time = 0.0
    for i in range(n):
        if expf[i] > 0:
            gc[(i, 0)] = expf[i]
            total_time += expf[i] * float(t[i][0])

    steps = 0
    while True:
        slack = allowed - total_time
        if slack <= 0:
            break
        best_deriv = 0.0
        best = None  # (i, k, kp, dt)
        for (i, k), cnt in list(gc.items()):
            if cnt <= 0:
                continue
            ti = t[i]
            ei = e[i]
            tk = float(ti[k])
            ek = float(ei[k])
            bd = 0.0
            bk = -1
            bdt = 0.0
            for kp in range(k + 1, len(ti)):
                dt = float(ti[kp]) - tk
                if dt > slack:
                    break
                de = ek - float(ei[kp])
                if dt > 0.0 and de > 0.0:
                    d = de / dt
                    if d > bd:
                        bd = d
                        bk = kp
                        bdt = dt
            if bk >= 0 and bd > best_deriv:
                best_deriv = bd
                best = (i, k, bk, bdt)
        if best is None:
            break
        i, k, kp, dt = best
        m = min(gc[(i, k)], int(slack // dt))
        if m <= 0:
            m = 1
        gc[(i, k)] -= m
        if gc[(i, k)] <= 0:
            del gc[(i, k)]
        gc[(i, kp)] += m
        total_time += m * dt
        steps += 1

    return gc, steps


def configure_dvfs_for_ops(
    ops: list[Operator.Operator],
    config: ModelConfig,
    dvfs_config: DVFSConfig,
    dump_pareto_points_to_file: bool = False,
    algorithm: str = "auto",
    pg_config: "str | PowerGatingConfig | None" = None,
    timing_result: dict | None = None,
    _precomputed_points: list[list[Operator.Operator]] | None = None,
) -> list[Operator.Operator]:
    """
    Configure DVFS for all operators based on dvfs_config.performance_degradation_percentage.
    We use a greedy algorithm to mimic gradient descent based on d(energy)/d(perf_degrad) for each operator.
    We iteratively apply DVFS to the operator that gives the best energy reduction per performance degradation
    until we reach the target performance degradation slack.

    The performance degradation budget is global: it covers both PG overhead and DVFS slowdown,
    measured against the original raw execution time (no PG, no DVFS).
    """
    # logging.set_verbosity(logging.INFO)

    # if not dump_pareto_points_to_file and dvfs_config.performance_degradation_percentage == 0:
    #     # no need to generate pareto points
    #     return [configure_dvfs_for_op(op, config, dvfs_config) for op in ops]

    # DVFS_C_ms: millisecond-scale compute DVFS. Merge ops into frequency-candidate
    # regions (each >= frequency_adjustment_interval_ns), then run a GA that assigns one
    # compute V/f per region. Runs for any budget (at 0% it still harvests the free slack
    # of memory/ICI-bound regions). CUSTOM_ALL_ms is the per-component (eNPU-ms) analog:
    # regions by bottleneck component, one independent V/f per domain per region.
    if dvfs_config.policy == DVFSPolicy.DVFS_C_ms:
        return configure_dvfs_c_ms_with_regions(
            ops,
            config,
            dvfs_config,
            dump_pareto_points_to_file,
            pg_config=pg_config,
        )
    if dvfs_config.policy == DVFSPolicy.CUSTOM_ALL_ms:
        from neusim.npusim.frontend.dvfs_enpu_ms import configure_enpu_ms_with_regions

        return configure_enpu_ms_with_regions(
            ops,
            config,
            dvfs_config,
            dump_pareto_points_to_file,
            pg_config=pg_config,
            timing_result=timing_result,
        )

    # For DVFS_C / DVFS_C_NO_PARETO with performance degradation, use the GA algorithm
    # instead of the general greedy pareto-based algorithm (compute-only search space).
    if (
        dvfs_config.policy in (DVFSPolicy.DVFS_C, DVFSPolicy.DVFS_C_NO_PARETO)
        and dvfs_config.performance_degradation_percentage > 0
    ):
        if dvfs_config.policy == DVFSPolicy.DVFS_C_NO_PARETO:
            return configure_dvfs_c_with_degradation(
                ops,
                config,
                dvfs_config,
                dump_pareto_points_to_file,
                population_size=400,
                max_generations=600,
                mutation_prob=0.05,
                elitism_count=10,
                pg_config=pg_config,
                timing_result=timing_result,
            )
        return configure_dvfs_c_with_degradation(
            ops,
            config,
            dvfs_config,
            dump_pareto_points_to_file,
            pg_config=pg_config,
            timing_result=timing_result,
        )

    # 1. Generate pareto energy-latency points with a wide per-op budget.
    #    The real global budget is enforced by the greedy loop below, not
    #    by the per-op pareto generation.  Using 100% ensures the pareto
    #    front is not prematurely truncated (same approach as the GA).
    # 1. generate the pareto energy-latency points for each operator
    _t_pareto_start = time.time()
    pareto_ops = (
        _precomputed_points
        if _precomputed_points is not None
        else generate_pareto_energy_latency_points_for_all_ops(
            ops,
            config,
            dvfs_config,
            dump_pareto_points_to_file,
            algorithm=algorithm,
            pg_config=pg_config,
        )
    )
    _t_pareto_end = time.time()

    # if dvfs_config.performance_degradation_percentage == 0:
    #     # no need to perform perf-degrading DVFS
    #     return [configure_dvfs_for_op(op, config, dvfs_config) for op in ops]

    # Detect LLM decode workload by checking op descriptions.
    # For decode, count = num_layers * output_seqlen, but all tokens share
    # the same computation graph, so the correct expansion granularity is
    # per-layer (count / output_seqlen), with each instance weighted by
    # output_seqlen.
    has_decode = any("decode" in op.description.lower() for op in ops)
    has_prefill = any("prefill" in op.description.lower() for op in ops)
    if has_decode and has_prefill:
        raise ValueError(
            "Ops list contains both decode and prefill operators. "
            "Mixed decode/prefill is not supported for DVFS expansion."
        )

    is_llm_decode = has_decode and not has_prefill
    if is_llm_decode:
        from neusim.configs.models.LLMConfig import LLMConfig

        assert isinstance(
            config, LLMConfig
        ), f"LLM decode ops detected but config is {type(config).__name__}, not LLMConfig"
        instance_weight = config.output_seqlen
    else:
        instance_weight = 1

    # 2. greedy selection based on d(energy)/d(perf_degrad)
    # Expand deduplicated ops for finer-grained greedy allocation.
    # Each expanded instance represents `instance_weight` actual executions.
    # For decode: expand by num_layers (= count / output_seqlen) per op.
    # For non-decode: expand by count (same as before, instance_weight=1).
    expanded_indices: list[int] = []  # expanded index -> original op index
    for i, op in enumerate(ops):
        assert (
            op.stats.count % instance_weight == 0
        ), f"Op '{op.description}' count {op.stats.count} is not divisible by instance_weight {instance_weight}"
        expansion_factor = op.stats.count // instance_weight
        expanded_indices.extend([i] * expansion_factor)
    num_expanded = len(expanded_indices)

    # Pre-extract pareto energy/time into numpy arrays to avoid repeated
    # attribute access in the hot loop. Deduplicate across expanded instances
    # that share the same original op.
    assert len(pareto_ops) == len(ops)
    orig_pareto_energies, orig_pareto_times = _build_weighted_pareto_arrays(
        pareto_ops, instance_weight
    )
    pareto_energies = [
        orig_pareto_energies[expanded_indices[j]] for j in range(num_expanded)
    ]
    pareto_times = [orig_pareto_times[expanded_indices[j]] for j in range(num_expanded)]

    # 3. Greedy inter-op budget allocation.
    #    Start from the fastest Pareto point (index 0) for every expanded op instance
    #    — this is the 0% perf-degradation point (baseline speed, lowest energy at that speed).
    #    Then greedily spend the available performance slack by moving individual ops
    #    to slower Pareto points that save energy, picking the move with the best
    #    energy-saved / time-cost ratio (steepest descent) at each step.
    _t_search_start = time.time()
    _search_iterations = 0
    # For very large (high expanded-op-count) workloads the per-instance greedy below
    # is ~O(num_expanded^2); use the equivalent group-level allocator instead.
    _use_grouped = num_expanded > _GROUPED_GREEDY_GATE
    selected_ops_indices = [
        0
    ] * num_expanded  # start from the fastest pareto point (0% degrad)

    def _get_next_valid_indices_for_all_ops(
        max_delta_ns: int | float,
    ) -> tuple[list[list[int]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return valid next indices and candidate/current arrays for each expanded op.

        Deduplicates computation: expanded instances from the same original op at the
        same pareto index share identical valid indices and energy/time values.

        Return:
        - next_indices: list of list of valid indices for each op; shape: (num_expanded, variable)
        - next_energy: candidate energy array; shape: (num_expanded, max_valid)
        - next_exe_time: candidate exec time array; shape: (num_expanded, max_valid)
        - curr_energy: current energy for each expanded op; shape: (num_expanded,)
        - curr_exe_time: current exec time for each expanded op; shape: (num_expanded,)
        """
        # Deduplicate: compute valid indices once per unique (orig_op, pareto_idx)
        group_cache: dict[tuple[int, int], list[int]] = {}
        next_indices: list[list[int]] = [None] * num_expanded  # type: ignore[list-item]

        for j in range(num_expanded):
            key = (expanded_indices[j], selected_ops_indices[j])
            if key not in group_cache:
                curr_index = selected_ops_indices[j]
                curr_time = pareto_times[j][curr_index]
                valid_indices: list[int] = []
                times_arr = pareto_times[j]
                for k in range(curr_index + 1, len(times_arr)):
                    if times_arr[k] - curr_time <= max_delta_ns:
                        valid_indices.append(k)
                    else:
                        break
                group_cache[key] = valid_indices
            next_indices[j] = group_cache[key]

        # Build reverse mapping: group key -> list of expanded row indices
        group_rows: dict[tuple[int, int], list[int]] = {}
        for j in range(num_expanded):
            key = (expanded_indices[j], selected_ops_indices[j])
            group_rows.setdefault(key, []).append(j)

        # Compute current energy/time per group, broadcast to all matching rows
        curr_energy = np.empty(num_expanded)
        curr_exe_time = np.empty(num_expanded)
        for rows in group_rows.values():
            first_j = rows[0]
            idx = selected_ops_indices[first_j]
            e_val = pareto_energies[first_j][idx]
            t_val = pareto_times[first_j][idx]
            for r in rows:
                curr_energy[r] = e_val
                curr_exe_time[r] = t_val

        max_len = max((len(indices) for indices in next_indices), default=0)
        if max_len == 0:
            return (
                next_indices,
                np.empty((num_expanded, 0)),
                np.empty((num_expanded, 0)),
                curr_energy,
                curr_exe_time,
            )

        next_energy_arr = np.full((num_expanded, max_len), np.inf)
        next_exe_time_arr = np.full((num_expanded, max_len), np.inf)

        for rows in group_rows.values():
            first_j = rows[0]
            indices = next_indices[first_j]
            if not indices:
                continue
            e_row = pareto_energies[first_j][indices]
            t_row = pareto_times[first_j][indices]
            n = len(indices)
            for r in rows:
                next_energy_arr[r, :n] = e_row
                next_exe_time_arr[r, :n] = t_row

        return (
            next_indices,
            next_energy_arr,
            next_exe_time_arr,
            curr_energy,
            curr_exe_time,
        )

    orig_total_time_ns = sum(op.stats.execution_time_ns * op.stats.count for op in ops)
    total_allowed_time_ns = orig_total_time_ns * (
        1 + dvfs_config.performance_degradation_percentage
    )

    # Incremental total time tracking (avoid O(num_expanded) recomputation each iteration)
    total_time_ns = sum(pareto_times[j][0] for j in range(num_expanded))

    # Track greedy search trace: absolute (total_time_ns, total_energy_J) at each step.
    # Recording every step calls _current_energy() (an O(num_expanded) sum) per
    # iteration, which is O(num_expanded^2) overall and dominates the run time for
    # large MoE workloads (num_expanded ~ 10k-40k). The trace is visualization-only,
    # so for large problems we skip the per-step record (keeping only the endpoints).
    # This does not affect the selected DVFS configuration in any way.
    _record_trace = num_expanded <= 2000

    def _current_energy():
        return sum(
            pareto_energies[j][selected_ops_indices[j]] for j in range(num_expanded)
        )

    greedy_trace = [
        {
            "step": 0,
            "total_time_ns": float(total_time_ns),
            "total_energy_J": float(_current_energy()),
        }
    ]

    while not _use_grouped:
        remaining_slack_ns = total_allowed_time_ns - total_time_ns
        if remaining_slack_ns <= 0:
            break
        _search_iterations += 1

        # get next valid plans and their energy and execution time for each op
        (
            next_indices,  # shape: (num_expanded, variable)
            next_energy,  # shape: (num_expanded, max_valid)
            next_exe_time_ns,  # shape: (num_expanded, max_valid)
            energy_array,  # shape: (num_expanded,)
            exe_time_array,  # shape: (num_expanded,)
        ) = _get_next_valid_indices_for_all_ops(remaining_slack_ns)

        # Compute energy_saved / time_cost for each candidate move.
        # We are moving from the current (fast) point to a slower point that saves energy.
        # delta_exe_time > 0 means the candidate is slower (costs performance slack).
        # delta_energy > 0 means the candidate saves energy.
        delta_exe_time_ns = (
            next_exe_time_ns - exe_time_array[:, np.newaxis]
        )  # shape: (num_expanded, max_valid)
        delta_energy = (
            energy_array[:, np.newaxis] - next_energy
        )  # shape: (num_expanded, max_valid)
        with np.errstate(divide="ignore", invalid="ignore"):
            derivs = np.where(
                (delta_exe_time_ns > 0) & (delta_energy > 0),
                delta_energy / delta_exe_time_ns,
                -np.inf,
            )  # shape: (num_expanded, max_valid)

        # Greedy: pick the move with the highest energy_saved / time_cost ratio
        # (steepest descent — most energy saved per unit of performance slack spent).
        if derivs.size == 0:
            break
        flat_idx = np.argmax(derivs)
        best_deriv = derivs.flat[flat_idx]
        if best_deriv == -np.inf:
            # no more valid plans to apply DVFS
            break
        op_to_update, plan_to_apply_idx = np.unravel_index(flat_idx, derivs.shape)

        old_idx = selected_ops_indices[op_to_update]
        new_idx = next_indices[op_to_update][plan_to_apply_idx]
        selected_ops_indices[op_to_update] = new_idx
        total_time_ns += (
            pareto_times[op_to_update][new_idx] - pareto_times[op_to_update][old_idx]
        )

        if _record_trace:
            greedy_trace.append(
                {
                    "step": len(greedy_trace),
                    "total_time_ns": float(total_time_ns),
                    "total_energy_J": float(_current_energy()),
                }
            )

        logging.debug(
            "Search step: remaining_slack_ns=%s, best_deriv=%s (energy/time), updated_op_index=%s, updated_op=%s",
            remaining_slack_ns,
            best_deriv,
            expanded_indices[op_to_update],
            ops[expanded_indices[op_to_update]].name,
        )

    # 4. Group expanded instances by (original_op_index, selected_pareto_index)
    # and assign the selected DVFS config to each group.
    # The grouping is for merging ops with the same op config and DVFS config.
    if _use_grouped:
        group_counts, _search_iterations = _greedy_allocate_grouped(
            ops,
            instance_weight,
            pareto_ops,
            dvfs_config.performance_degradation_percentage,
        )
    else:
        group_counts = Counter()
        for j, orig_idx in enumerate(expanded_indices):
            group_counts[(orig_idx, selected_ops_indices[j])] += 1

    result_ops = []
    for (orig_idx, pareto_idx), cnt in sorted(group_counts.items()):
        new_op = ops[orig_idx].model_copy(deep=True)
        new_op.stats.count = cnt * instance_weight
        selected = pareto_ops[orig_idx][pareto_idx]
        new_op.dvfs_sa = selected.dvfs_sa.model_copy()
        new_op.dvfs_vu = selected.dvfs_vu.model_copy()
        new_op.dvfs_sram = selected.dvfs_sram.model_copy()
        new_op.dvfs_hbm_mc = selected.dvfs_hbm_mc.model_copy()
        new_op.dvfs_hbm_die = selected.dvfs_hbm_die.model_copy()
        new_op.dvfs_hbm_io = selected.dvfs_hbm_io.model_copy()
        new_op.dvfs_ici_mc = selected.dvfs_ici_mc.model_copy()
        new_op.dvfs_ici_phy = selected.dvfs_ici_phy.model_copy()
        result_ops.append(new_op)

    # Collapse SA/VU/SRAM into shared V/f domains for CUSTOM/CUSTOM_ALL when the
    # policy asks for fewer than 5 domains. The pareto search above optimizes the
    # compute/memory domains independently (= 5 domains); this post-hoc coupling
    # merges the compute components onto the fastest-needed member, which is the
    # coupled-optimal operating point (see couple_compute_domains). HBM/ICI are
    # untouched. Energy is (re)computed from these configs by the caller.
    if dvfs_config.policy in (DVFSPolicy.CUSTOM, DVFSPolicy.CUSTOM_ALL):
        _mode = getattr(dvfs_config, "custom_compute_domain_mode", "dom5")
        if (_mode or "dom5").lower() != "dom5":
            from neusim.npusim.backend.dvfs_custom_policy import couple_compute_domains

            for nop in result_ops:
                _plan = {"sa": nop.dvfs_sa, "vu": nop.dvfs_vu, "sram": nop.dvfs_sram}
                couple_compute_domains(_plan, _mode)
                nop.dvfs_sa = _plan["sa"]
                nop.dvfs_vu = _plan["vu"]
                nop.dvfs_sram = _plan["sram"]

    # Save greedy search trace for visualization
    _save_search_trace(
        config,
        dvfs_config,
        ops,
        {
            "algorithm": "greedy",
            "policy": str(dvfs_config.policy.value),
            "perf_degrad_pct": float(dvfs_config.performance_degradation_percentage),
            "orig_total_time_ns": float(orig_total_time_ns),
            "trace": greedy_trace,
        },
    )

    _t_search_end = time.time()
    if timing_result is not None:
        timing_result["pareto_generation_seconds"] = round(
            _t_pareto_end - _t_pareto_start, 4
        )
        if _precomputed_points is None:
            timing_result["pareto_generation"] = deepcopy(
                getattr(
                    generate_pareto_energy_latency_points_for_all_ops,
                    "last_run_stats",
                    {},
                )
            )
        timing_result["inter_op_search_seconds"] = round(
            _t_search_end - _t_search_start, 4
        )
        timing_result["avg_pareto_points"] = round(
            sum(len(p) for p in pareto_ops) / len(pareto_ops), 2
        )
        timing_result["num_expanded_ops"] = num_expanded
        timing_result["inter_op_search_iterations"] = _search_iterations

    ops[:] = result_ops
    return ops


def configure_dvfs_for_ops_all_budgets(
    ops: list[Operator.Operator],
    config: ModelConfig,
    dvfs_config: DVFSConfig,
    budgets: list[float],
    dump_pareto_points_to_file: bool = False,
    algorithm: str = "auto",
    pg_config: "str | PowerGatingConfig | None" = None,
) -> dict[float, list[Operator.Operator]]:
    """Configure a generic request-level policy for several slowdown budgets.

    Candidate Pareto points dominate the cost of CUSTOM, CUSTOM_ALL, and IDEAL
    request searches.  This entry point builds one budget-independent 100%
    candidate envelope, then reuses it for every requested global budget.  The
    per-budget allocator remains :func:`configure_dvfs_for_ops`, but candidate
    filtering/reduction is not rebuilt at each budget.  This is an explicit
    artifact batching heuristic, not independent-search equivalence.

    ``budgets`` are fractions (``0.2`` means 20%).  DVFS-C and millisecond
    policies have dedicated all-budget implementations and are rejected here.
    Timing details are published through ``last_timings`` for artifact drivers.
    """
    if not ops:
        raise ValueError("ops must contain at least one operator")
    if not budgets:
        raise ValueError("budgets must contain at least one value")
    sorted_budgets = sorted({float(value) for value in budgets})
    if any(not np.isfinite(value) or value < 0 for value in sorted_budgets):
        raise ValueError("budgets must be finite, non-negative fractions")

    supported = {
        DVFSPolicy.CUSTOM,
        DVFSPolicy.CUSTOM_ALL,
        DVFSPolicy.IDEAL,
    }
    if dvfs_config.policy not in supported:
        raise ValueError(
            "generic all-budget search supports CUSTOM, CUSTOM_ALL, and IDEAL; "
            f"got {dvfs_config.policy.value!r}"
        )

    plan_timings: dict = {
        "point_gen_s": 0.0,
        "search_s": {},
        "search_details": {},
        "budgets": list(sorted_budgets),
        "num_ops": len(ops),
        "avg_pareto_points": 0.0,
        "total_s": 0.0,
        "algorithm": "shared_100pct_candidate_envelope_heuristic",
        "candidate_envelope_fraction": 1.0,
        "candidate_envelope_semantics": (
            "one 100%-slowdown candidate set shared across all request budgets"
        ),
        "candidate_set_shared_across_budgets": True,
        "independent_per_budget_candidate_semantics_preserved": False,
        "budget_dependent_candidate_reduction_preserved": False,
        "shared_envelope_caveat": (
            "candidate filtering/reduction is performed once at the 100% "
            "envelope rather than independently for each budget"
        ),
        "ideal_shared_envelope_caveat": (
            "Ideal reduction is applied once at the 100% envelope, not independently per budget"
        )
        if dvfs_config.policy == DVFSPolicy.IDEAL
        else None,
    }
    configure_dvfs_for_ops_all_budgets.last_timings = plan_timings

    # A common 100% per-operator envelope is broad enough for every paper
    # budget and prevents the candidate set from changing between thresholds.
    envelope = max(1.0, max(sorted_budgets))
    envelope_config = dvfs_config.model_copy(
        update={"performance_degradation_percentage": envelope}
    )
    point_start = time.perf_counter()
    pareto_ops = generate_pareto_energy_latency_points_for_all_ops(
        ops,
        config,
        envelope_config,
        dump_pareto_points_to_file,
        algorithm=algorithm,
        pg_config=pg_config,
    )
    plan_timings["point_gen_s"] = time.perf_counter() - point_start
    plan_timings["pareto_generation"] = deepcopy(
        getattr(
            generate_pareto_energy_latency_points_for_all_ops,
            "last_run_stats",
            {},
        )
    )
    plan_timings["avg_pareto_points"] = sum(len(points) for points in pareto_ops) / len(
        pareto_ops
    )

    results: dict[float, list[Operator.Operator]] = {}
    for budget in sorted_budgets:
        step_config = dvfs_config.model_copy(
            update={"performance_degradation_percentage": budget}
        )
        step_timing: dict = {}
        search_start = time.perf_counter()
        configured = configure_dvfs_for_ops(
            [op.model_copy(deep=True) for op in ops],
            config,
            step_config,
            dump_pareto_points_to_file,
            algorithm=algorithm,
            pg_config=pg_config,
            timing_result=step_timing,
            _precomputed_points=pareto_ops,
        )
        plan_timings["search_s"][budget] = time.perf_counter() - search_start
        plan_timings["search_details"][budget] = step_timing
        results[budget] = [op.model_copy(deep=True) for op in configured]

    plan_timings["total_s"] = plan_timings["point_gen_s"] + sum(
        plan_timings["search_s"].values()
    )
    return results
