"""DVFS-aware scheduling for a statically allocated inference fleet.

The scheduler runs after FleetSim forms a batch. It converts request SLO slack
into a performance-degradation budget, applies the two paper safeguards, and
selects the lowest-energy feasible point. The vPod allocation never changes.
"""

import glob
import json
import math
import os
from collections import deque
from collections.abc import Callable, Sequence
from functools import cache
from typing import Any

from absl import logging

from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.fleetsim.npusim_backend_interface import (
    FrozenDeepSeekConfig,
    FrozenLLMConfig,
    PhaseMetrics,
)
from neusim.npusim.frontend.llm_ops_generator import (
    DeepSeekOpsGenerator,
    LLMOpsGenerator,
)
from neusim.npusim.frontend.Operator import DVFSPolicy
from neusim.npusim.frontend.power_analysis_lib import analyze_all_operator_energy

FrozenConfig = FrozenLLMConfig | FrozenDeepSeekConfig
LookupKey = tuple[str, int, int, str, str, int]
LookupPoint = tuple[int, float]
LookupSeries = tuple[tuple[float, int, float], ...]


def quantize_perf_degrad(value: float, step: float = 0.01) -> float:
    """Round a degradation ratio down to the cache granularity."""
    return math.floor(value / step) * step


def _dvfs_policy_from_str(name: str) -> DVFSPolicy:
    """Convert the FleetSim policy spelling to NeuSim's policy enum."""
    try:
        return DVFSPolicy.from_str(name)
    except ValueError as exc:
        options = [policy.value for policy in DVFSPolicy]
        raise ValueError(
            f"Unknown DVFS policy {name!r}; choose from {options}"
        ) from exc


# Key: (model_name, input_seqlen, output_seqlen, version, phase, batch_size)
# Value: sorted (perf_degrad, time_ns_per_stage, energy_J_per_chip) points.
_dvfs_lookup_cache: dict[LookupKey, LookupSeries] = {}
_dvfs_cache_hits = 0
_dvfs_cache_misses = 0
_dvfs_plans_applied = 0
_dvfs_plans_rejected_nonbeneficial = 0


def reset_dvfs_lookup_cache() -> None:
    """Clear service-level lookup state (primarily useful between tests/runs)."""
    global _dvfs_cache_hits, _dvfs_cache_misses
    global _dvfs_plans_applied, _dvfs_plans_rejected_nonbeneficial
    _dvfs_lookup_cache.clear()
    _dvfs_cache_hits = 0
    _dvfs_cache_misses = 0
    _dvfs_plans_applied = 0
    _dvfs_plans_rejected_nonbeneficial = 0


def get_dvfs_lookup_stats() -> dict[str, int]:
    """Return lookup and applied-plan counters for strict-run audits."""
    return {
        "entries": len(_dvfs_lookup_cache),
        "points": sum(len(points) for points in _dvfs_lookup_cache.values()),
        "hits": _dvfs_cache_hits,
        "misses": _dvfs_cache_misses,
        "plans_applied": _dvfs_plans_applied,
        "plans_rejected_nonbeneficial": _dvfs_plans_rejected_nonbeneficial,
    }


def load_dvfs_lookup_cache(cache_dir: str, *, strict: bool = False) -> int:
    """Load per-point or grouped DVFS JSON files from a policy directory.

    Returns the number of performance-degradation points loaded. ``strict``
    turns absent directories, malformed files, and empty caches into errors so
    an AE run cannot silently consume an online or peak-frequency fallback.
    """
    reset_dvfs_lookup_cache()
    if not cache_dir or not os.path.isdir(cache_dir):
        message = f"DVFS lookup cache directory not found: {cache_dir}"
        if strict:
            raise FileNotFoundError(message)
        logging.warning("%s", message)
        return 0

    pattern = os.path.join(cache_dir, "*", "*", "*", "*", "bs*.json")
    files = sorted(glob.glob(pattern))
    loaded_files = 0
    loaded: dict[LookupKey, dict[float, LookupPoint]] = {}
    for path in files:
        try:
            with open(path) as cache_file:
                data = json.load(cache_file)
            if "points" in data:
                metadata = data["metadata"]
                key: LookupKey = (
                    str(metadata["model"]),
                    int(metadata["input_seqlen"]),
                    int(metadata["output_seqlen"]),
                    str(metadata["version"]),
                    str(metadata["phase"]),
                    int(metadata["batch_size"]),
                )
                points = loaded.setdefault(key, {})
                for degrad, values in data["points"].items():
                    points[float(degrad)] = (
                        int(values["time_ns_per_stage"]),
                        float(values["energy_J_per_chip"]),
                    )
            else:
                key = (
                    str(data["model"]),
                    int(data["input_seqlen"]),
                    int(data["output_seqlen"]),
                    str(data["version"]),
                    str(data["phase"]),
                    int(data["batch_size"]),
                )
                loaded.setdefault(key, {})[float(data["perf_degrad"])] = (
                    int(data["time_ns_per_stage"]),
                    float(data["energy_J_per_chip"]),
                )
            loaded_files += 1
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if strict:
                reset_dvfs_lookup_cache()
                raise ValueError(
                    f"Invalid DVFS lookup cache file {path}: {exc}"
                ) from exc
            logging.warning("Failed to load DVFS cache file %s: %s", path, exc)

    for key, points in loaded.items():
        _dvfs_lookup_cache[key] = tuple(
            (degrad, values[0], values[1]) for degrad, values in sorted(points.items())
        )
    num_points = sum(len(points) for points in _dvfs_lookup_cache.values())
    if strict and num_points == 0:
        raise ValueError(f"DVFS lookup cache contains no points: {cache_dir}")
    logging.info(
        "Loaded DVFS lookup cache: %d entries (%d points) from %d files in %s",
        len(_dvfs_lookup_cache),
        num_points,
        loaded_files,
        cache_dir,
    )
    return num_points


def _lookup_dvfs_energy(
    model_name: str,
    input_seqlen: int,
    output_seqlen: int,
    version: str,
    phase: str,
    batch_size: int,
    max_perf_degrad: float,
    num_pipeline_stages: int,
    t_target_ns: int,
) -> tuple[int, float, float] | None:
    """Return the lowest-energy cached point that meets the time budget."""
    global _dvfs_cache_hits, _dvfs_cache_misses
    key: LookupKey = (
        model_name,
        input_seqlen,
        output_seqlen,
        version,
        phase,
        batch_size,
    )
    points = _dvfs_lookup_cache.get(key)
    if points is None:
        _dvfs_cache_misses += 1
        return None

    best_time: int | None = None
    best_energy = float("inf")
    best_degrad = 0.0
    for degrad, time_per_stage, energy_per_chip in points:
        if degrad > max_perf_degrad:
            break
        if time_per_stage * num_pipeline_stages > t_target_ns:
            continue
        if energy_per_chip < best_energy:
            best_time = time_per_stage
            best_energy = energy_per_chip
            best_degrad = degrad

    if best_time is None:
        baseline = next((point for point in points if point[0] == 0.0), None)
        if baseline is None:
            _dvfs_cache_misses += 1
            return None
        _dvfs_cache_hits += 1
        return baseline[1], baseline[2], 0.0
    _dvfs_cache_hits += 1
    return best_time, best_energy, best_degrad


@cache
def _compute_dvfs_at_degrad(
    frozen_config: FrozenConfig,
    perf_degrad: float,
    dvfs_policy: DVFSPolicy,
    phase: str,
) -> tuple[int, float]:
    """Run NeuSim's batch-level DVFS algorithm at one degradation point."""
    if isinstance(frozen_config, FrozenDeepSeekConfig):
        config: LLMConfig | DeepSeekConfig = DeepSeekConfig(
            **frozen_config.model_dump()
        )
        generator = DeepSeekOpsGenerator(config)
    else:
        config = LLMConfig(**frozen_config.model_dump())
        generator = LLMOpsGenerator(config)

    if config.enable_dvfs is not True:
        raise ValueError("service DVFS requires a detailed-power FleetSim baseline")

    generated = generator.generate(
        dump_to_file=False,
        separate_prefill_decode=True,
        analyze_energy=True,
    )
    if not isinstance(generated, tuple) or len(generated) != 3:
        raise TypeError("NeuSim LLM generator did not return split operator lists")
    _, prefill_ops, decode_ops = generated
    if not isinstance(prefill_ops, list) or not isinstance(decode_ops, list):
        raise TypeError("NeuSim LLM generator returned non-list operator collections")

    for op in decode_ops:
        count = op.stats.count
        if count < config.output_seqlen or count % config.output_seqlen:
            raise ValueError(
                f"decode op count {count} must be a positive multiple of "
                f"output_seqlen {config.output_seqlen}; op={op}"
            )
        op.stats.count = count // config.output_seqlen

    ops = prefill_ops if phase == "prefill" else decode_ops
    if phase == "decode":
        # Decode counts above are per iteration. Match that unit in the
        # optimizer's instance weighting, as in the original Figure 22 code.
        config.output_seqlen = 1
    elif phase != "prefill":
        raise ValueError(f"Unknown inference phase: {phase}")

    dvfs_config = (
        f"{dvfs_policy.value}_{perf_degrad}" if perf_degrad > 0 else dvfs_policy.value
    )
    analyze_all_operator_energy(
        ops,
        config,
        pg_config=None,
        dvfs_config=dvfs_config,
    )
    return (
        sum(op.stats.execution_time_ns * op.stats.count for op in ops),
        sum(op.stats.total_energy_J * op.stats.count for op in ops),
    )


_SWEEP_STEPS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0)


@cache
def _cached_dvfs_energy(
    frozen_config: FrozenConfig,
    max_perf_degrad: float,
    dvfs_policy: DVFSPolicy,
    phase: str,
    t_target_ns: int,
    num_pipeline_stages: int,
) -> tuple[int, float, float]:
    """Find the best online-computed point within the SLO budget."""
    best_time: int | None = None
    best_energy = float("inf")
    best_degrad = 0.0
    for step in _SWEEP_STEPS:
        if step > max_perf_degrad:
            break
        q_step = quantize_perf_degrad(step)
        time_per_stage, energy_per_chip = _compute_dvfs_at_degrad(
            frozen_config, q_step, dvfs_policy, phase
        )
        if time_per_stage * num_pipeline_stages > t_target_ns:
            continue
        if energy_per_chip < best_energy:
            best_time = time_per_stage
            best_energy = energy_per_chip
            best_degrad = q_step

    if best_time is None:
        best_time, best_energy = _compute_dvfs_at_degrad(
            frozen_config, 0.0, dvfs_policy, phase
        )
    return best_time, best_energy, best_degrad


def compute_t_target_prefill(
    requests: Sequence[LLMRequest],
    event_timestamp: int,
    assign_ttft_target: Callable[[int], float],
) -> int:
    """Return the tightest TTFT execution-time budget in a prefill batch."""
    min_target = float("inf")
    for request in requests:
        queuing_delay_ns = event_timestamp - request.enqueue_timestamp
        ttft_target_ns = int(assign_ttft_target(request.input_seqlen) * 1e9)
        min_target = min(min_target, ttft_target_ns - queuing_delay_ns)
    return int(min_target)


def compute_t_target_decode(
    requests: Sequence[LLMRequest],
    event_timestamp: int,
    assign_tpot_target: Callable[[int], float],
    num_iterations: int,
) -> int:
    """Return the tightest proportional TPOT budget in a decode batch."""
    min_target = float("inf")
    for request in requests:
        tpot_target_ns = int(
            assign_tpot_target(request.input_seqlen + request.output_seqlen) * 1e6
        )
        remaining_tokens = max(1, request.output_seqlen - request.current_decode_step)
        if request.current_decode_step <= 1:
            queuing_delay_ns = event_timestamp - request.prefill_end_timestamp
            total_budget_ns = tpot_target_ns * remaining_tokens - queuing_delay_ns
        else:
            total_budget_ns = tpot_target_ns * remaining_tokens
        min_target = min(
            min_target,
            int(total_budget_ns * (num_iterations / remaining_tokens)),
        )
    return int(min_target)


def apply_safeguard1(
    t_target: int,
    baseline_time_ns: int,
    request_queue: deque[LLMRequest],
    event_timestamp: int,
    assign_slo_fn: Callable[[int], float],
    prefill_or_decode: str,
    slo_unit_to_ns: float,
) -> int:
    """Protect the deadlines of the first 100 requests behind this batch."""
    check_limit = min(100, len(request_queue))
    for index, request in enumerate(request_queue):
        if index >= check_limit:
            break
        if prefill_or_decode == "prefill":
            slo_budget_ns = int(assign_slo_fn(request.input_seqlen) * slo_unit_to_ns)
            current_wait_ns = event_timestamp - request.enqueue_timestamp
        else:
            slo_budget_ns = int(
                assign_slo_fn(request.input_seqlen + request.output_seqlen)
                * slo_unit_to_ns
            )
            remaining_tokens = max(
                1, request.output_seqlen - request.current_decode_step
            )
            slo_budget_ns *= remaining_tokens
            if request.current_decode_step <= 1 and request.prefill_end_timestamp > 0:
                current_wait_ns = event_timestamp - request.prefill_end_timestamp
            else:
                current_wait_ns = 0

        max_allowed = slo_budget_ns - current_wait_ns - baseline_time_ns
        if max_allowed < baseline_time_ns:
            max_allowed = baseline_time_ns
        if max_allowed < t_target:
            t_target = max_allowed
    return max(t_target, baseline_time_ns)


def compute_dvfs_plan_for_batch(
    phase_metrics: PhaseMetrics,
    frozen_config: FrozenConfig,
    requests: Sequence[LLMRequest],
    event_timestamp: int,
    baseline_time_ns: int,
    num_pipeline_stages: int,
    num_chips: int,
    prefill_or_decode: str,
    num_iterations: int,
    metrics_server: Any,
    request_queue: deque[LLMRequest],
    workload_config: Any,
) -> tuple[int, float, float]:
    """Return ``(time_ns, energy_J, degradation)`` for one formed batch."""
    global _dvfs_plans_applied, _dvfs_plans_rejected_nonbeneficial

    baseline_energy_J = phase_metrics.energy_per_chip_J * num_chips
    if not metrics_server.slo_targets or metrics_server.dvfs_locked_to_peak:
        return baseline_time_ns, baseline_energy_J, 0.0

    if prefill_or_decode == "prefill":
        t_target = compute_t_target_prefill(
            requests, event_timestamp, metrics_server.assign_ttft_target
        )
        t_target = apply_safeguard1(
            t_target,
            baseline_time_ns,
            request_queue,
            event_timestamp,
            metrics_server.assign_ttft_target,
            "prefill",
            1e9,
        )
    elif prefill_or_decode == "decode":
        t_target = compute_t_target_decode(
            requests,
            event_timestamp,
            metrics_server.assign_tpot_target,
            num_iterations,
        )
        t_target = apply_safeguard1(
            t_target,
            baseline_time_ns,
            request_queue,
            event_timestamp,
            metrics_server.assign_tpot_target,
            "decode",
            1e6,
        )
    else:
        raise ValueError(f"Unknown inference phase: {prefill_or_decode}")

    if t_target <= baseline_time_ns:
        return baseline_time_ns, baseline_energy_J, 0.0

    max_perf_degrad = quantize_perf_degrad(
        min(
            (t_target - baseline_time_ns) / baseline_time_ns,
            workload_config.dvfs_max_perf_degrad,
        )
    )
    cache_result = _lookup_dvfs_energy(
        model_name=frozen_config.model_name,
        input_seqlen=frozen_config.input_seqlen,
        output_seqlen=frozen_config.output_seqlen,
        version=frozen_config.name,
        phase=prefill_or_decode,
        batch_size=frozen_config.microbatch_size_ici,
        max_perf_degrad=max_perf_degrad,
        num_pipeline_stages=num_pipeline_stages,
        t_target_ns=t_target,
    )
    if cache_result is not None:
        time_per_stage, energy_per_chip, best_degrad = cache_result
        dvfs_time_ns = time_per_stage * num_pipeline_stages
        dvfs_energy_J = energy_per_chip * num_chips
    else:
        if workload_config.dvfs_require_cache_hit:
            key = (
                frozen_config.model_name,
                frozen_config.input_seqlen,
                frozen_config.output_seqlen,
                frozen_config.name,
                prefill_or_decode,
                frozen_config.microbatch_size_ici,
            )
            raise KeyError(f"Required DVFS lookup cache point is missing: {key}")
        policy = _dvfs_policy_from_str(workload_config.dvfs_policy)
        time_per_stage, energy_per_chip, best_degrad = _cached_dvfs_energy(
            frozen_config,
            max_perf_degrad,
            policy,
            prefill_or_decode,
            t_target,
            num_pipeline_stages,
        )
        dvfs_time_ns = time_per_stage * num_pipeline_stages
        dvfs_energy_J = energy_per_chip * num_chips

    if dvfs_energy_J >= baseline_energy_J:
        _dvfs_plans_rejected_nonbeneficial += 1
        return baseline_time_ns, baseline_energy_J, 0.0

    _dvfs_plans_applied += 1
    logging.debug(
        "DVFS %s: baseline=%d ns, target=%d ns, selected=%d ns, "
        "baseline=%.4f J, selected=%.4f J, degradation=%.2f",
        prefill_or_decode,
        baseline_time_ns,
        t_target,
        dvfs_time_ns,
        baseline_energy_J,
        dvfs_energy_J,
        best_degrad,
    )
    return dvfs_time_ns, dvfs_energy_J, best_degrad
