"""Authoritative trace-util eNPU-ms policy port.

The implementation comes from trace-util commit
8ad6961b2a266e91ebb1162c7e2c5df61d10b1a4. It assigns one five-domain
configuration to every operator and every execution represented by a
component-labelled millisecond region. NeuSim compatibility adds validation,
serial execution for tests, hard slowdown feasibility, exact power-gating
aware remeasurement, and auditable timing metadata.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
from absl import logging

from neusim.npusim.frontend.dvfs_candidate_batch import (
    MS_CANDIDATE_BATCH_SIZE_ENV,
    ORDERED_RAY_CANDIDATE_BATCH_MODE,
    SERIAL_CANDIDATE_MODE,
    analyze_operator_energy_batch,
    analyze_operator_energy_candidates,
    configured_ms_candidate_batch_size,
)
from neusim.npusim.frontend.Operator import (
    ComponentDVFSConfig,
    DVFSConfig,
    DVFSPolicy,
)

AUTHORITATIVE_SOURCE_COMMIT = "8ad6961b2a266e91ebb1162c7e2c5df61d10b1a4"
AUTHORITATIVE_SOURCE_SHA256 = (
    "5dac84fa466e1270ec9b2d9909cc5e1221bd9b517f0d2038fc3363251230c112"
)

_DOMAINS = ("sa", "vu", "sram", "hbm", "ici")
_DOMAIN_SUBS = {
    "sa": ("sa",),
    "vu": ("vu",),
    "sram": ("sram",),
    "hbm": ("hbm_mc", "hbm_die", "hbm_io"),
    "ici": ("ici_mc",),
}
_GRID_KEY = {
    "sa": "sa",
    "vu": "vu",
    "sram": "sram",
    "hbm": "hbm_mc",
    "ici": "ici_mc",
}
_DOMAIN_TIME = {
    "sa": "sa_time_ns",
    "vu": "vu_time_ns",
    "sram": "vmem_time_ns",
    "hbm": "memory_time_ns",
    "ici": "ici_time_ns",
}
_SUB_USES_FLOPS = frozenset({"sa", "vu"})

_PEAK = ComponentDVFSConfig(
    policy=DVFSPolicy.NONE,
    voltage_V=0.7,
    frequency_GHz=1.7,
    voltage_regulator_scaling_time_ns=0,
)


def _build_vr_grid(rows):
    """Tabulate regulator efficiency by voltage and activity."""
    volts = np.array(sorted({float(row.voltage_V) for row in rows}))
    activities = np.array(sorted({float(row.activity_factor) for row in rows}))
    if not len(volts) or not len(activities):
        raise ValueError("voltage-regulator table must not be empty")
    grid = np.zeros((len(volts), len(activities)))
    for voltage_index, voltage in enumerate(volts):
        voltage_rows = sorted(
            (
                row
                for row in rows
                if math.isclose(float(row.voltage_V), voltage, abs_tol=1e-12)
            ),
            key=lambda row: row.activity_factor,
        )
        for activity_index, activity in enumerate(activities):
            grid[voltage_index, activity_index] = next(
                (
                    row.power_efficiency_percent
                    for row in voltage_rows
                    if row.activity_factor >= activity - 1e-9
                ),
                voltage_rows[-1].power_efficiency_percent,
            )
    return volts, activities, grid


def _efficiency(grid_tuple, voltage, activity):
    """Vectorized source-compatible ceil lookup in the regulator table."""
    volts, activities, grid = grid_tuple
    voltage_index = np.clip(
        np.searchsorted(volts, voltage, side="left"),
        0,
        len(volts) - 1,
    )
    activity_index = np.clip(
        np.searchsorted(activities, activity, side="left"),
        0,
        len(activities) - 1,
    )
    return grid[voltage_index, activity_index]


def _domain_grids():
    """Return active V/f points for the five authoritative search domains.

    The recovered source included a 0 GHz HBM/ICI sentinel. NeuSim treats a
    nonpositive frequency as unscaled latency, so searching that point would
    create low power at baseline latency. Zero is clock-gated state, not an
    active DVFS point, and is deliberately excluded here.
    """
    from neusim.npusim.backend.dvfs_power_getter import (
        get_all_dvfs_configs_for_component,
    )

    grids = {}
    for domain in _DOMAINS:
        points = get_all_dvfs_configs_for_component(
            _GRID_KEY[domain],
            DVFSPolicy.CUSTOM_ALL_ms,
        )
        active = [
            point.model_copy()
            for point in points
            if point.frequency_GHz is not None and point.frequency_GHz > 0.0
        ]
        active.sort(
            key=lambda point: (
                float(point.frequency_GHz or 0.0),
                float(point.voltage_V or 0.0),
            )
        )
        if not active:
            raise ValueError(f"no positive-frequency points for {domain}")
        grids[domain] = active
    return grids


def _full_config(grids, indices):
    """Convert five domain indices into NeuSim's eight component settings."""
    from neusim.npusim.backend import dvfs_power_getter as getter

    hbm_mc = grids["hbm"][indices[3]]
    frequency = float(hbm_mc.frequency_GHz or 0.0)
    if frequency <= 0.0:
        raise ValueError("active HBM configuration must have positive frequency")
    target_bw = getter._baseline_bw_hbm() * frequency / 1.7
    die_points = sorted(
        (point for point in getter.HBM_DIE_POINTS if abs(point.voltage_V - 1.2) < 0.02),
        key=lambda point: point.bandwidth_GBs,
    )
    io_points = sorted(
        getter.HBM_IO_POINTS,
        key=lambda point: point.bandwidth_GBs,
    )
    if not die_points or not io_points:
        raise ValueError("HBM die and I/O calibration points must not be empty")
    die = next(
        (point for point in die_points if point.bandwidth_GBs >= target_bw - 1.0),
        die_points[-1],
    )
    io = next(
        (point for point in io_points if point.bandwidth_GBs >= target_bw - 1.0),
        io_points[-1],
    )
    return {
        "sa": grids["sa"][indices[0]].model_copy(),
        "vu": grids["vu"][indices[1]].model_copy(),
        "sram": grids["sram"][indices[2]].model_copy(),
        "hbm_mc": hbm_mc.model_copy(),
        "hbm_die": ComponentDVFSConfig(
            policy=DVFSPolicy.CUSTOM,
            voltage_V=die.voltage_V,
            frequency_GHz=frequency,
            voltage_regulator_scaling_time_ns=0,
        ),
        "hbm_io": ComponentDVFSConfig(
            policy=DVFSPolicy.CUSTOM,
            voltage_V=io.voltage_V,
            frequency_GHz=frequency,
            voltage_regulator_scaling_time_ns=0,
        ),
        "ici_mc": grids["ici"][indices[4]].model_copy(),
        "ici_phy": _PEAK.model_copy(),
    }


def _apply_config(op, config):
    for component, value in config.items():
        setattr(op, f"dvfs_{component}", value.model_copy())
    return op


def _analyze_many(
    jobs,
    config,
    pg_config,
    dvfs_config,
    remote_batch=None,
):
    """Analyze jobs in exact input order and report submission provenance."""
    return analyze_operator_energy_candidates(
        jobs,
        config,
        pg_config,
        dvfs_config,
        serial=os.environ.get("DVFS_PARETO_SERIAL") == "1",
        remote_batch=remote_batch,
    )


def _precompute(ops, config, interval_ns, pg_config):
    """Build regions and the source GA's separable time/energy tables."""
    from neusim.npusim.frontend.dvfs_region_merge import (
        build_regions_by,
        component_label_op,
    )
    from neusim.npusim.frontend.power_analysis_lib import (
        DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
        FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
    )

    dvfs_serial = os.environ.get("DVFS_PARETO_SERIAL") == "1"
    remote_candidate_batch = None
    if not dvfs_serial:
        import ray

        remote_candidate_batch = ray.remote(analyze_operator_energy_batch)
    candidate_batching = {
        "candidate_evaluation_mode": (
            SERIAL_CANDIDATE_MODE if dvfs_serial else ORDERED_RAY_CANDIDATE_BATCH_MODE
        ),
        "candidate_batch_size": configured_ms_candidate_batch_size(),
        "candidate_batch_size_env": MS_CANDIDATE_BATCH_SIZE_ENV,
        "candidate_count": 0,
        "submitted_candidate_tasks": 0,
        "candidate_result_order": "operator_major_domain_grid_major_then_peak",
    }

    grids = _domain_grids()
    grid_sizes = {domain: len(grids[domain]) for domain in _DOMAINS}
    dvfs_rows = [
        row
        for row in DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE
        if row.scaling_time_ns == 20
    ]
    vr_dvfs = _build_vr_grid(dvfs_rows)
    vr_fixed = _build_vr_grid(list(FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE))
    regions = build_regions_by(ops, interval_ns, component_label_op)
    if not regions:
        raise ValueError("eNPU-ms requires at least one region")

    region_of = np.full(len(ops), -1, dtype=np.int32)
    for region_index, region in enumerate(regions):
        for op_index in region.op_indices:
            region_of[op_index] = region_index
    if np.any(region_of < 0):
        raise RuntimeError("region construction omitted an operator")

    peak_indices = tuple(grid_sizes[domain] - 1 for domain in _DOMAINS)
    dvfs_eval = DVFSConfig(
        policy=DVFSPolicy.CUSTOM_ALL_ms,
        frequency_adjustment_interval_ns=float(interval_ns),
    )
    pre_ops = []
    for op_index, op in enumerate(ops):
        jobs = []
        job_keys = []
        for domain_index, domain in enumerate(_DOMAINS):
            for grid_index in range(grid_sizes[domain]):
                indices = list(peak_indices)
                indices[domain_index] = grid_index
                candidate = op.model_copy(deep=True)
                candidate.stats.count = 1
                jobs.append(
                    _apply_config(
                        candidate,
                        _full_config(grids, tuple(indices)),
                    )
                )
                job_keys.append((domain, grid_index))
        peak = op.model_copy(deep=True)
        peak.stats.count = 1
        jobs.append(_apply_config(peak, _full_config(grids, peak_indices)))
        evaluated, operator_batching = _analyze_many(
            jobs,
            config,
            pg_config,
            dvfs_eval,
            remote_candidate_batch,
        )
        candidate_batching["candidate_count"] += operator_batching["candidate_count"]
        candidate_batching["submitted_candidate_tasks"] += operator_batching[
            "submitted_candidate_tasks"
        ]
        domain_results = evaluated[:-1]
        peak_result = evaluated[-1]
        peak_stats = peak_result.stats
        peak_time = float(peak_stats.execution_time_ns)

        domain_time = {domain: np.zeros(grid_sizes[domain]) for domain in _DOMAINS}
        subdata = {
            domain: {
                sub: {
                    "raw_dynamic": np.zeros(grid_sizes[domain]),
                    "static_power": np.zeros(grid_sizes[domain]),
                    "voltage": np.zeros(grid_sizes[domain]),
                    "scaling_ns": 20,
                }
                for sub in _DOMAIN_SUBS[domain]
            }
            for domain in _DOMAINS
        }
        for (domain, grid_index), result in zip(
            job_keys,
            domain_results,
            strict=True,
        ):
            stats = result.stats
            execution_time = float(stats.execution_time_ns)
            domain_time[domain][grid_index] = float(
                getattr(stats, _DOMAIN_TIME[domain])
            )
            for sub in _DOMAIN_SUBS[domain]:
                component = getattr(result, f"dvfs_{sub}")
                efficiency = component.voltage_conversion_power_efficiency_percent
                values = subdata[domain][sub]
                values["raw_dynamic"][grid_index] = (
                    getattr(stats, f"dynamic_energy_{sub}_J") * efficiency / 100.0
                )
                values["static_power"][grid_index] = (
                    getattr(stats, f"static_energy_{sub}_J")
                    * efficiency
                    / 100.0
                    / execution_time
                    if execution_time > 0.0
                    else 0.0
                )
                values["voltage"][grid_index] = float(component.voltage_V)
                values["scaling_ns"] = component.voltage_regulator_scaling_time_ns

        phy_component = peak_result.dvfs_ici_phy
        phy_efficiency = phy_component.voltage_conversion_power_efficiency_percent
        ici_phy = {
            "raw_dynamic": (
                peak_stats.dynamic_energy_ici_phy_J * phy_efficiency / 100.0
            ),
            "static_power": (
                peak_stats.static_energy_ici_phy_J * phy_efficiency / 100.0 / peak_time
                if peak_time > 0.0
                else 0.0
            ),
            "voltage": float(phy_component.voltage_V),
            "scaling_ns": (phy_component.voltage_regulator_scaling_time_ns),
        }
        other_power = (
            (peak_stats.dynamic_energy_other_J + peak_stats.static_energy_other_J)
            / peak_time
            if peak_time > 0.0
            else 0.0
        )
        pre_ops.append(
            {
                "domain_time": domain_time,
                "subdata": subdata,
                "ici_phy": ici_phy,
                "other_power": other_power,
                "flops_util": float(min(1.0, max(0.0, op.stats.flops_util))),
                "count": int(op.stats.count),
                "region": int(region_of[op_index]),
            }
        )

    pre = {
        "grids": grids,
        "grid_sizes": grid_sizes,
        "vr_dvfs": vr_dvfs,
        "vr_fixed": vr_fixed,
        "regions": regions,
        "num_regions": len(regions),
        "pre_ops": pre_ops,
        "candidate_batching": candidate_batching,
    }
    peak_plan = np.array(
        [[grid_sizes[domain] - 1 for domain in _DOMAINS]] * len(regions),
        dtype=np.int32,
    )
    baseline_time, baseline_energy = _region_totals(pre, peak_plan)
    pre["baseline_time"] = float(baseline_time)
    pre["baseline_energy"] = float(baseline_energy)
    pre["peak_plan"] = peak_plan
    return pre


def _op_metrics(pre_op, pre, indices):
    population = indices.shape[0]
    domain_times = np.empty((population, len(_DOMAINS)))
    for domain_index, domain in enumerate(_DOMAINS):
        domain_times[:, domain_index] = pre_op["domain_time"][domain][
            indices[:, domain_index]
        ]
    execution_time = domain_times.max(axis=1)
    energy = np.zeros(population)
    for domain_index, domain in enumerate(_DOMAINS):
        selected = indices[:, domain_index]
        active_time = domain_times[:, domain_index]
        for sub in _DOMAIN_SUBS[domain]:
            values = pre_op["subdata"][domain][sub]
            activity = (
                np.full(population, pre_op["flops_util"])
                if sub in _SUB_USES_FLOPS
                else np.divide(
                    active_time,
                    execution_time,
                    out=np.zeros(population),
                    where=execution_time > 0.0,
                )
            )
            table = pre["vr_dvfs"] if values["scaling_ns"] != 0 else pre["vr_fixed"]
            efficiency = _efficiency(
                table,
                values["voltage"][selected],
                activity,
            )
            energy += (
                (
                    values["raw_dynamic"][selected]
                    + values["static_power"][selected] * execution_time
                )
                * 100.0
                / efficiency
            )

    phy = pre_op["ici_phy"]
    ici_activity = np.divide(
        domain_times[:, _DOMAINS.index("ici")],
        execution_time,
        out=np.zeros(population),
        where=execution_time > 0.0,
    )
    phy_table = pre["vr_dvfs"] if phy["scaling_ns"] != 0 else pre["vr_fixed"]
    energy += (
        (phy["raw_dynamic"] + phy["static_power"] * execution_time)
        * 100.0
        / _efficiency(
            phy_table,
            np.full(population, phy["voltage"]),
            ici_activity,
        )
    )
    energy += pre_op["other_power"] * execution_time
    return execution_time, energy


def _region_totals(pre, region_indices):
    total_time = 0.0
    total_energy = 0.0
    for pre_op in pre["pre_ops"]:
        region = pre_op["region"]
        execution, energy = _op_metrics(
            pre_op,
            pre,
            region_indices[region : region + 1],
        )
        total_time += float(execution[0]) * pre_op["count"]
        total_energy += float(energy[0]) * pre_op["count"]
    return total_time, total_energy


def _evaluate_population(pre, population):
    population_size = population.shape[0]
    total_time = np.zeros(population_size)
    total_energy = np.zeros(population_size)
    for pre_op in pre["pre_ops"]:
        execution, energy = _op_metrics(
            pre_op,
            pre,
            population[:, pre_op["region"], :],
        )
        total_time += execution * pre_op["count"]
        total_energy += energy * pre_op["count"]
    return total_time, total_energy


def _validate_ga_parameters(population_size, max_generations, elitism_count):
    if population_size < 3:
        raise ValueError("population_size must be at least 3")
    if max_generations < 1:
        raise ValueError("max_generations must be positive")
    if elitism_count < 1 or elitism_count >= population_size:
        raise ValueError("elitism_count must be in [1, population_size)")


def _run_ga(
    pre,
    budget,
    random_state,
    *,
    population_size=300,
    max_generations=200,
    crossover_prob=0.8,
    mutation_prob=0.08,
    elitism_count=10,
    seed_population=None,
    seed_individual=None,
):
    """Run the recovered regional GA with a hard feasibility constraint."""
    _validate_ga_parameters(
        population_size,
        max_generations,
        elitism_count,
    )
    num_regions = pre["num_regions"]
    sizes = [pre["grid_sizes"][domain] for domain in _DOMAINS]
    peak = np.array([size - 1 for size in sizes], dtype=np.int32)
    baseline_time = pre["baseline_time"]
    baseline_energy = pre["baseline_energy"]
    allowed_time = baseline_time * (1.0 + budget)

    population = np.empty(
        (population_size, num_regions, len(_DOMAINS)),
        dtype=np.int32,
    )
    if seed_population is not None and seed_population.shape == population.shape:
        population[:] = seed_population
    else:
        for domain_index, size in enumerate(sizes):
            population[:, :, domain_index] = random_state.randint(
                0,
                size,
                size=(population_size, num_regions),
            )
    population[0] = peak[None, :]
    if seed_individual is not None:
        population[1] = seed_individual

    def score(values):
        total_time, total_energy = values
        valid = (
            (total_time > 0.0)
            & (total_energy > 0.0)
            & np.isfinite(total_time)
            & np.isfinite(total_energy)
            & (total_time <= allowed_time + max(1e-6, abs(allowed_time) * 1e-12))
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            objective = (
                baseline_time / total_time * (baseline_energy / total_energy) ** 3
            )
        return np.where(valid, objective, -np.inf)

    best_plan = population[0].copy()
    best_score = -np.inf
    best_time = baseline_time
    best_energy = baseline_energy
    for _ in range(max_generations):
        times, energies = _evaluate_population(pre, population)
        scores = score((times, energies))
        best_index = int(np.argmax(scores))
        if scores[best_index] > best_score:
            best_score = float(scores[best_index])
            best_plan = population[best_index].copy()
            best_time = float(times[best_index])
            best_energy = float(energies[best_index])

        weights = np.where(np.isfinite(scores), scores, 0.0)
        weight_sum = float(weights.sum())
        probabilities = (
            weights / weight_sum
            if weight_sum > 0.0
            else np.full(population_size, 1.0 / population_size)
        )
        flat = population.reshape(population_size, num_regions * 5)
        new_flat = np.empty_like(flat)
        elite = np.argsort(scores)[-elitism_count:]
        new_flat[:elitism_count] = flat[elite]
        offspring_count = population_size - elitism_count
        parents = random_state.choice(
            population_size,
            size=(offspring_count, 2),
            p=probabilities,
        )
        children = flat[parents[:, 0]].copy()
        other = flat[parents[:, 1]]
        chromosome_length = num_regions * 5
        if chromosome_length > 1:
            do_crossover = random_state.random(offspring_count) < crossover_prob
            points = random_state.randint(
                1,
                chromosome_length,
                size=offspring_count,
            )
            take_other = (
                np.arange(chromosome_length)[None, :] >= points[:, None]
            ) & do_crossover[:, None]
            children = np.where(take_other, other, children)
        mutate = (
            random_state.random((offspring_count, chromosome_length)) < mutation_prob
        )
        steps = random_state.choice(
            np.array([-1, 1]),
            size=(offspring_count, chromosome_length),
        )
        children = np.where(mutate, children + steps, children)
        for domain_index, size in enumerate(sizes):
            children[:, domain_index::5] = np.clip(
                children[:, domain_index::5],
                0,
                size - 1,
            )
        new_flat[elitism_count:] = children
        population = new_flat.reshape(
            population_size,
            num_regions,
            5,
        )

    times, energies = _evaluate_population(pre, population)
    scores = score((times, energies))
    best_index = int(np.argmax(scores))
    if scores[best_index] > best_score:
        best_plan = population[best_index].copy()
        best_time = float(times[best_index])
        best_energy = float(energies[best_index])
    return best_plan, population, best_time, best_energy


def _apply_back(ops, pre, plan):
    """Apply one complete five-domain plan to every member of each region."""
    output = []
    for region_index, region in enumerate(pre["regions"]):
        config = _full_config(
            pre["grids"],
            tuple(int(value) for value in plan[region_index]),
        )
        for op_index in region.op_indices:
            candidate = ops[op_index].model_copy(deep=True)
            output.append(_apply_config(candidate, config))
    if len(output) != len(ops):
        raise RuntimeError("eNPU-ms output omitted or duplicated operators")
    return output


def _measure_schedule(
    ops,
    config,
    pg_config,
    interval_ns,
    budget,
):
    from neusim.npusim.frontend.power_analysis_lib import (
        analyze_operator_energy,
    )

    dvfs = DVFSConfig(
        policy=DVFSPolicy.CUSTOM_ALL_ms,
        performance_degradation_percentage=float(budget),
        frequency_adjustment_interval_ns=float(interval_ns),
    )
    evaluated = [op.model_copy(deep=True) for op in ops]
    for op in evaluated:
        analyze_operator_energy(
            op,
            config,
            pg_config=pg_config,
            dvfs_config=dvfs,
            set_dvfs_config_for_op=False,
        )
    total_time = sum(
        float(op.stats.execution_time_ns) * int(op.stats.count) for op in evaluated
    )
    total_energy = sum(
        float(op.stats.total_energy_J) * int(op.stats.count) for op in evaluated
    )
    return total_time, total_energy


def _pg_label(pg_config):
    if pg_config is None:
        return "NoPG"
    if isinstance(pg_config, str):
        return pg_config
    return type(pg_config).__name__


def configure_enpu_ms_all_budgets(
    ops,
    config,
    budgets,
    interval_ns=5_000_000.0,
    dump_pareto_points_to_file=False,
    pg_config=None,
    population_size=300,
    max_generations=200,
    crossover_prob=0.8,
    mutation_prob=0.08,
    elitism_count=None,
    exact_fallback_candidates=8,
    seed=42,
    timing_result=None,
):
    """Run the authoritative one-plan-per-region GA for several budgets."""
    del dump_pareto_points_to_file
    if not ops:
        raise ValueError("eNPU-ms requires at least one operator")
    sorted_budgets = sorted({float(value) for value in budgets})
    if not sorted_budgets:
        raise ValueError("budgets must contain at least one value")
    if any(not math.isfinite(value) or value < 0.0 for value in sorted_budgets):
        raise ValueError("budgets must be finite, non-negative fractions")
    if elitism_count is None:
        elitism_count = min(10, population_size - 1)
    _validate_ga_parameters(
        population_size,
        max_generations,
        elitism_count,
    )
    if exact_fallback_candidates < 0:
        raise ValueError("exact_fallback_candidates must be non-negative")

    logging.set_verbosity(logging.INFO)
    total_start = time.perf_counter()
    precompute_start = time.perf_counter()
    pre = _precompute(ops, config, float(interval_ns), pg_config)
    point_gen_seconds = time.perf_counter() - precompute_start
    peak_schedule = _apply_back(ops, pre, pre["peak_plan"])
    exact_measurement_seconds = 0.0
    measurement_start = time.perf_counter()
    peak_time, peak_energy = _measure_schedule(
        peak_schedule,
        config,
        pg_config,
        interval_ns,
        0.0,
    )
    exact_measurement_seconds += time.perf_counter() - measurement_start
    raw_time = float(
        sum(float(op.stats.execution_time_ns) * int(op.stats.count) for op in ops)
    )
    if raw_time <= 0.0 or peak_time <= 0.0 or peak_energy <= 0.0:
        raise ValueError("eNPU-ms baseline must have positive time and energy")
    transition_allowance = max(0.0, peak_time - raw_time)
    zero_budget_allowed_time = max(raw_time, peak_time)
    natural_headroom = max(0.0, raw_time - peak_time)

    random_state = np.random.RandomState(seed)
    results = {}
    ga_seconds = {}
    selected_times = {}
    selected_energies = {}
    model_times = {}
    model_energies = {}
    allowed_times = {}
    exact_fallback = {}
    exact_fallback_recovered = {}
    exact_candidate_counts = {}
    incumbent_schedule = peak_schedule
    incumbent_time = peak_time
    incumbent_energy = peak_energy
    previous_population = None
    previous_plan = None

    for budget in sorted_budgets:
        ga_start = time.perf_counter()
        candidate_plan, population, model_time, model_energy = _run_ga(
            pre,
            budget,
            random_state,
            population_size=population_size,
            max_generations=max_generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            elitism_count=elitism_count,
            seed_population=previous_population,
            seed_individual=previous_plan,
        )
        ga_seconds[budget] = time.perf_counter() - ga_start
        previous_population = population
        previous_plan = candidate_plan
        candidate_schedule = _apply_back(ops, pre, candidate_plan)
        measurement_start = time.perf_counter()
        exact_time, exact_energy = _measure_schedule(
            candidate_schedule,
            config,
            pg_config,
            interval_ns,
            budget,
        )
        exact_measurement_seconds += time.perf_counter() - measurement_start
        allowed = zero_budget_allowed_time + raw_time * budget
        feasible = exact_time <= allowed + max(
            1e-5,
            abs(allowed) * 1e-12,
        )
        exact_fallback[budget] = not feasible
        exact_fallback_recovered[budget] = False
        exact_candidate_counts[budget] = 1

        # The recovered GA searches with a separable model. If its model-best
        # plan misses the exact budget, rank a bounded set of other modeled-
        # feasible final-population plans and remeasure them through NeuSim.
        # This preserves hard feasibility without turning every GA evaluation
        # into an expensive full power-model call.
        if not feasible and exact_fallback_candidates:
            population_times, population_energies = _evaluate_population(
                pre,
                population,
            )
            model_allowed = pre["baseline_time"] * (1.0 + budget)
            model_feasible = np.flatnonzero(
                population_times
                <= model_allowed + max(1e-6, abs(model_allowed) * 1e-12)
            )
            ranked = model_feasible[np.argsort(population_energies[model_feasible])]
            seen = {candidate_plan.tobytes()}
            retry_best = None
            for population_index in ranked:
                retry_plan = population[int(population_index)]
                signature = retry_plan.tobytes()
                if signature in seen:
                    continue
                seen.add(signature)
                retry_schedule = _apply_back(ops, pre, retry_plan)
                measurement_start = time.perf_counter()
                retry_time, retry_energy = _measure_schedule(
                    retry_schedule,
                    config,
                    pg_config,
                    interval_ns,
                    budget,
                )
                exact_measurement_seconds += time.perf_counter() - measurement_start
                exact_candidate_counts[budget] += 1
                if retry_time <= allowed + max(1e-5, abs(allowed) * 1e-12) and (
                    retry_best is None or retry_energy < retry_best[3]
                ):
                    retry_best = (
                        retry_plan.copy(),
                        retry_schedule,
                        retry_time,
                        retry_energy,
                    )
                if exact_candidate_counts[budget] >= exact_fallback_candidates + 1:
                    break
            if retry_best is not None:
                (
                    candidate_plan,
                    candidate_schedule,
                    exact_time,
                    exact_energy,
                ) = retry_best
                feasible = True
                exact_fallback_recovered[budget] = True

        # Promote an exact-feasible retry (or retain the model-best plan) as
        # the explicit warm start for the next, looser request budget.
        previous_plan = candidate_plan
        if feasible and exact_energy < incumbent_energy - 1e-15:
            incumbent_schedule = candidate_schedule
            incumbent_time = exact_time
            incumbent_energy = exact_energy

        results[budget] = [op.model_copy(deep=True) for op in incumbent_schedule]
        selected_times[budget] = float(incumbent_time)
        selected_energies[budget] = float(incumbent_energy)
        model_times[budget] = float(model_time)
        model_energies[budget] = float(model_energy)
        allowed_times[budget] = float(allowed)
        logging.info(
            "eNPU-ms budget=%.4f regions=%d energy=%.6f J time=%.0f ns",
            budget,
            pre["num_regions"],
            incumbent_energy,
            incumbent_time,
        )

    timings = {
        "algorithm": "authoritative_5domain_regional_GA_all_budgets",
        "implementation_provenance": "authoritative_trace_util_port",
        "authoritative_source_commit": AUTHORITATIVE_SOURCE_COMMIT,
        "authoritative_source_sha256": AUTHORITATIVE_SOURCE_SHA256,
        "source_deviations": [
            "hard feasibility instead of a finite fitness bonus",
            "exact PG-aware remeasurement and feasible checkpointing",
            "bounded exact retries after a model/exact feasibility mismatch",
            "nonpositive active V/f sentinel points excluded",
            "NeuSim request-budget accounting excludes fixed peak-plan overhead",
        ],
        "zero_frequency_points_excluded": True,
        "region_plan_scope": "one plan for all ops and executions in a region",
        "point_gen_s": point_gen_seconds,
        **pre.get("candidate_batching", {}),
        "region_candidate_evaluation_s": 0.0,
        "ga_s": ga_seconds,
        "search_s": ga_seconds,
        "exact_measurement_s": exact_measurement_seconds,
        "total_s": time.perf_counter() - total_start,
        "num_regions": pre["num_regions"],
        "num_ops": len(ops),
        "domain_order": list(_DOMAINS),
        "domain_state_counts": {
            domain: len(pre["grids"][domain]) for domain in _DOMAINS
        },
        "power_gating_config": _pg_label(pg_config),
        "raw_baseline_time_ns": raw_time,
        "fastest_plan_time_ns": float(peak_time),
        "fixed_transition_allowance_ns": transition_allowance,
        "fixed_peak_plan_overhead_ns": transition_allowance,
        "fixed_overhead_scope": (
            "peak CUSTOM_ALL_ms transition plus requested power-gating delay"
        ),
        "zero_budget_allowed_time_ns": zero_budget_allowed_time,
        "natural_headroom_ns": natural_headroom,
        "allowed_time_ns": allowed_times,
        "selected_time_ns": selected_times,
        "selected_energy_J": selected_energies,
        "model_selected_time_ns": model_times,
        "model_selected_energy_J": model_energies,
        "model_selected_metrics_scope": (
            "GA model-best before any exact-feasibility retry"
        ),
        "movement_from_fastest_pct": {
            budget: 100.0 * (selected_times[budget] - peak_time) / raw_time
            for budget in sorted_budgets
        },
        "variable_budget_used_pct": {
            budget: 100.0
            * max(
                0.0,
                selected_times[budget] - zero_budget_allowed_time,
            )
            / raw_time
            for budget in sorted_budgets
        },
        "exact_infeasible_candidate_fallback": exact_fallback,
        "exact_fallback_recovered": exact_fallback_recovered,
        "exact_candidate_counts": exact_candidate_counts,
        "candidate_slowdown_envelope": "full_positive_frequency_grid",
    }
    configure_enpu_ms_all_budgets.last_timings = timings
    if timing_result is not None:
        timing_result.update(timings)
    return results


configure_enpu_ms_all_budgets.last_timings = {}


def configure_enpu_ms_with_regions(
    ops,
    config,
    dvfs_config,
    dump_pareto_points_to_file=False,
    population_size=300,
    max_generations=200,
    crossover_prob=0.8,
    mutation_prob=0.08,
    elitism_count=None,
    exact_fallback_candidates=8,
    seed=42,
    pg_config=None,
    timing_result=None,
):
    """Run the authoritative eNPU-ms port for one request budget."""
    mode = (
        getattr(dvfs_config, "custom_compute_domain_mode", "dom5") or "dom5"
    ).lower()
    if mode != "dom5":
        raise ValueError(
            "authoritative eNPU-ms uses five independent domains and requires "
            f"custom_compute_domain_mode='dom5', got {mode!r}"
        )
    budget = float(dvfs_config.performance_degradation_percentage)
    configured = configure_enpu_ms_all_budgets(
        ops,
        config,
        [budget],
        interval_ns=dvfs_config.frequency_adjustment_interval_ns,
        dump_pareto_points_to_file=dump_pareto_points_to_file,
        pg_config=pg_config,
        population_size=population_size,
        max_generations=max_generations,
        crossover_prob=crossover_prob,
        mutation_prob=mutation_prob,
        elitism_count=elitism_count,
        exact_fallback_candidates=exact_fallback_candidates,
        seed=seed,
        timing_result=timing_result,
    )
    ops[:] = [op.model_copy(deep=True) for op in configured[budget]]
    return ops
