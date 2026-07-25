import math

import numpy as np
import pytest

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.models.ModelConfig import ModelConfig
from neusim.npusim.frontend import dvfs_enpu_ms
from neusim.npusim.frontend.dvfs_enpu_ms import (
    AUTHORITATIVE_SOURCE_COMMIT,
    AUTHORITATIVE_SOURCE_SHA256,
    _apply_back,
    _domain_grids,
    _run_ga,
    configure_enpu_ms_all_budgets,
    configure_enpu_ms_with_regions,
)
from neusim.npusim.frontend.dvfs_region_merge import (
    build_regions_by,
    component_label_op,
)
from neusim.npusim.frontend.Operator import DVFSConfig, DVFSPolicy, Operator


def _op(name: str, duration_ns: int, bounded_by: str, count: int = 1) -> Operator:
    op = Operator(name=name)
    op.description = "prefill"
    op.stats.execution_time_ns = duration_ns
    op.stats.count = count
    op.stats.bounded_by = bounded_by
    op.stats.flops_util = 0.5
    op.stats.sa_time_ns = max(1, duration_ns // 4)
    op.stats.vu_time_ns = max(1, duration_ns // 5)
    op.stats.vmem_time_ns = max(1, duration_ns // 3)
    if bounded_by == "Memory":
        op.stats.memory_time_ns = duration_ns
        op.stats.memory_traffic_bytes = 1024
    elif bounded_by == "ICI/NVLink":
        op.stats.ici_time_ns = duration_ns
    elif bounded_by == "VU":
        op.stats.vu_time_ns = duration_ns
    elif bounded_by == "VMEM":
        op.stats.vmem_time_ns = duration_ns
    else:
        op.stats.sa_time_ns = duration_ns
    return op


def _hardware_plan(op: Operator) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            component,
            getattr(op, f"dvfs_{component}").policy,
            getattr(op, f"dvfs_{component}").voltage_V,
            getattr(op, f"dvfs_{component}").frequency_GHz,
            getattr(
                op,
                f"dvfs_{component}",
            ).voltage_regulator_scaling_time_ns,
        )
        for component in (
            "sa",
            "vu",
            "sram",
            "hbm_mc",
            "hbm_die",
            "hbm_io",
            "ici_mc",
            "ici_phy",
        )
    )


def test_authoritative_domain_grids_exclude_zero_frequency_sentinels():
    grids = _domain_grids()

    assert tuple(grids) == ("sa", "vu", "sram", "hbm", "ici")
    assert all(grids.values())
    assert all(
        point.frequency_GHz is not None and point.frequency_GHz > 0.0
        for points in grids.values()
        for point in points
    )
    assert all(
        [
            float(point.frequency_GHz)
            for point in points
        ]
        == sorted(float(point.frequency_GHz) for point in points)
        for points in grids.values()
    )


def test_apply_back_uses_one_complete_plan_per_region_and_preserves_trace():
    ops = [
        _op("sa-a", 3_000_000, "Compute", count=7),
        _op("sa-b", 3_000_000, "Compute", count=7),
        _op("hbm-a", 3_000_000, "Memory", count=7),
        _op("hbm-b", 3_000_000, "Memory", count=7),
    ]
    regions = build_regions_by(ops, 5_000_000, component_label_op)
    assert [region.op_indices for region in regions] == [[0, 1], [2, 3]]

    grids = _domain_grids()
    plan = np.array(
        [
            [len(grids[domain]) - 1 for domain in grids],
            [0 for _domain in grids],
        ],
        dtype=np.int32,
    )
    configured = _apply_back(
        ops,
        {"grids": grids, "regions": regions},
        plan,
    )

    assert [(op.name, op.stats.count) for op in configured] == [
        ("sa-a", 7),
        ("sa-b", 7),
        ("hbm-a", 7),
        ("hbm-b", 7),
    ]
    assert _hardware_plan(configured[0]) == _hardware_plan(configured[1])
    assert _hardware_plan(configured[2]) == _hardware_plan(configured[3])
    assert _hardware_plan(configured[0]) != _hardware_plan(configured[2])


def test_ga_never_selects_an_over_budget_low_energy_candidate(monkeypatch):
    pre = {
        "num_regions": 1,
        "grid_sizes": {domain: 2 for domain in ("sa", "vu", "sram", "hbm", "ici")},
        "baseline_time": 100.0,
        "baseline_energy": 100.0,
    }

    def evaluate(_pre, population):
        at_peak = np.all(population == 1, axis=(1, 2))
        return (
            np.where(at_peak, 100.0, 200.0),
            np.where(at_peak, 100.0, 1.0),
        )

    monkeypatch.setattr(dvfs_enpu_ms, "_evaluate_population", evaluate)
    plan, _population, selected_time, selected_energy = _run_ga(
        pre,
        budget=0.0,
        random_state=np.random.RandomState(4),
        population_size=12,
        max_generations=3,
    )

    assert np.all(plan == 1)
    assert selected_time == 100.0
    assert selected_energy == 100.0


def test_enpu_ms_is_exact_feasible_monotonic_and_pg_aware(monkeypatch):
    monkeypatch.setenv("DVFS_PARETO_SERIAL", "1")
    config = ModelConfig(
        model_type="test",
        freq_GHz=1.7,
        num_sa=4,
        num_vu=4,
        enable_dvfs=True,
        output_file_path="/tmp/enpu-ms-test.csv",
    )
    compute = _op("compute", 3_000_000, "Compute")
    compute.stats.memory_time_ns = 500_000
    memory = _op("memory", 3_000_000, "Memory")
    memory.stats.sa_time_ns = 800_000
    memory.stats.vu_time_ns = 400_000
    memory.stats.vmem_time_ns = 266_666
    raw_ops = [compute, memory]

    from neusim.npusim.frontend import power_analysis_lib

    original_analyze = power_analysis_lib.analyze_operator_energy
    seen_pg_configs = []

    def record_pg(op, analysis_config, pg_config=None, *args, **kwargs):
        seen_pg_configs.append(pg_config)
        return original_analyze(
            op,
            analysis_config,
            pg_config,
            *args,
            **kwargs,
        )

    monkeypatch.setattr(
        power_analysis_lib,
        "analyze_operator_energy",
        record_pg,
    )
    configured = configure_enpu_ms_all_budgets(
        raw_ops,
        config,
        (0.0, 0.1),
        interval_ns=5_000_000,
        pg_config="NoPG",
        population_size=12,
        max_generations=3,
        seed=7,
    )
    timings = configure_enpu_ms_all_budgets.last_timings

    assert timings["implementation_provenance"] == "authoritative_trace_util_port"
    assert timings["authoritative_source_commit"] == AUTHORITATIVE_SOURCE_COMMIT
    assert timings["authoritative_source_sha256"] == AUTHORITATIVE_SOURCE_SHA256
    assert timings["zero_frequency_points_excluded"] is True
    assert timings["candidate_slowdown_envelope"] == (
        "full_positive_frequency_grid"
    )
    assert timings["power_gating_config"] == "NoPG"
    assert timings["num_regions"] == 1
    assert seen_pg_configs
    assert set(seen_pg_configs) == {"NoPG"}

    baseline_time = timings["raw_baseline_time_ns"]
    assert baseline_time == 6_000_000
    assert timings["fastest_plan_time_ns"] >= baseline_time
    assert timings["fixed_transition_allowance_ns"] == pytest.approx(
        timings["fastest_plan_time_ns"] - baseline_time
    )
    energies = []
    for budget in (0.0, 0.1):
        selected_time = timings["selected_time_ns"][budget]
        selected_energy = timings["selected_energy_J"][budget]
        assert selected_time <= timings["allowed_time_ns"][budget] + 1e-5
        assert timings["allowed_time_ns"][budget] == pytest.approx(
            timings["zero_budget_allowed_time_ns"] + baseline_time * budget
        )
        energies.append(selected_energy)

        assert [op.name for op in configured[budget]] == [
            "compute",
            "memory",
        ]
        assert [op.stats.count for op in configured[budget]] == [1, 1]
        assert _hardware_plan(configured[budget][0]) == _hardware_plan(
            configured[budget][1]
        )

        evaluated = [op.model_copy(deep=True) for op in configured[budget]]
        dvfs = DVFSConfig(
            policy=DVFSPolicy.CUSTOM_ALL_ms,
            performance_degradation_percentage=budget,
            frequency_adjustment_interval_ns=5_000_000,
        )
        for op in evaluated:
            original_analyze(
                op,
                config,
                pg_config="NoPG",
                dvfs_config=dvfs,
                set_dvfs_config_for_op=False,
            )
        assert math.isclose(
            sum(op.stats.total_energy_J * op.stats.count for op in evaluated),
            selected_energy,
            rel_tol=1e-10,
            abs_tol=1e-12,
        )

    assert energies[1] <= energies[0] + 1e-12


def test_exact_infeasible_model_best_retries_feasible_population(monkeypatch):
    ops = [_op("raw", 100, "Compute")]
    peak = np.ones((1, 5), dtype=np.int32)
    model_best = np.zeros((1, 5), dtype=np.int32)
    exact_feasible = model_best.copy()
    exact_feasible[0, 0] = 1
    population = np.stack(
        [model_best, exact_feasible, peak],
        axis=0,
    )
    pre = {
        "num_regions": 1,
        "peak_plan": peak,
        "baseline_time": 100.0,
        "baseline_energy": 100.0,
        "grid_sizes": {
            domain: 2
            for domain in ("sa", "vu", "sram", "hbm", "ici")
        },
        "grids": {
            domain: [None, None]
            for domain in ("sa", "vu", "sram", "hbm", "ici")
        },
        "regions": [],
    }

    monkeypatch.setattr(dvfs_enpu_ms, "_precompute", lambda *_args: pre)

    def run_ga(*_args, **_kwargs):
        return model_best, population, 100.0, 1.0

    monkeypatch.setattr(dvfs_enpu_ms, "_run_ga", run_ga)

    def apply_back(input_ops, _pre, plan):
        output = [op.model_copy(deep=True) for op in input_ops]
        output[0].name = ",".join(str(int(value)) for value in plan[0])
        return output

    monkeypatch.setattr(dvfs_enpu_ms, "_apply_back", apply_back)

    def measure(schedule, *_args):
        signature = schedule[0].name
        if signature == "0,0,0,0,0":
            return 120.0, 1.0
        if signature == "1,0,0,0,0":
            return 100.0, 50.0
        return 100.0, 100.0

    monkeypatch.setattr(dvfs_enpu_ms, "_measure_schedule", measure)

    def evaluate(_pre, candidates):
        at_model_best = np.all(candidates == model_best, axis=(1, 2))
        at_exact_feasible = np.all(
            candidates == exact_feasible,
            axis=(1, 2),
        )
        return (
            np.full(len(candidates), 100.0),
            np.where(at_model_best, 1.0, np.where(at_exact_feasible, 2.0, 100.0)),
        )

    monkeypatch.setattr(dvfs_enpu_ms, "_evaluate_population", evaluate)
    configured = configure_enpu_ms_all_budgets(
        ops,
        ChipConfig(),
        (0.0,),
        population_size=3,
        max_generations=1,
    )
    timings = configure_enpu_ms_all_budgets.last_timings

    assert configured[0.0][0].name == "1,0,0,0,0"
    assert timings["selected_time_ns"][0.0] == 100.0
    assert timings["selected_energy_J"][0.0] == 50.0
    assert timings["exact_infeasible_candidate_fallback"][0.0] is True
    assert timings["exact_fallback_recovered"][0.0] is True
    assert timings["exact_candidate_counts"][0.0] >= 2


def test_single_budget_wrapper_rejects_non_authoritative_domain_mode():
    ops = [_op("compute", 3_000_000, "Compute")]
    dvfs = DVFSConfig(
        policy=DVFSPolicy.CUSTOM_ALL_ms,
        custom_compute_domain_mode="dom3",
    )

    with pytest.raises(ValueError, match="five independent domains"):
        configure_enpu_ms_with_regions(ops, ChipConfig(), dvfs)


def test_enpu_ms_rejects_empty_trace_without_overwriting_timings():
    previous = configure_enpu_ms_all_budgets.last_timings
    with pytest.raises(ValueError, match="at least one operator"):
        configure_enpu_ms_all_budgets([], ChipConfig(), (0.0,))
    assert configure_enpu_ms_all_budgets.last_timings is previous
