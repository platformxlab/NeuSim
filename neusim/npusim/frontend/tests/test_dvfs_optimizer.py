import csv
import io
import math
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from neusim.configs.models.ModelConfig import ModelConfig
from neusim.npusim.backend import dvfs_power_getter
from neusim.npusim.frontend import dvfs_optimizer, power_analysis_lib
from neusim.npusim.frontend.Operator import (
    ComponentDVFSConfig,
    DVFSConfig,
    DVFSPolicy,
    Operator,
)


def _point(name: str, execution_time_ns: int, energy_j: float) -> Operator:
    point = Operator(name=name)
    point.stats.execution_time_ns = execution_time_ns
    point.stats.static_energy_other_J = energy_j
    return point


def _ideal_candidate_config(identifier: int) -> dict[str, ComponentDVFSConfig]:
    selected = ComponentDVFSConfig(
        policy=DVFSPolicy.IDEAL,
        voltage_V=float(identifier),
        frequency_GHz=1.0,
    )
    fixed = ComponentDVFSConfig(
        policy=DVFSPolicy.NONE,
        voltage_V=0.7,
        frequency_GHz=1.7,
    )
    return {
        "sa": selected,
        "vu": fixed,
        "sram": fixed,
        "hbm_mc": fixed,
        "hbm_die": fixed,
        "hbm_io": fixed,
        "ici_mc": fixed,
        "ici_phy": fixed,
    }


def _legacy_energy_frontier(
    candidates: list[Operator],
) -> list[Operator]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.stats.execution_time_ns,
            candidate.stats.total_energy_J,
        ),
    )
    frontier = []
    for candidate in ordered:
        if (
            not frontier
            or candidate.stats.total_energy_J < frontier[-1].stats.total_energy_J
        ):
            frontier.append(candidate)
    return frontier


def test_parse_ms_policy_keeps_api_default_and_accepts_paper_interval():
    compatibility = dvfs_optimizer.get_global_dvfs_config_helper("DVFSCms_0.1")
    paper = dvfs_optimizer.get_global_dvfs_config_helper(
        "CustomAllms_dom3_0.1_5000000"
    )

    assert compatibility.frequency_adjustment_interval_ns == 1_000_000.0
    assert paper.policy == DVFSPolicy.CUSTOM_ALL_ms
    assert paper.performance_degradation_percentage == 0.1
    assert paper.frequency_adjustment_interval_ns == 5_000_000.0
    assert paper.custom_compute_domain_mode == "dom3"
    assert dvfs_optimizer.PAPER_MS_INTERVAL_NS == 5_000_000.0


def test_weighted_pareto_arrays_have_one_entry_per_original_op():
    pareto_ops = [
        [_point("a0", 10, 4.0), _point("a1", 20, 2.0)],
        [_point("b0", 30, 8.0)],
    ]

    energies, times = dvfs_optimizer._build_weighted_pareto_arrays(
        pareto_ops, instance_weight=3
    )

    assert len(energies) == len(times) == len(pareto_ops) == 2
    np.testing.assert_array_equal(energies[0], np.array([12.0, 6.0]))
    np.testing.assert_array_equal(times[1], np.array([90]))


def test_request_allocator_preserves_timing_counters_with_one_point_per_op():
    ops = [_point("a", 100, 1.0), _point("b", 200, 2.0)]
    pareto = [[op.model_copy(deep=True)] for op in ops]
    timing: dict[str, object] = {}
    config = ModelConfig(model_type="test", output_file_path="")
    dvfs_config = DVFSConfig(policy=DVFSPolicy.CUSTOM)

    with (
        patch.object(
            dvfs_optimizer,
            "generate_pareto_energy_latency_points_for_all_ops",
            return_value=pareto,
        ),
        patch.object(dvfs_optimizer, "_save_search_trace"),
    ):
        result = dvfs_optimizer.configure_dvfs_for_ops(
            ops, config, dvfs_config, timing_result=timing
        )

    assert [op.name for op in result] == ["a", "b"]
    assert timing["avg_pareto_points"] == 1.0
    assert timing["num_expanded_ops"] == 2
    assert "pareto_generation_seconds" in timing
    assert "inter_op_search_seconds" in timing
    assert "inter_op_search_iterations" in timing



def test_no_pareto_batch_reuses_points_sorts_budgets_and_keeps_best():
    op = _point("batched", 100, 1.0)
    config = ModelConfig(model_type="test", output_file_path="")
    selected_point = dvfs_power_getter.SA_POINTS[-2]
    analyzed_point_configs = []

    def fake_analyze(candidate, *_args, **_kwargs):
        if candidate.dvfs_sa.policy == DVFSPolicy.DVFS_C_NO_PARETO:
            analyzed_point_configs.append(candidate)
        return candidate

    candidate_energy = {0.0: 10.0, 0.1: 12.0, 0.2: 8.0}

    def fake_ga(candidate_ops, _config, step_dvfs, *_args, **_kwargs):
        result = [candidate.model_copy(deep=True) for candidate in candidate_ops]
        result[0].stats.static_energy_other_J = candidate_energy[
            step_dvfs.performance_degradation_percentage
        ]
        _kwargs["timing_result"].update(
            {
                "ga_execution_mode": "scalar_exact_batched_ltr",
                "ga_exact_batch_size": 32,
                "ga_exact_batch_size_env": "DVFS_GA_EXACT_BATCH_SIZE",
                "budget_marker": step_dvfs.performance_degradation_percentage,
            }
        )
        return result

    ga = Mock(side_effect=fake_ga)
    ga.last_best_individual = np.array([0], dtype=np.int32)
    ga.last_population = np.zeros((1, 1), dtype=np.int32)

    with (
        patch.dict("os.environ", {"DVFS_PARETO_SERIAL": "1"}),
        patch.object(dvfs_power_getter, "SA_POINTS", [selected_point]),
        patch.object(power_analysis_lib, "analyze_operator_energy", fake_analyze),
        patch.object(dvfs_optimizer, "configure_dvfs_for_op", side_effect=lambda item, *_: item),
        patch.object(dvfs_optimizer, "configure_dvfs_c_with_degradation", ga),
    ):
        results = dvfs_optimizer.configure_dvfs_c_no_pareto_all_budgets(
            [op], config, budgets=[0.2, 0.0, 0.1]
        )

    assert list(results) == [0.0, 0.1, 0.2]
    energies = [results[budget][0].stats.total_energy_J for budget in results]
    assert energies == [10.0, 10.0, 8.0]
    assert len(ga.call_args_list) == 3
    first_kwargs = ga.call_args_list[0].kwargs
    assert first_kwargs["population_size"] == 1000
    assert first_kwargs["max_generations"] == 500
    assert first_kwargs["mutation_prob"] == 0.15
    assert first_kwargs["elitism_count"] == 50
    assert all(
        call.kwargs["_precomputed_points"]
        is ga.call_args_list[0].kwargs["_precomputed_points"]
        for call in ga.call_args_list
    )

    timings = dvfs_optimizer.configure_dvfs_c_no_pareto_all_budgets.last_timings
    assert timings["point_gen_s"] >= 0.0
    assert list(timings["ga_s"]) == [0.0, 0.1, 0.2]
    assert list(timings["ga_details"]) == [0.0, 0.1, 0.2]
    for budget, details in timings["ga_details"].items():
        assert details["ga_execution_mode"] == "scalar_exact_batched_ltr"
        assert details["ga_exact_batch_size"] == 32
        assert details["ga_exact_batch_size_env"] == "DVFS_GA_EXACT_BATCH_SIZE"
        assert details["budget_marker"] == budget
    assert timings["budgets"] == [0.0, 0.1, 0.2]
    assert timings["num_ops"] == 1
    assert timings["num_vf_points_per_op"] == 1
    assert timings["total_s"] >= timings["point_gen_s"]

    assert len(analyzed_point_configs) == 1
    compute = analyzed_point_configs[0]
    compute_vf = {
        (getattr(compute, field).voltage_V, getattr(compute, field).frequency_GHz)
        for field in ("dvfs_sa", "dvfs_vu", "dvfs_sram")
    }
    assert len(compute_vf) == 1



def test_ms_batch_honors_serial_mode_and_reports_shared_precompute_timing():
    op = _point("epoch", 6_000_000, 1.0)
    op.stats.count = 1
    config = ModelConfig(model_type="test", output_file_path="")
    selected_point = dvfs_power_getter.SA_POINTS[-2]

    def fake_analyze(candidate, *_args, **_kwargs):
        candidate.stats.static_energy_other_J = 1.0
        return candidate

    with (
        patch.dict("os.environ", {"DVFS_PARETO_SERIAL": "1"}),
        patch.object(dvfs_power_getter, "SA_POINTS", [selected_point]),
        patch.object(power_analysis_lib, "analyze_operator_energy", fake_analyze),
    ):
        results = dvfs_optimizer.configure_dvfs_c_ms_all_budgets(
            [op],
            config,
            budgets=[0.1, 0.0],
            interval_ns=5_000_000,
            population_size=20,
            max_generations=1,
        )

    assert list(results) == [0.0, 0.1]
    assert all(len(configured) == 1 for configured in results.values())
    timings = dvfs_optimizer.configure_dvfs_c_ms_all_budgets.last_timings
    assert timings["budgets"] == [0.0, 0.1]
    assert timings["num_regions"] == 1
    assert timings["num_genes"] == 1
    assert timings["num_vf_points_per_region"] == 1
    assert list(timings["ga_s"]) == [0.0, 0.1]
    assert timings["total_s"] >= timings["point_gen_s"]

    configured = results[0.0][0]
    compute_vf = {
        (getattr(configured, field).voltage_V, getattr(configured, field).frequency_GHz)
        for field in ("dvfs_sa", "dvfs_vu", "dvfs_sram")
    }
    assert len(compute_vf) == 1


def test_regular_dvfsc_batch_reuses_regular_pareto_and_raw_baseline():
    op = _point("regular", 100, 1.0)
    config = ModelConfig(model_type="test", output_file_path="")
    pareto_point = _point("regular", 140, 0.7)
    generated = Mock(return_value=[[pareto_point]])
    seen: dict[str, object] = {}

    def zero_config(candidate, _config, dvfs_config):
        assert dvfs_config.policy == DVFSPolicy.DVFS_C
        result = candidate.model_copy(deep=True)
        result.stats.execution_time_ns = 105
        return result

    def fake_ga(candidate_ops, _config, step_dvfs, *_args, **kwargs):
        seen["raw_time"] = candidate_ops[0].stats.execution_time_ns
        seen["points"] = kwargs["_precomputed_points"]
        return [candidate.model_copy(deep=True) for candidate in candidate_ops]

    ga = Mock(side_effect=fake_ga)
    ga.last_best_individual = np.array([0], dtype=np.int32)
    ga.last_population = np.zeros((3, 1), dtype=np.int32)

    with (
        patch.object(
            dvfs_optimizer,
            "generate_pareto_energy_latency_points_for_all_ops",
            generated,
        ),
        patch.object(dvfs_optimizer, "configure_dvfs_for_op", zero_config),
        patch.object(
            power_analysis_lib,
            "analyze_operator_energy",
            side_effect=lambda candidate, *_args, **_kwargs: candidate,
        ),
        patch.object(dvfs_optimizer, "configure_dvfs_c_with_degradation", ga),
    ):
        dvfs_optimizer.configure_dvfs_c_no_pareto_all_budgets(
            [op],
            config,
            budgets=[0.1],
            population_size=3,
            max_generations=1,
            elitism_count=1,
            ga_policy=DVFSPolicy.DVFS_C,
        )

    assert generated.call_count == 1
    envelope_config = generated.call_args.args[2]
    assert envelope_config.policy == DVFSPolicy.DVFS_C
    assert envelope_config.performance_degradation_percentage == 1.0
    assert seen["raw_time"] == 100
    assert op.stats.execution_time_ns == 100
    points = seen["points"]
    assert len(points[0]) == 2
    assert points[0][0].stats.execution_time_ns == 105
    timings = dvfs_optimizer.configure_dvfs_c_no_pareto_all_budgets.last_timings
    assert timings["candidate_generation"] == "regular_100pct_pareto"
    assert timings["baseline_semantics"] == "original_raw_trace_time"
    assert timings["raw_baseline_time_ns"] == 100


def test_batched_exhaustive_ideal_matches_legacy_frontier_with_stable_ties():
    specs = {
        1: (20, 5.0),
        2: (10, 9.0),
        3: (10, 9.0),
        4: (15, 8.0),
        5: (15, 7.0),
        6: (12, 10.0),
    }
    configs = [_ideal_candidate_config(identifier) for identifier in specs]
    source_op = _point("ideal-ties", 10, 0.0)
    config = ModelConfig(model_type="test", output_file_path="")
    ideal = DVFSConfig(policy=DVFSPolicy.IDEAL)

    def fake_analyze(candidate, *_args, **_kwargs):
        identifier = int(candidate.dvfs_sa.voltage_V)
        execution_time, energy = specs[identifier]
        candidate.stats.execution_time_ns = execution_time
        candidate.stats.static_energy_other_J = energy
        return candidate

    reference_candidates = [
        fake_analyze(candidate)
        for candidate in dvfs_optimizer._configured_operator_candidates(
            source_op,
            configs,
        )
    ]
    expected = [
        int(candidate.dvfs_sa.voltage_V)
        for candidate in _legacy_energy_frontier(reference_candidates)
    ]
    assert expected == [2, 5, 1]

    for batch_size in (1, 2, 99):
        with (
            patch.dict(
                "os.environ",
                {
                    "DVFS_PARETO_SERIAL": "1",
                    "DVFS_PARETO_BATCH_SIZE": str(batch_size),
                },
            ),
            patch.object(
                dvfs_optimizer,
                "iter_all_dvfs_configs_for_op",
                side_effect=lambda *_args: iter(configs),
            ),
            patch.object(
                power_analysis_lib,
                "analyze_operator_energy",
                side_effect=fake_analyze,
            ),
        ):
            actual = (
                dvfs_optimizer.generate_pareto_energy_latency_points_for_op_exhaustive_search(
                    source_op,
                    config,
                    ideal,
                )
            )
        assert [int(candidate.dvfs_sa.voltage_V) for candidate in actual] == expected


def test_indexed_dvfsc_frontier_keeps_lowest_power_and_earliest_tie():
    def power_point(name: str, execution_time_ns: int, power_w: float) -> Operator:
        return _point(name, execution_time_ns, power_w * execution_time_ns / 1e9)

    indexed = [
        (0, power_point("slow-power", 10, 5.0)),
        (2, power_point("later-tie", 10, 3.0)),
        (1, power_point("earlier-tie", 10, 3.0)),
        (4, power_point("slow-at-20", 20, 4.0)),
        (3, power_point("best-at-20", 20, 2.0)),
    ]

    frontier = dvfs_optimizer._extract_indexed_pareto_front(
        indexed,
        DVFSPolicy.DVFS_C,
    )

    assert [ordinal for ordinal, _ in frontier] == [1, 3]
    assert [candidate.name for _, candidate in frontier] == [
        "earlier-tie",
        "best-at-20",
    ]


def test_exhaustive_ideal_bounds_lazy_ray_batch_window():
    tracker = {
        "yielded_configs": 0,
        "pending_candidates": 0,
        "pending_batches": 0,
        "max_pending_candidates": 0,
        "max_pending_batches": 0,
        "yielded_at_first_get": None,
    }
    source_op = _point("bounded", 100, 0.0)
    config = ModelConfig(model_type="test", output_file_path="")
    ideal = DVFSConfig(policy=DVFSPolicy.IDEAL)

    def lazy_configs():
        for identifier in range(1, 8):
            tracker["yielded_configs"] += 1
            yield _ideal_candidate_config(identifier)

    def fake_analyze(candidate, *_args, **_kwargs):
        identifier = int(candidate.dvfs_sa.voltage_V)
        candidate.stats.execution_time_ns = 100 + identifier
        candidate.stats.static_energy_other_J = float(20 - identifier)
        return candidate

    class FakeRemote:
        def __init__(self, function):
            self.function = function

        def remote(self, *args):
            count = len(args[0])
            tracker["pending_candidates"] += count
            tracker["pending_batches"] += 1
            tracker["max_pending_candidates"] = max(
                tracker["max_pending_candidates"],
                tracker["pending_candidates"],
            )
            tracker["max_pending_batches"] = max(
                tracker["max_pending_batches"],
                tracker["pending_batches"],
            )
            return self.function, args, count

    def fake_get(future):
        function, args, count = future
        if tracker["yielded_at_first_get"] is None:
            tracker["yielded_at_first_get"] = tracker["yielded_configs"]
        result = function(*args)
        tracker["pending_candidates"] -= count
        tracker["pending_batches"] -= 1
        return result

    fake_ray = SimpleNamespace(
        available_resources=lambda: {"CPU": 8},
        remote=lambda function: FakeRemote(function),
        get=fake_get,
    )
    with (
        patch.dict(
            "os.environ",
            {
                "DVFS_PARETO_SERIAL": "0",
                "DVFS_PARETO_BATCH_SIZE": "2",
                "DVFS_PARETO_MAX_INFLIGHT_BATCHES": "3",
            },
        ),
        patch.dict(sys.modules, {"ray": fake_ray}),
        patch.object(
            dvfs_optimizer,
            "iter_all_dvfs_configs_for_op",
            side_effect=lambda *_args: lazy_configs(),
        ),
        patch.object(
            power_analysis_lib,
            "analyze_operator_energy",
            side_effect=fake_analyze,
        ),
    ):
        result = (
            dvfs_optimizer.generate_pareto_energy_latency_points_for_op_exhaustive_search(
                source_op,
                config,
                ideal,
            )
        )

    assert result
    assert tracker["yielded_at_first_get"] == 6
    assert tracker["yielded_at_first_get"] < 7
    assert tracker["max_pending_candidates"] == 6
    assert tracker["max_pending_batches"] == 3
    stats = (
        dvfs_optimizer.generate_pareto_energy_latency_points_for_op_exhaustive_search.last_run_stats
    )
    assert stats["candidate_batch_size"] == 2
    assert stats["inflight_batch_limit"] == 3
    assert stats["submitted_batches"] == 4
    assert stats["max_inflight_candidates"] == 6
    assert stats["num_analyzed_candidates"] == 7


def test_bounded_ray_failure_cancels_remaining_batch_tasks():
    candidates = [_point(f"candidate-{index}", 10, 1.0) for index in range(4)]
    config = ModelConfig(model_type="test", output_file_path="")
    ideal = DVFSConfig(policy=DVFSPolicy.IDEAL)
    cancelled = []

    class FakeRemote:
        def remote(self, *args):
            return object(), args

    fake_ray = SimpleNamespace(
        available_resources=lambda: {"CPU": 8},
        remote=lambda _function: FakeRemote(),
        get=Mock(side_effect=RuntimeError("synthetic Ray failure")),
        cancel=lambda future, force=False: cancelled.append((future, force)),
    )
    run_stats = {}
    with (
        patch.dict(
            "os.environ",
            {
                "DVFS_PARETO_BATCH_SIZE": "1",
                "DVFS_PARETO_MAX_INFLIGHT_BATCHES": "3",
            },
        ),
        patch.dict(sys.modules, {"ray": fake_ray}),
        np.testing.assert_raises_regex(RuntimeError, "synthetic Ray failure"),
    ):
        list(
            dvfs_optimizer._iter_bounded_analyzed_candidate_batches(
                candidates,
                config,
                None,
                ideal,
                False,
                serial=False,
                run_stats=run_stats,
            )
        )

    assert len(cancelled) == 2
    assert all(force is False for _, force in cancelled)
    assert run_stats["cancelled_pending_batches"] == 2


def test_ideal_outer_exhaustive_scheduler_is_sequential_and_preserves_order():
    ops = [_point("first", 10, 1.0), _point("second", 20, 1.0)]
    ops[0].stats.count = 2
    ops[1].stats.count = 3
    config = ModelConfig(model_type="test", output_file_path="")
    ideal = DVFSConfig(policy=DVFSPolicy.IDEAL)
    per_op = Mock(side_effect=lambda op, *_args: [op.name])
    forbidden_ray = SimpleNamespace(
        remote=Mock(side_effect=AssertionError("outer Ray fanout is forbidden"))
    )

    with (
        patch.dict("os.environ", {"DVFS_PARETO_SERIAL": "0"}),
        patch.dict(sys.modules, {"ray": forbidden_ray}),
        patch.object(
            dvfs_optimizer,
            "generate_pareto_energy_latency_points_for_op",
            per_op,
        ),
    ):
        result = dvfs_optimizer.generate_pareto_energy_latency_points_for_all_ops(
            ops,
            config,
            ideal,
        )

    assert result == [["first"], ["second"]]
    assert [call.args[0].name for call in per_op.call_args_list] == ["first", "second"]
    assert all(call.args[4] == 80 for call in per_op.call_args_list)
    forbidden_ray.remote.assert_not_called()
    stats = (
        dvfs_optimizer.generate_pareto_energy_latency_points_for_all_ops.last_run_stats
    )
    assert stats["outer_scheduler"] == "sequential_ideal_operators"
    assert stats["nested_ray_fanout"] is False


def test_nonideal_explicit_exhaustive_scheduler_also_avoids_outer_ray():
    ops = [_point("custom-first", 10, 1.0), _point("custom-second", 20, 1.0)]
    config = ModelConfig(model_type="test", output_file_path="")
    custom = DVFSConfig(policy=DVFSPolicy.CUSTOM)
    per_op = Mock(side_effect=lambda op, *_args: [op.name])
    forbidden_ray = SimpleNamespace(
        remote=Mock(side_effect=AssertionError("outer Ray fanout is forbidden"))
    )

    with (
        patch.dict("os.environ", {"DVFS_PARETO_SERIAL": "0"}),
        patch.dict(sys.modules, {"ray": forbidden_ray}),
        patch.object(
            dvfs_optimizer,
            "generate_pareto_energy_latency_points_for_op",
            per_op,
        ),
    ):
        result = dvfs_optimizer.generate_pareto_energy_latency_points_for_all_ops(
            ops,
            config,
            custom,
            algorithm="exhaustive",
        )

    assert result == [["custom-first"], ["custom-second"]]
    forbidden_ray.remote.assert_not_called()
    stats = (
        dvfs_optimizer.generate_pareto_energy_latency_points_for_all_ops.last_run_stats
    )
    assert stats["outer_scheduler"] == "sequential_exhaustive_operators"
    assert stats["nested_ray_fanout"] is False


def test_balanced_ideal_cap_is_bounded_and_preserves_true_vf_endpoints():
    counts = dvfs_power_getter._balanced_cartesian_counts(
        [24, 24, 26, 26, 26],
        dvfs_power_getter.MAX_EXHAUSTIVE_DVFS_CONFIGS,
    )
    assert math.prod(counts) <= dvfs_power_getter.MAX_EXHAUSTIVE_DVFS_CONFIGS
    assert all(
        2 <= count <= original
        for count, original in zip(
            counts,
            [24, 24, 26, 26, 26],
            strict=True,
        )
    )

    unordered = [
        ComponentDVFSConfig(
            policy=DVFSPolicy.IDEAL,
            voltage_V=0.5 + index / 100,
            frequency_GHz=frequency,
        )
        for index, frequency in enumerate((0.8, 0.1, 0.5, 1.7, 0.3))
    ]
    sampled = dvfs_power_getter._evenly_sample_by_frequency(unordered, 3)
    frequencies = [config.frequency_GHz for config in sampled]
    assert frequencies == sorted(frequencies)
    assert frequencies[0] == 0.1
    assert frequencies[-1] == 1.7


def test_backend_ideal_cartesian_iterator_consumes_combinations_lazily():
    op = _point("lazy-backend", 100, 0.0)
    op.stats.sa_time_ns = 10
    op.stats.vu_time_ns = 10
    op.stats.vmem_time_ns = 10
    op.stats.memory_time_ns = 10
    op.stats.ici_time_ns = 10
    points = [
        ComponentDVFSConfig(
            policy=DVFSPolicy.IDEAL,
            voltage_V=0.7,
            frequency_GHz=frequency,
        )
        for frequency in (0.5, 1.0)
    ]
    real_product = dvfs_power_getter.itertools.product
    yielded = {"combinations": 0}

    def tracking_product(*iterables):
        for combination in real_product(*iterables):
            yielded["combinations"] += 1
            yield combination

    with (
        patch.object(
            dvfs_power_getter,
            "get_all_dvfs_configs_for_component",
            side_effect=lambda *_args: list(points),
        ),
        patch.object(dvfs_power_getter.itertools, "product", tracking_product),
    ):
        iterator = dvfs_power_getter.iter_all_dvfs_configs_for_op(
            op,
            DVFSPolicy.IDEAL,
            perf_degrade_threshold=10.0,
        )
        assert yielded["combinations"] == 0
        first = next(iterator)
        assert yielded["combinations"] == 1

    assert first["sa"].frequency_GHz in (0.5, 1.0)


def test_backend_ideal_iterator_integrates_both_reductions_and_endpoint_cap():
    op = _point("capped-backend", 100, 0.0)
    op.stats.sa_time_ns = 10
    op.stats.vu_time_ns = 10
    op.stats.vmem_time_ns = 10
    op.stats.memory_time_ns = 10
    op.stats.ici_time_ns = 10
    frequencies = (0.8, 0.1, 1.7, 0.5)
    points = [
        ComponentDVFSConfig(
            policy=DVFSPolicy.IDEAL,
            voltage_V=0.5 + index / 100,
            frequency_GHz=frequency,
        )
        for index, frequency in enumerate(frequencies)
    ]

    with (
        patch.object(dvfs_power_getter, "MAX_EXHAUSTIVE_DVFS_CONFIGS", 100),
        patch.object(
            dvfs_power_getter,
            "get_all_dvfs_configs_for_component",
            side_effect=lambda *_args: list(points),
        ),
    ):
        generated = list(
            dvfs_power_getter.iter_all_dvfs_configs_for_op(
                op,
                DVFSPolicy.IDEAL,
                perf_degrade_threshold=100.0,
            )
        )

    stats = dvfs_power_getter.iter_all_dvfs_configs_for_op.last_stats
    assert stats["raw_product_after_budget_filter"] == 4**5
    assert stats["per_voltage_extrema_reduction_applied"] is True
    assert stats["balanced_endpoint_cap_applied"] is True
    assert stats["final_candidate_product"] <= 100
    assert len(generated) == stats["final_candidate_product"]
    for component in ("sa", "vu", "sram", "hbm_mc", "ici_mc"):
        retained = {config[component].frequency_GHz for config in generated}
        assert min(retained) == min(frequencies)
        assert max(retained) == max(frequencies)


def test_empty_ideal_enumeration_uses_legacy_default_fallback():
    op = _point("fallback", 100, 0.0)
    config = ModelConfig(model_type="test", output_file_path="")
    ideal = DVFSConfig(policy=DVFSPolicy.IDEAL)
    default_config = _ideal_candidate_config(1)

    with (
        patch.dict("os.environ", {"DVFS_PARETO_SERIAL": "1"}),
        patch.object(
            dvfs_optimizer,
            "iter_all_dvfs_configs_for_op",
            return_value=iter(()),
        ),
        patch.object(
            dvfs_optimizer,
            "get_dvfs_config",
            return_value=default_config,
        ),
        patch.object(
            power_analysis_lib,
            "analyze_operator_energy",
            side_effect=lambda candidate, *_args, **_kwargs: candidate,
        ),
    ):
        result = (
            dvfs_optimizer.generate_pareto_energy_latency_points_for_op_exhaustive_search(
                op,
                config,
                ideal,
            )
        )

    assert len(result) == 1
    assert result[0].dvfs_sa.voltage_V == 1.0


def test_batched_dump_matches_legacy_sorted_csv_and_failure_is_atomic(tmp_path):
    specs = {
        1: (20, 5.0),
        2: (10, 9.0),
        3: (10, 9.0),
        4: (15, 7.0),
        5: (12, 10.0),
    }
    configs = [_ideal_candidate_config(identifier) for identifier in specs]
    op = _point("dump-order", 100, 0.0)
    output_file = tmp_path / "analysis" / "trace.csv"
    config = ModelConfig(model_type="test", output_file_path=str(output_file))
    ideal = DVFSConfig(policy=DVFSPolicy.IDEAL)

    def fake_analyze(candidate, *_args, **_kwargs):
        identifier = int(candidate.dvfs_sa.voltage_V)
        execution_time, energy = specs[identifier]
        candidate.stats.execution_time_ns = execution_time
        candidate.stats.static_energy_other_J = energy
        return candidate

    reference = [
        fake_analyze(candidate)
        for candidate in dvfs_optimizer._configured_operator_candidates(op, configs)
    ]
    reference.sort(
        key=lambda candidate: (
            candidate.stats.execution_time_ns,
            candidate.stats.total_energy_J,
        )
    )
    expected = io.StringIO(newline="")
    expected_writer = csv.DictWriter(
        expected,
        fieldnames=reference[0].to_csv_dict().keys(),
    )
    expected_writer.writeheader()
    expected_writer.writerows(candidate.to_csv_dict() for candidate in reference)

    environment = {
        "DVFS_PARETO_SERIAL": "1",
        "DVFS_PARETO_BATCH_SIZE": "2",
    }
    with (
        patch.dict("os.environ", environment),
        patch.object(
            dvfs_optimizer,
            "iter_all_dvfs_configs_for_op",
            side_effect=lambda *_args: iter(configs),
        ),
        patch.object(
            power_analysis_lib,
            "analyze_operator_energy",
            side_effect=fake_analyze,
        ),
    ):
        dvfs_optimizer.generate_pareto_energy_latency_points_for_op_exhaustive_search(
            op,
            config,
            ideal,
            dump_pareto_points_to_file=True,
        )

    dump_path = output_file.parent / "pareto_points" / "dump-order_dvfs_pareto_points.csv"
    with open(dump_path, newline="") as dumped:
        assert dumped.read() == expected.getvalue()

    dump_path.write_text("sentinel\n", encoding="utf-8")

    def fail_late(candidate, *_args, **_kwargs):
        if int(candidate.dvfs_sa.voltage_V) == 4:
            raise RuntimeError("synthetic later-batch failure")
        return fake_analyze(candidate)

    with (
        patch.dict("os.environ", environment),
        patch.object(
            dvfs_optimizer,
            "iter_all_dvfs_configs_for_op",
            side_effect=lambda *_args: iter(configs),
        ),
        patch.object(
            power_analysis_lib,
            "analyze_operator_energy",
            side_effect=fail_late,
        ),
        np.testing.assert_raises_regex(
            RuntimeError,
            "synthetic later-batch failure",
        ),
    ):
        dvfs_optimizer.generate_pareto_energy_latency_points_for_op_exhaustive_search(
            op,
            config,
            ideal,
            dump_pareto_points_to_file=True,
        )

    assert dump_path.read_text(encoding="utf-8") == "sentinel\n"


def test_dump_spool_hierarchical_merge_has_bounded_fan_in(tmp_path):
    run_paths = []
    for ordinal in range(10):
        candidate = _point(f"dump-{ordinal}", 10 - ordinal, float(ordinal + 1))
        run_path = tmp_path / f"run-{ordinal}.pickle"
        dvfs_optimizer._write_analyzed_dump_run(
            [(ordinal, candidate)],
            DVFSPolicy.IDEAL,
            str(run_path),
        )
        run_paths.append(str(run_path))

    real_read = dvfs_optimizer._read_analyzed_dump_run
    descriptors = {"active": 0, "maximum": 0}

    def tracked_read(path):
        descriptors["active"] += 1
        descriptors["maximum"] = max(
            descriptors["maximum"],
            descriptors["active"],
        )
        try:
            yield from real_read(path)
        finally:
            descriptors["active"] -= 1

    with patch.object(dvfs_optimizer, "_read_analyzed_dump_run", tracked_read):
        collapsed = dvfs_optimizer._collapse_analyzed_dump_runs(
            run_paths,
            max_fan_in=3,
        )
        records = list(
            dvfs_optimizer.heapq.merge(
                *(dvfs_optimizer._read_analyzed_dump_run(path) for path in collapsed),
                key=lambda record: record[0],
            )
        )

    assert len(collapsed) <= 3
    assert descriptors["maximum"] <= 3
    assert descriptors["active"] == 0
    assert len(records) == 10
    assert [record[0] for record in records] == sorted(record[0] for record in records)
