import math

import pytest

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.frontend.dvfs_region_merge import (
    HBM,
    HFC,
    LFC,
    SA,
    VU,
    FrequencyRegion,
    build_initial_regions,
    build_merged_operator_series,
    build_regions,
    build_regions_by,
    component_label_op,
    merge_region_to_operator,
    merge_regions,
)
from neusim.npusim.frontend.Operator import (
    EinsumStatistics,
    Operator,
    OpType,
)


def _merge(labels, counts, lengths, interval_ns):
    initial = build_initial_regions(labels, counts)
    return merge_regions(initial, labels, lengths, interval_ns)


def _op(
    name: str,
    duration_ns: int,
    component: str,
    *,
    count: int = 1,
) -> Operator:
    op = Operator(name=name)
    op.stats.execution_time_ns = duration_ns
    op.stats.count = count
    op.stats.bounded_by = (
        "Compute" if component in {"sa", "vu", "sram"} else "Memory"
    )
    field = {
        "sa": "sa_time_ns",
        "vu": "vu_time_ns",
        "sram": "vmem_time_ns",
        "hbm": "memory_time_ns",
        "ici": "ici_time_ns",
    }[component]
    setattr(op.stats, field, duration_ns)
    return op


def test_component_regions_relabel_by_duration_and_collapse():
    regions = _merge(
        labels=[VU, SA, VU],
        counts=[1, 1, 1],
        lengths=[20.0, 10.0, 30.0],
        interval_ns=25.0,
    )

    assert len(regions) == 1
    assert regions[0].op_indices == [0, 1, 2]
    assert regions[0].label == VU


def test_binary_region_tie_prefers_high_frequency_label():
    regions = _merge(
        labels=[LFC, HFC],
        counts=[1, 1],
        lengths=[2.0, 2.0],
        interval_ns=10.0,
    )

    assert len(regions) == 1
    assert regions[0].label == HFC


def test_region_merging_preserves_count_boundaries():
    regions = _merge(
        labels=[SA, HBM],
        counts=[1, 2],
        lengths=[1.0, 1.0],
        interval_ns=10.0,
    )

    assert [region.op_indices for region in regions] == [[0], [1]]
    assert [region.count for region in regions] == [1, 2]


def test_build_regions_uses_count_weighted_request_lengths():
    ops = [
        _op("compute", 600_000, "sa", count=10),
        _op("memory", 600_000, "hbm", count=10),
    ]

    regions = build_regions(ops, interval_ns=5_000_000)

    assert [region.op_indices for region in regions] == [[0], [1]]
    assert [region.label for region in regions] == [HFC, LFC]
    assert [region.duration_ns for region in regions] == [600_000, 600_000]
    assert [region.request_duration_ns for region in regions] == [
        6_000_000,
        6_000_000,
    ]
    assert [region.repeat_count for region in regions] == [10, 10]


def test_dvfsc_and_enpu_ms_use_distinct_label_granularities():
    ops = [
        _op("sa", 6_000_000, "sa"),
        _op("vu", 6_000_000, "vu"),
    ]

    binary = build_regions(ops, interval_ns=5_000_000)
    component = build_regions_by(
        ops,
        interval_ns=5_000_000,
        label_fn=component_label_op,
    )

    assert [region.op_indices for region in binary] == [[0, 1]]
    assert [region.label for region in binary] == [HFC]
    assert [region.op_indices for region in component] == [[0], [1]]
    assert [region.label for region in component] == [SA, VU]


@pytest.mark.parametrize("right_length", [12.0, 20.0])
def test_short_middle_region_merges_left_when_left_is_not_longer(right_length):
    regions = _merge(
        labels=[SA, VU, HBM],
        counts=[1, 1, 1],
        lengths=[12.0, 1.0, right_length],
        interval_ns=10.0,
    )

    assert [region.op_indices for region in regions] == [[0, 1], [2]]
    assert [region.label for region in regions] == [SA, HBM]


def test_exact_interval_regions_are_not_merged():
    regions = _merge(
        labels=[SA, VU],
        counts=[1, 1],
        lengths=[5.0, 5.0],
        interval_ns=5.0,
    )

    assert [region.op_indices for region in regions] == [[0], [1]]


def test_isolated_short_equal_count_run_remains_under_interval():
    ops = [
        _op("leading", 1_000_000, "sa", count=1),
        _op("body", 1_000_000, "hbm", count=8),
        _op("tail", 1_000_000, "vu", count=1),
    ]

    regions = build_regions_by(
        ops,
        interval_ns=5_000_000,
        label_fn=component_label_op,
    )

    assert [region.op_indices for region in regions] == [[0], [1], [2]]
    assert [region.request_duration_ns for region in regions] == [
        1_000_000,
        8_000_000,
        1_000_000,
    ]


def test_component_ties_use_source_component_order():
    op = _op("tie", 10, "sa")
    op.stats.vu_time_ns = 10

    assert component_label_op(op) == SA


def test_merge_regions_does_not_mutate_initial_regions():
    initial = build_initial_regions([SA, VU], [1, 1])

    merged = merge_regions(initial, [SA, VU], [1.0, 1.0], 5.0)

    assert [region.op_indices for region in initial] == [[0], [1]]
    assert [region.op_indices for region in merged] == [[0, 1]]


@pytest.mark.parametrize("interval", [0.0, -1.0, math.inf, math.nan])
def test_invalid_intervals_are_rejected(interval):
    with pytest.raises(ValueError, match="finite and positive"):
        build_regions([_op("op", 1, "sa")], interval_ns=interval)


def test_invalid_counts_and_durations_are_rejected():
    invalid_count = _op("count", 1, "sa", count=0)
    with pytest.raises(ValueError, match="count must be positive"):
        build_regions([invalid_count], interval_ns=1.0)

    invalid_duration = _op("duration", 1, "sa")
    invalid_duration.stats.execution_time_ns = -1
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_regions([invalid_duration], interval_ns=1.0)


def test_mismatched_public_helper_inputs_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        build_initial_regions([SA], [1, 1])

    with pytest.raises(ValueError, match="same length"):
        merge_regions([], [SA], [], 1.0)


def test_deprecated_physical_merge_aggregates_in_source_order():
    first = Operator(
        name="first",
        description="prefill first",
        op_type=OpType.MXU,
        stats=EinsumStatistics(
            count=3,
            bounded_by="Compute",
            execution_time_ns=10,
            sa_time_ns=10,
            vu_time_ns=1,
            vmem_time_ns=2,
            memory_time_ns=3,
            ici_time_ns=0,
            memory_traffic_bytes=100,
            flop_count=1_000,
            max_vmem_demand_bytes=50,
            num_setpm_sa=1,
            static_energy_sa_J=9.0,
            dynamic_energy_sa_J=7.0,
        ),
    )
    second = Operator(
        name="second",
        description="decode second",
        op_type=OpType.VPU,
        stats=EinsumStatistics(
            count=3,
            bounded_by="Compute",
            execution_time_ns=20,
            sa_time_ns=2,
            vu_time_ns=20,
            vmem_time_ns=3,
            memory_time_ns=4,
            ici_time_ns=5,
            memory_traffic_bytes=200,
            flop_count=2_000,
            max_vmem_demand_bytes=40,
            num_setpm_sa=2,
            static_energy_sa_J=11.0,
            dynamic_energy_sa_J=13.0,
        ),
    )

    merged_ops, regions = build_merged_operator_series(
        [first, second],
        ChipConfig(),
        interval_ns=5_000_000,
    )

    assert len(merged_ops) == 1
    assert [region.op_indices for region in regions] == [[0, 1]]
    assert regions[0].count == 3
    assert regions[0].duration_ns == 30
    assert regions[0].request_duration_ns == 90

    merged = merged_ops[0]
    assert merged.name == "ms_merged[2]:second"
    assert merged.description == (
        "ms_merged[2]:prefill first | decode second"
    )
    assert merged.stats.count == 3
    assert merged.stats.sa_time_ns == 12
    assert merged.stats.vu_time_ns == 21
    assert merged.stats.vmem_time_ns == 5
    assert merged.stats.memory_time_ns == 7
    assert merged.stats.ici_time_ns == 5
    assert merged.stats.execution_time_ns == 21
    assert merged.stats.bounded_by == "Compute"
    assert merged.stats.memory_traffic_bytes == 300
    assert merged.stats.flop_count == 3_000
    assert merged.stats.max_vmem_demand_bytes == 50
    assert merged.stats.num_setpm_sa == 3
    assert merged.stats.static_energy_sa_J == 0.0
    assert merged.stats.dynamic_energy_sa_J == 0.0
    assert merged.stats.flops_util > 0.0
    assert merged.stats.hbm_bw_util > 0.0

    assert first.stats.execution_time_ns == 10
    assert second.stats.execution_time_ns == 20
    assert first.stats.static_energy_sa_J == 9.0
    assert second.stats.dynamic_energy_sa_J == 13.0


def test_physical_merge_rejects_mixed_counts():
    first = _op("first", 10, "sa", count=1)
    second = _op("second", 10, "sa", count=2)
    mixed = FrequencyRegion([0, 1], HFC, count=1)

    with pytest.raises(ValueError, match="uniform operator count"):
        merge_region_to_operator([first, second], mixed, ChipConfig())
