"""Operator-region construction for millisecond-scale DVFS policies.

The authoritative trace-util implementation uses a count-weighted one-layer
dataflow model. It first forms maximal, count-compatible runs of operators
with the same policy label, then repeatedly absorbs the globally shortest
under-interval region into its shorter immediate same-count neighbor.

build_regions is the DVFS-C-ms entry point and uses binary HFC/LFC labels.
eNPU-ms uses build_regions_by with component_label_op to distinguish the five
bottleneck components.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass

import neusim.npusim.frontend.Operator as Operator

PAPER_MS_INTERVAL_NS = 5_000_000.0

HFC = "HFC"
LFC = "LFC"

SA = "SA"
VU = "VU"
SRAM = "SRAM"
HBM = "HBM"
ICI = "ICI"


def label_op(op: Operator.Operator) -> str:
    """Return HFC for a compute-bound op and LFC otherwise."""
    return HFC if op.stats.bounded_by == "Compute" else LFC


def component_label_op(op: Operator.Operator) -> str:
    """Return the op's bottleneck component, using source-compatible tie order."""
    stats = op.stats
    pairs = [
        (stats.sa_time_ns, SA),
        (stats.vu_time_ns, VU),
        (stats.vmem_time_ns, SRAM),
        (stats.memory_time_ns, HBM),
        (stats.ici_time_ns, ICI),
    ]
    return max(pairs, key=lambda pair: pair[0])[1]


def op_length_ns(op: Operator.Operator) -> float:
    """Return request time represented by this deduplicated operator row."""
    return float(op.stats.execution_time_ns) * int(op.stats.count)


@dataclass
class FrequencyRegion:
    """A contiguous, equal-count group of operator indices.

    The first three fields match trace-util's public helper. duration_ns is the
    single-occurrence duration retained for NeuSim compatibility; the
    authoritative merge comparisons use request_duration_ns.
    """

    op_indices: list[int]
    label: str
    count: int
    duration_ns: float = 0.0

    @property
    def repeat_count(self) -> int:
        """Compatibility alias used by NeuSim's request optimizers."""
        return self.count

    @property
    def request_duration_ns(self) -> float:
        """Count-weighted duration represented by this deduplicated region."""
        return self.duration_ns * self.count

    @property
    def start_index(self) -> int:
        return self.op_indices[0]

    @property
    def end_index(self) -> int:
        return self.op_indices[-1]


# Compatibility name used by NeuSim's eNPU-ms module.
DVFSRegion = FrequencyRegion


def _length(region: FrequencyRegion, lengths: Sequence[float]) -> float:
    return sum(lengths[index] for index in region.op_indices)


def _relabel(
    region: FrequencyRegion,
    labels: Sequence[str],
    lengths: Sequence[float],
) -> None:
    duration_by_label: dict[str, float] = {}
    label_order: list[str] = []
    for index in region.op_indices:
        label = labels[index]
        if label not in duration_by_label:
            duration_by_label[label] = 0.0
            label_order.append(label)
        duration_by_label[label] += lengths[index]

    if set(duration_by_label).issubset({HFC, LFC}):
        region.label = (
            HFC
            if duration_by_label.get(HFC, 0.0) >= duration_by_label.get(LFC, 0.0)
            else LFC
        )
        return

    # Component-labelled ties retain the first component encountered in the
    # region, matching trace-util's stable max over label_order.
    region.label = max(label_order, key=duration_by_label.__getitem__)


def build_initial_regions(
    labels: Sequence[str],
    counts: Sequence[int],
) -> list[FrequencyRegion]:
    """Group consecutive rows sharing both label and count into maximal runs."""
    if len(labels) != len(counts):
        raise ValueError(
            "labels and counts must have the same length: "
            f"{len(labels)} != {len(counts)}"
        )

    regions: list[FrequencyRegion] = []
    for index, (label, count) in enumerate(zip(labels, counts, strict=True)):
        if regions and regions[-1].label == label and regions[-1].count == count:
            regions[-1].op_indices.append(index)
        else:
            regions.append(
                FrequencyRegion(
                    op_indices=[index],
                    label=label,
                    count=int(count),
                )
            )
    return regions


def _collapse_adjacent(
    regions: Sequence[FrequencyRegion],
) -> list[FrequencyRegion]:
    """Collapse neighboring regions sharing both their current label and count."""
    if not regions:
        return []

    output = [
        FrequencyRegion(
            list(regions[0].op_indices),
            regions[0].label,
            regions[0].count,
        )
    ]
    for region in regions[1:]:
        if region.label == output[-1].label and region.count == output[-1].count:
            output[-1].op_indices.extend(region.op_indices)
        else:
            output.append(
                FrequencyRegion(
                    list(region.op_indices),
                    region.label,
                    region.count,
                )
            )
    return output


def merge_regions(
    regions: Sequence[FrequencyRegion],
    labels: Sequence[str],
    lengths: Sequence[float],
    interval_ns: float,
) -> list[FrequencyRegion]:
    """Greedily merge source-compatible, count-aware frequency regions.

    At each iteration, regions are considered globally shortest first. The
    first region below interval_ns that has an immediate same-count neighbor
    is merged into the shorter eligible neighbor (left on a tie). The result
    is relabelled by original-label duration and adjacent equal label/count
    regions are collapsed. An isolated short count run remains.
    """
    if len(labels) != len(lengths):
        raise ValueError(
            "labels and lengths must have the same length: "
            f"{len(labels)} != {len(lengths)}"
        )
    interval = float(interval_ns)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError(
            f"interval_ns must be finite and positive, got {interval_ns!r}"
        )

    merged_regions = [
        FrequencyRegion(
            list(region.op_indices),
            region.label,
            int(region.count),
        )
        for region in regions
    ]
    for region in merged_regions:
        if any(index < 0 or index >= len(labels) for index in region.op_indices):
            raise ValueError(
                f"region contains an out-of-range operator index: {region.op_indices}"
            )

    while len(merged_regions) > 1:
        order = sorted(
            range(len(merged_regions)),
            key=lambda index: _length(merged_regions[index], lengths),
        )
        chosen: tuple[int, int] | None = None
        for source in order:
            if _length(merged_regions[source], lengths) >= interval:
                break
            left = (
                source - 1
                if source > 0
                and merged_regions[source - 1].count == merged_regions[source].count
                else None
            )
            right = (
                source + 1
                if source < len(merged_regions) - 1
                and merged_regions[source + 1].count == merged_regions[source].count
                else None
            )
            if left is None and right is None:
                continue
            if left is None:
                target = right
            elif right is None:
                target = left
            else:
                target = (
                    left
                    if _length(merged_regions[left], lengths)
                    <= _length(merged_regions[right], lengths)
                    else right
                )
            assert target is not None
            chosen = (source, target)
            break

        if chosen is None:
            break

        source, target = chosen
        low, high = (source, target) if source < target else (target, source)
        combined = FrequencyRegion(
            op_indices=(
                merged_regions[low].op_indices + merged_regions[high].op_indices
            ),
            label=HFC,
            count=merged_regions[source].count,
        )
        _relabel(combined, labels, lengths)
        merged_regions = merged_regions[:low] + [combined] + merged_regions[high + 1 :]
        merged_regions = _collapse_adjacent(merged_regions)

    return merged_regions


def build_regions_by(
    ops: Sequence[Operator.Operator],
    interval_ns: float,
    label_fn: Callable[[Operator.Operator], str],
) -> list[FrequencyRegion]:
    """Build regions with an explicit policy labelling function."""
    interval = float(interval_ns)
    if not math.isfinite(interval) or interval <= 0.0:
        raise ValueError(
            f"interval_ns must be finite and positive, got {interval_ns!r}"
        )
    if not ops:
        return []

    counts: list[int] = []
    durations: list[float] = []
    for op in ops:
        count = int(op.stats.count)
        duration = float(op.stats.execution_time_ns)
        if count <= 0:
            raise ValueError(f"operator count must be positive, got {count}")
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError(
                "operator execution_time_ns must be finite and non-negative, "
                f"got {op.stats.execution_time_ns!r}"
            )
        counts.append(count)
        durations.append(duration)

    labels = [label_fn(op) for op in ops]
    lengths = [
        duration * count for duration, count in zip(durations, counts, strict=True)
    ]
    regions = merge_regions(
        build_initial_regions(labels, counts),
        labels,
        lengths,
        interval,
    )
    for region in regions:
        region.duration_ns = sum(durations[index] for index in region.op_indices)

    flattened = tuple(index for region in regions for index in region.op_indices)
    if flattened != tuple(range(len(ops))):
        raise RuntimeError(
            "frequency-region construction reordered, duplicated, or dropped "
            f"operators: {flattened}"
        )
    return regions


def build_regions(
    ops: Sequence[Operator.Operator],
    interval_ns: float = PAPER_MS_INTERVAL_NS,
) -> list[FrequencyRegion]:
    """Build authoritative DVFS-C-ms regions with binary HFC/LFC labels."""
    return build_regions_by(ops, interval_ns, label_op)


_ENERGY_PREFIXES = ("static_energy_", "dynamic_energy_")


def merge_region_to_operator(
    ops: Sequence[Operator.Operator],
    region: FrequencyRegion,
    config,
) -> Operator.Operator:
    """Physically merge one region into a source-compatible Operator.

    Integer primary statistics are additive, except max_vmem_demand_bytes,
    which takes the maximum. Energy fields are cleared for later analysis.
    Execution time and bounded_by are reconstructed from the summed component
    times, and component utilizations are refreshed.

    Active millisecond-policy callers evaluate the original member operators
    and only share a frequency plan; this physical helper is retained for
    trace-util API compatibility.
    """
    from neusim.npusim.frontend.op_analysis_lib import (
        analyze_operator_component_util,
    )

    if not region.op_indices:
        raise ValueError("cannot merge an empty frequency region")
    members = [ops[index] for index in region.op_indices]
    distinct_counts = {int(member.stats.count) for member in members}
    if len(distinct_counts) != 1:
        raise ValueError(
            "merge requires a uniform operator count, got " f"{sorted(distinct_counts)}"
        )

    dominant = max(
        members,
        key=lambda member: member.stats.execution_time_ns,
    )
    merged = deepcopy(dominant)
    stats = merged.stats

    for field_name, field_info in type(stats).model_fields.items():
        if field_name in {
            "count",
            "execution_time_ns",
            "bounded_by",
            "parsed_op_type",
        }:
            continue
        if any(field_name.startswith(prefix) for prefix in _ENERGY_PREFIXES):
            setattr(stats, field_name, 0.0)
            continue
        if field_info.annotation is int:
            values = [getattr(member.stats, field_name, 0) for member in members]
            if field_name == "max_vmem_demand_bytes":
                setattr(stats, field_name, max(values))
            else:
                setattr(stats, field_name, sum(values))

    stats.count = int(members[0].stats.count)
    execution_time = max(
        stats.sa_time_ns,
        stats.vu_time_ns,
        stats.vmem_time_ns,
        stats.memory_time_ns,
        stats.ici_time_ns,
    )
    stats.execution_time_ns = int(execution_time)
    bounded_by = "ICI/NVLink"
    for component_time, label in (
        (stats.sa_time_ns, "Compute"),
        (stats.vu_time_ns, "Compute"),
        (stats.vmem_time_ns, "Compute"),
        (stats.memory_time_ns, "Memory"),
        (stats.ici_time_ns, "ICI/NVLink"),
    ):
        if component_time == execution_time:
            bounded_by = label
            break
    stats.bounded_by = bounded_by

    merged.description = f"ms_merged[{len(members)}]:" + " | ".join(
        member.description for member in members
    )
    merged.name = f"ms_merged[{len(members)}]:{dominant.name}"
    analyze_operator_component_util(merged, config)
    return merged


def build_merged_operator_series(
    ops: Sequence[Operator.Operator],
    config,
    interval_ns: float = PAPER_MS_INTERVAL_NS,
) -> tuple[list[Operator.Operator], list[FrequencyRegion]]:
    """Deprecated physical-merging adapter retained for source compatibility.

    A merged operator computes max(sum(component times)), which can understate
    the sequential sum(max(component times)). Active DVFS-C-ms and eNPU-ms
    paths therefore keep member operators separate and share only their plan.
    """
    regions = build_regions(ops, interval_ns)
    merged_ops = [merge_region_to_operator(ops, region, config) for region in regions]
    return merged_ops, regions
