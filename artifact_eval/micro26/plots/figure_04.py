#!/usr/bin/env python3
"""Plot Figure 4: component utilization over request execution time."""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import reduce
from math import gcd
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from common import (
    MODEL_LABELS,
    count_weight,
    load_csv_records,
    mark_incomplete,
    paper_style,
    require_combinations,
    save_figure,
)
from matplotlib.patches import Patch

PANELS = [
    ("llama3-70b", "prefill", 50.0, (0.0, 20.0, 40.0), 0.005),
    ("llama3-70b", "decode", 500.0, (0.0, 200.0, 400.0), 0.005),
    ("llama3_1-405b", "prefill", 30.0, (0.0, 10.0, 20.0), 0.001),
    ("llama3_1-405b", "decode", 10.0, (0.0, 4.0, 8.0), 0.005),
    ("deepseekv3-671b", "prefill", 35.0, (0.0, 15.0, 30.0), 0.005),
    ("deepseekv3-671b", "decode", 1100.0, (0.0, 500.0, 1000.0), 0.025),
]
SERIES = [
    ("SA temporal", "MXU time", "#c0392b", "solid"),
    ("VU temporal", "VPU time", "#e67e22", "dashed"),
    ("SRAM BW", "Vmem time", "#f1c40f", "dotted"),
    ("HBM BW", "Memory time", "#2471a3", (0, (4, 1, 1, 1))),
    ("ICI BW", "ICI/NVLink time", "#27ae60", (0, (1, 1))),
]
COMPONENT_COLUMNS = {label: column for label, column, _, _ in SERIES}
SERIES_DISPLAY = {
    "SA temporal": "SA Temporal Util.",
    "VU temporal": "VU Temporal Util.",
    "SRAM BW": "SRAM BW Util.",
    "HBM BW": "HBM BW Util.",
    "ICI BW": "ICI BW Util.",
}
NUM_RESAMPLE_POINTS = 2000
DIRECT_PLOT_THRESHOLD = 5000


def _parse_pp(config: str) -> int:
    """Extract ICI x DCN pipeline parallelism from a config name."""
    pp_ici = 1
    pp_dcn = 1
    for part in config.split("-"):
        if part.startswith("ppdcn"):
            try:
                pp_dcn = int(part[5:])
            except ValueError:
                pass
        elif part.startswith("pp"):
            try:
                pp_ici = int(part[2:])
            except ValueError:
                pass
    return pp_ici * pp_dcn


def _row_name(row: dict[str, Any]) -> str:
    return str(row.get("operator_name", row.get("Name", row.get("Op Name", ""))))


def _reconstruct_pipeline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand a single-stage trace to its full pipeline and drop overlapped transfers."""
    configs = {str(row.get("config", "")) for row in rows if row.get("config")}
    if len(configs) > 1:
        raise ValueError(
            f"Figure 4 panel mixes pipeline configurations: {sorted(configs)}"
        )
    total_pp = _parse_pp(next(iter(configs), ""))
    if total_pp <= 1:
        return rows
    output = []
    for row in rows:
        if "pipeline" in _row_name(row).lower():
            continue
        copied = dict(row)
        copied["Count"] = count_weight(row) * total_pp
        output.append(copied)
    return output


def _op(row: dict[str, Any]) -> dict[str, Any] | None:
    duration = float(row.get("Execution time", row.get("execution_time_ns", 0.0)))
    if duration <= 0:
        return None
    return {
        "duration": duration,
        "count": max(1, int(round(count_weight(row)))),
        "util": {
            label: min(100.0, max(0.0, 100.0 * float(row.get(column, 0.0)) / duration))
            for label, column in COMPONENT_COLUMNS.items()
        },
    }


def _decompose(
    ops: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], int]:
    """Identify leading/trailing ops and one repeated layer using count GCD."""
    layer_counts = [op["count"] for op in ops if op["count"] > 1]
    if not layer_counts:
        return ops, [], [], 1
    iterations = reduce(gcd, layer_counts)
    if iterations <= 1:
        return ops, [], [], 1
    first = next(
        (index for index, op in enumerate(ops) if op["count"] >= iterations),
        len(ops),
    )
    last = (
        len(ops)
        - 1
        - next(
            (
                index
                for index, op in enumerate(reversed(ops))
                if op["count"] >= iterations
            ),
            len(ops),
        )
    )
    return ops[:first], ops[first : last + 1], ops[last + 1 :], iterations


def _flat_arrays(
    ops: list[dict[str, Any]], divide_count: int = 1
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    durations: list[float] = []
    values = {label: [] for label in COMPONENT_COLUMNS}
    for op in ops:
        repetitions = op["count"] // divide_count
        durations.extend([op["duration"]] * repetitions)
        for label in values:
            values[label].extend([op["util"][label]] * repetitions)
    return np.asarray(durations), {
        label: np.asarray(series) for label, series in values.items()
    }


def _lookup(
    boundaries: np.ndarray, values: dict[str, np.ndarray], times: np.ndarray
) -> dict[str, np.ndarray]:
    indexes = np.searchsorted(boundaries, times, side="right") - 1
    indexes = np.clip(indexes, 0, len(boundaries) - 2)
    return {label: series[indexes] for label, series in values.items()}


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _to_steps(
    durations: np.ndarray, values: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    boundaries = np.concatenate(([0.0], np.cumsum(durations)))
    xs = np.empty(2 * len(durations))
    xs[0::2] = boundaries[:-1]
    xs[1::2] = boundaries[1:]
    return xs, {label: np.repeat(series, 2) for label, series in values.items()}


def _timeline(
    rows: list[dict[str, Any]],
    xmax_ms: float | None = None,
    smoothing_fraction: float = 0.005,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Reconstruct repeated-layer order without physically unrolling large counts."""
    rows = _reconstruct_pipeline(rows)
    ops = [converted for row in rows if (converted := _op(row)) is not None]
    if not ops:
        return np.asarray([]), {label: np.asarray([]) for label in COMPONENT_COLUMNS}

    leading, layer, trailing, iterations = _decompose(ops)
    lead_durations, lead_values = _flat_arrays(leading)
    iter_durations, iter_values = _flat_arrays(layer, iterations)
    trail_durations, trail_values = _flat_arrays(trailing)
    logical_ops = (
        len(lead_durations) + iterations * len(iter_durations) + len(trail_durations)
    )
    total_ns = (
        lead_durations.sum() + iterations * iter_durations.sum() + trail_durations.sum()
    )
    if total_ns <= 0:
        return np.asarray([]), {label: np.asarray([]) for label in COMPONENT_COLUMNS}

    if logical_ops <= DIRECT_PLOT_THRESHOLD:
        durations = np.concatenate(
            (lead_durations, np.tile(iter_durations, iterations), trail_durations)
        )
        values = {
            label: np.concatenate(
                (
                    lead_values[label],
                    np.tile(iter_values[label], iterations),
                    trail_values[label],
                )
            )
            for label in COMPONENT_COLUMNS
        }
        window = max(1, int(round(smoothing_fraction * 2 * logical_ops)))
        xs, step_values = _to_steps(durations, values)
        return xs / 1e6, {
            label: _smooth(series, window) for label, series in step_values.items()
        }

    visible_ns = min(total_ns, xmax_ms * 1e6) if xmax_ms is not None else total_ns
    grid_ns = np.linspace(0.0, visible_ns, NUM_RESAMPLE_POINTS)
    values = {label: np.zeros(NUM_RESAMPLE_POINTS) for label in COMPONENT_COLUMNS}
    lead_total = lead_durations.sum()
    iter_total = iter_durations.sum()
    layer_end = lead_total + iterations * iter_total

    lead_mask = grid_ns < lead_total
    if lead_mask.any():
        sampled = _lookup(
            np.concatenate(([0.0], np.cumsum(lead_durations))),
            lead_values,
            grid_ns[lead_mask],
        )
        for label in values:
            values[label][lead_mask] = sampled[label]

    layer_mask = (grid_ns >= lead_total) & (grid_ns < layer_end)
    if layer_mask.any() and iter_total > 0:
        within = np.mod(grid_ns[layer_mask] - lead_total, iter_total)
        sampled = _lookup(
            np.concatenate(([0.0], np.cumsum(iter_durations))),
            iter_values,
            within,
        )
        for label in values:
            values[label][layer_mask] = sampled[label]

    trail_mask = grid_ns >= layer_end
    if trail_mask.any() and len(trail_durations):
        sampled = _lookup(
            np.concatenate(([0.0], np.cumsum(trail_durations))),
            trail_values,
            grid_ns[trail_mask] - layer_end,
        )
        for label in values:
            values[label][trail_mask] = sampled[label]

    window = max(1, int(NUM_RESAMPLE_POINTS * smoothing_fraction))
    return grid_ns / 1e6, {
        label: _smooth(series, window) for label, series in values.items()
    }


def plot(input_path: Path, output: Path, allow_partial: bool = False) -> None:
    rows = load_csv_records(input_path)
    rows = [row for row in rows if row.get("policy", "NoDVFS") == "NoDVFS"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model"), row.get("phase"))].append(row)
    require_combinations(
        (
            key
            for key, records in grouped.items()
            if any(
                float(row.get("Execution time", row.get("execution_time_ns", 0.0))) > 0
                for row in records
            )
        ),
        ((model, phase) for model, phase, *_ in PANELS),
        "Figure 4 timeline panel matrix",
        allow_partial,
    )

    paper_style(16)
    plt.rcParams["xtick.major.pad"] = 8
    plt.rcParams["ytick.major.pad"] = 8
    fig, axes = plt.subplots(1, 6, figsize=(36.0, 6.42), squeeze=False)
    for ax, (model, phase, xmax, xticks, smoothing_fraction) in zip(
        axes.flat, PANELS, strict=False
    ):
        panel_rows = grouped.get((model, phase), [])
        if not panel_rows:
            ax.text(
                0.5,
                0.5,
                "missing input",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            xs, util = _timeline(
                panel_rows,
                xmax_ms=xmax,
                smoothing_fraction=smoothing_fraction,
            )
            for label, _, color, linestyle in SERIES:
                ax.plot(
                    xs,
                    util[label],
                    color=color,
                    linestyle=linestyle,
                    label=label,
                    linewidth=1.5,
                )
                ax.fill_between(xs, util[label], color=color, alpha=0.12)
        ax.set_xlim(0, xmax)
        ax.set_xticks(xticks)
        ax.set_ylim(0, 105)
        ax.set_title(phase.capitalize(), fontsize=32)
        ax.tick_params(axis="both", labelsize=34)
        ax.set_axisbelow(True)
        ax.grid(True, color="lightgray", linestyle="solid", linewidth=0.5)
        if ax is axes[0, 0]:
            ax.set_ylabel("Utilization (%)", fontsize=36)
        else:
            ax.tick_params(axis="y", labelleft=False, left=False)
    handles = [
        Patch(
            facecolor=mcolors.to_rgba(color, alpha=0.3),
            edgecolor=color,
            linewidth=3.5,
            linestyle=linestyle,
            label=SERIES_DISPLAY[label],
        )
        for label, _, color, linestyle in SERIES
    ]
    fig.legend(
        handles=handles,
        ncol=5,
        loc="upper center",
        frameon=True,
        edgecolor="black",
        bbox_to_anchor=(0.5, 0.984),
        fontsize=30,
        handlelength=2.5,
        handleheight=0.8,
        handletextpad=0.5,
    )
    fig.tight_layout(rect=(0, 0.086, 1, 0.754))
    fig.subplots_adjust(wspace=0.08)
    fig.text(0.5, 0.039, "Time (ms)", ha="center", fontsize=36)
    for index, model in enumerate(("llama3-70b", "llama3_1-405b", "deepseekv3-671b")):
        pair = axes[0, index * 2 : index * 2 + 2]
        positions = [axis.get_position() for axis in pair]
        center = (positions[0].x0 + positions[-1].x1) / 2
        top = max(position.y1 for position in positions)
        fig.text(
            center,
            top + 0.083,
            MODEL_LABELS[model],
            ha="center",
            va="bottom",
            fontsize=32,
        )
    mark_incomplete(fig, allow_partial)
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="NoDVFS inference CSV or directory containing the six panel CSVs",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally results/figures/figure_04_utilization_timeline.pdf)",
    )
    parser.add_argument(
        "--allow-partial",
        "--allow-incomplete",
        action="store_true",
        help="Allow an explicitly reduced quick-run matrix",
    )
    args = parser.parse_args()
    plot(args.input, args.output, args.allow_partial)


if __name__ == "__main__":
    main()
