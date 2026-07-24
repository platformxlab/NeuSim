#!/usr/bin/env python3
"""Plot Figure 13: IVR conversion-energy overhead from per-operator CSVs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from common import (
    MODEL_LABELS,
    MODELS,
    PHASES,
    POLICY_ORDER,
    canonical_policy,
    infer_path_metadata,
    load_table,
    mark_incomplete,
    paper_style,
    plot_policy_line,
    require_combinations,
    save_figure,
    threshold_pct,
)
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox

THRESHOLDS = [0.0, 2.0, 5.0, 10.0, 20.0, 25.0, 30.0]
PAPER_MODEL_LABELS = {
    "llama3-70b": "Llama3-70B",
    "llama3_1-405b": "Llama3.1-405B",
    "deepseekv2-236b": "DeepSeek-V2-236B",
    "deepseekv3-671b": "DeepSeek-V3-671B",
}
PAPER_Y_AXES = {
    "prefill": ((6.125, 9.875), [6.5, 8.0, 9.5]),
    "decode": ((6.25, 8.75), [6.5, 7.5, 8.5]),
}
SOURCE_POLICY_OVERRIDES = {
    "NoDVFS": {"markersize": 8},
    "DVFS-C": {"markersize": 8},
    "eNPU-C": {"markersize": 11},
    "eNPU-All": {"markersize": 6, "linestyle": (0, (5, 2))},
    "Ideal": {"markersize": 9},
}


def _set_phase_y_axis(ax, phase: str, values: list[float]) -> None:
    """Use paper limits when possible, otherwise keep every measured value visible."""
    limits, ticks = PAPER_Y_AXES[phase]
    finite = [value for value in values if math.isfinite(value)]
    if not finite or (min(finite) >= limits[0] and max(finite) <= limits[1]):
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
        return

    data_min, data_max = min(finite), max(finite)
    spread = data_max - data_min
    magnitude = max(abs(data_min), abs(data_max), 1.0)
    padding = max(0.08 * spread, 0.04 * magnitude, 0.1)
    lower = data_min - padding
    upper = data_max + padding
    if data_max < 0:
        upper = max(0.0, upper)
    ax.set_ylim(lower, upper)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


SCHEMA_SPLIT = [
    ("DVFS SA", "sa"),
    ("DVFS VU", "vu"),
    ("DVFS SRAM", "sram"),
    ("DVFS HBM MC", "hbm_mc"),
    ("DVFS HBM DIE", "hbm_die"),
    ("DVFS HBM IO", "hbm_io"),
    ("DVFS ICI MC", "ici_mc"),
    ("DVFS ICI PHY", "ici_phy"),
]
SCHEMA_LEGACY = [
    ("DVFS SA", "sa"),
    ("DVFS VU", "vu"),
    ("DVFS SRAM", "sram"),
    ("DVFS HBM MC", "hbm_mc"),
    ("DVFS HBM PHY", "hbm_phy"),
    ("DVFS ICI MC", "ici_mc"),
    ("DVFS ICI PHY", "ici_phy"),
]


def _native_records(root: Path) -> list[dict]:
    files = [root] if root.is_file() else sorted(root.rglob("inference-v*.csv"))
    out = []
    for path in files:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            schema = (
                SCHEMA_SPLIT
                if "DVFS HBM DIE Voltage (V)" in (reader.fieldnames or [])
                else SCHEMA_LEGACY
            )
            loss = total = 0.0
            for row in reader:
                count = float(row.get("Count", 1) or 1)
                for prefix, suffix in schema:
                    efficiency = float(
                        row.get(f"{prefix} Power Efficiency (%)", 100) or 100
                    )
                    energy = float(row.get(f"static_energy_{suffix}_J", 0) or 0)
                    energy += float(row.get(f"dynamic_energy_{suffix}_J", 0) or 0)
                    loss += count * energy * (1.0 - efficiency / 100.0)
                total += count * float(row.get("total_energy_J", 0) or 0)
        if total:
            meta = infer_path_metadata(path)
            meta["ivr_overhead_pct"] = 100.0 * loss / total
            out.append(meta)
    return out


def plot(input_path: Path, output: Path, allow_partial: bool = False) -> None:
    if input_path.is_file() and input_path.suffix.lower() == ".json":
        try:
            rows = load_table(input_path)
        except ValueError:
            rows = _native_records(input_path)
    else:
        rows = _native_records(input_path)
    if not rows or not any("ivr_overhead_pct" in r for r in rows):
        raise ValueError(
            "Figure 13 needs per-operator IVR columns or ivr_overhead_pct records"
        )
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        policy = canonical_policy(str(row.get("policy", "NoDVFS")))
        grouped[
            (row.get("model"), row.get("phase"), policy, threshold_pct(row))
        ].append(float(row["ivr_overhead_pct"]))
    require_combinations(
        grouped,
        (
            (model, phase, policy, threshold)
            for model in MODELS
            for phase in PHASES
            for policy in ["NoDVFS", *POLICY_ORDER]
            for threshold in ([0.0] if policy == "NoDVFS" else THRESHOLDS)
        ),
        "Figure 13 IVR-overhead matrix",
        allow_partial,
    )

    paper_style(21)
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 8.0), sharex=True)
    for ri, phase in enumerate(PHASES):
        for ci, model in enumerate(MODELS):
            ax = axes[ri, ci]
            panel_values: list[float] = []
            for policy in ["NoDVFS", *POLICY_ORDER]:
                if policy == "NoDVFS":
                    vals = grouped.get((model, phase, policy, 0.0), [])
                    if vals:
                        baseline = mean(vals)
                        panel_values.append(baseline)
                        plot_policy_line(
                            ax,
                            [0, 30],
                            [baseline] * 2,
                            policy,
                            linewidth=2.0,
                            **SOURCE_POLICY_OVERRIDES[policy],
                        )
                    continue
                xs, ys = [], []
                for threshold in THRESHOLDS:
                    vals = grouped.get((model, phase, policy, threshold), [])
                    if vals:
                        xs.append(threshold)
                        ys.append(mean(vals))
                if xs:
                    panel_values.extend(ys)
                    plot_policy_line(
                        ax,
                        xs,
                        ys,
                        policy,
                        linewidth=2.0,
                        **SOURCE_POLICY_OVERRIDES[policy],
                    )
            ax.set_xlim(-0.9, 32.4)
            ax.set_xticks([0, 10, 20, 30])
            _set_phase_y_axis(ax, phase, panel_values)
            ax.grid(True, linewidth=0.5, alpha=0.5, linestyle=":")
            ax.tick_params(axis="both", labelsize=22)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")
            if ri == 0:
                ax.tick_params(axis="x", labelbottom=False)
            if ci == 0:
                ylabel_y = 0.536 if phase == "prefill" else 0.476
                ax.set_ylabel(
                    f"{phase.capitalize()}\nEnergy\nOverhead\u2005(%)",
                    labelpad=5,
                    y=ylabel_y,
                    fontsize=23,
                )
            else:
                ax.tick_params(axis="y", labelleft=False)
    crop_bbox = _source_layout(fig, axes)
    fig._micro26_bbox_inches = crop_bbox
    fig._micro26_save_dpi = 300
    mark_incomplete(fig, allow_partial)
    save_figure(fig, output)


def _source_layout(fig, axes) -> Bbox:
    save_dpi = 300
    fig_h = 8.0
    fig.subplots_adjust(
        left=0.04,
        right=0.99,
        wspace=0.40,
        hspace=0.30,
        bottom=1.0 / fig_h,
        top=1.0 - 1.4 / fig_h,
    )
    fig.canvas.draw()
    scale = save_dpi / fig.dpi
    figure_width = fig.get_size_inches()[0] * fig.dpi
    tick_overhang = []
    for column in range(axes.shape[1]):
        axis = axes[0, column]
        axis_left = axis.get_position().x0 * figure_width
        boxes = [
            tick.get_window_extent()
            for tick in axis.get_yticklabels()
            if tick.get_text().strip()
        ]
        tick_overhang.append(
            (axis_left - min(box.x0 for box in boxes)) * scale if boxes else 0.0
        )

    column_gap = 80.0
    total_gap_fraction = sum(
        (tick_overhang[column + 1] + column_gap) / (figure_width * scale)
        for column in range(axes.shape[1] - 1)
    )
    first_overhang_fraction = tick_overhang[0] / (figure_width * scale)
    adjusted_width = (
        0.99 - 0.04 - first_overhang_fraction - total_gap_fraction
    ) / axes.shape[1]
    positions = []
    x_position = 0.04 + first_overhang_fraction
    for column in range(axes.shape[1]):
        positions.append(x_position)
        if column < axes.shape[1] - 1:
            x_position += adjusted_width + (tick_overhang[column + 1] + column_gap) / (
                figure_width * scale
            )

    square_height = adjusted_width * fig.get_size_inches()[0] / fig.get_size_inches()[1]
    squashed_height = square_height * 0.88 * 0.9 * 0.9
    row_gap_fraction = 70 / (fig.get_size_inches()[1] * fig.dpi * scale)
    top_row_bottom = axes[0, 0].get_position().y0
    row_positions = [
        top_row_bottom,
        top_row_bottom - squashed_height - row_gap_fraction,
    ]
    for row in range(axes.shape[0]):
        for column in range(axes.shape[1]):
            axes[row, column].set_position(
                [
                    positions[column],
                    row_positions[row],
                    adjusted_width,
                    squashed_height,
                ]
            )

    graph_center = (
        axes[0, 0].get_position().x0
        + axes[0, -1].get_position().x0
        + axes[0, -1].get_position().width
    ) / 2
    handle_by_label = {}
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        for handle, label in zip(handles, labels, strict=True):
            handle_by_label.setdefault(label, handle)
    order = ["NoDVFS", "DVFS-C", "eNPU-C", "eNPU-All", "Ideal"]
    ordered_labels = [label for label in order if label in handle_by_label]
    legend = fig.legend(
        [handle_by_label[label] for label in ordered_labels],
        ordered_labels,
        loc="upper center",
        ncol=max(1, len(ordered_labels)),
        frameon=True,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        bbox_to_anchor=(graph_center, 0.92 - 5 / (fig_h * save_dpi)),
        fontsize=21,
        columnspacing=1.0,
        handletextpad=0.3,
    )
    for handle in legend.legend_handles:
        handle.set_markersize(16)
        handle.set_linewidth(3)

    fig.canvas.draw()
    figure_height = fig.get_size_inches()[1] * fig.dpi
    top_axes_y = max(
        axes[0, column].get_position().y1 for column in range(axes.shape[1])
    )
    model_texts = []
    for column, model in enumerate(MODELS):
        position = axes[0, column].get_position()
        model_texts.append(
            fig.text(
                position.x0 + position.width / 2,
                top_axes_y + 5 / figure_height,
                PAPER_MODEL_LABELS.get(model, MODEL_LABELS[model]),
                ha="center",
                va="bottom",
                fontsize=21,
            )
        )

    fig.canvas.draw()
    legend_bottom = legend.get_window_extent().y0 / figure_height
    model_top = max(text.get_window_extent().y1 for text in model_texts) / figure_height
    shift = model_top - legend_bottom + (17 / scale) / figure_height
    if shift > 0:
        for row in range(axes.shape[0]):
            for column in range(axes.shape[1]):
                position = axes[row, column].get_position()
                axes[row, column].set_position(
                    [
                        position.x0,
                        position.y0 - shift,
                        position.width,
                        position.height,
                    ]
                )
        for text in model_texts:
            x_value, y_value = text.get_position()
            text.set_position((x_value, y_value - shift))

    model_nudge = (5 / scale) / figure_height
    for text in model_texts:
        x_value, y_value = text.get_position()
        text.set_position((x_value, y_value + model_nudge))
    fig.canvas.draw()
    tick_boxes = [
        tick.get_window_extent()
        for column in range(axes.shape[1])
        for tick in axes[-1, column].get_xticklabels()
        if tick.get_window_extent().height > 0
    ]
    xlabel_y = min(box.y0 for box in tick_boxes) / figure_height if tick_boxes else 0.05
    xlabel = fig.text(
        graph_center,
        xlabel_y - (15 / scale) / figure_height,
        "Performance Degradation Threshold (%)",
        ha="center",
        va="top",
        fontsize=25,
    )

    fig.canvas.draw()
    margin = 15 / save_dpi
    legend_box = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    xlabel_box = xlabel.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ylabel_boxes = [
        axes[row, 0]
        .yaxis.label.get_window_extent()
        .transformed(fig.dpi_scale_trans.inverted())
        for row in range(axes.shape[0])
    ]
    ytick_boxes = [
        tick.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        for row in range(axes.shape[0])
        for tick in axes[row, 0].get_yticklabels()
        if tick.get_text().strip()
    ]
    leftmost = min(
        [
            *(box.x0 for box in ylabel_boxes),
            *(box.x0 for box in ytick_boxes),
        ]
    )
    return Bbox(
        [
            [leftmost - margin, xlabel_box.y0 - margin],
            [
                fig.get_size_inches()[0],
                legend_box.y1 + 5 / save_dpi,
            ],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Per-operator CSV/result root, or normalized ivr_overhead_pct table",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally results/figures/figure_13_ivr_overhead.pdf)",
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
