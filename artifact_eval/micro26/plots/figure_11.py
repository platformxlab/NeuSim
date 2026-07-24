#!/usr/bin/env python3
"""Plot Figure 11: DVFS energy saving, power saving, and latency overhead."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from common import (
    MODEL_LABELS,
    MODELS,
    PHASES,
    POLICY_ORDER,
    load_energy_results,
    mark_incomplete,
    paper_style,
    plot_policy_line,
    require_combinations,
    save_figure,
    threshold_pct,
)
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox, offset_copy

THRESHOLDS = [0.0, 2.0, 5.0, 10.0, 20.0, 25.0, 30.0]
SOURCE_MODEL_LABELS = {
    **MODEL_LABELS,
    "deepseekv2-236b": "DeepSeek-V2-236B",
    "deepseekv3-671b": "DeepSeek-V3-671B",
}
SOURCE_POLICY_OVERRIDES = {
    "DVFS-C": {"markersize": 8},
    "eNPU-C": {"markersize": 11},
    "eNPU-All": {"markersize": 6, "linestyle": (0, (5, 2))},
    "Ideal": {"markersize": 9},
}


def _key(row: dict) -> tuple:
    return row.get("model"), row.get("config"), row.get("phase")


def _metric(
    rows: list[dict], baseline: dict[tuple, dict], field: str, saving: bool
) -> float | None:
    values = []
    for row in rows:
        ref = baseline.get(_key(row))
        if not ref or float(ref.get(field, 0.0)) <= 0:
            continue
        ratio = float(row[field]) / float(ref[field])
        values.append((1.0 - ratio if saving else ratio - 1.0) * 100.0)
    return mean(values) if values else None


def plot(input_path: Path, output: Path, allow_partial: bool = False) -> None:
    all_rows = [
        r
        for r in load_energy_results(input_path)
        if r.get("pg_strategy", "NoPG") == "NoPG"
    ]
    baseline = {_key(r): r for r in all_rows if r.get("policy") == "NoDVFS"}
    if not baseline:
        raise ValueError("Figure 11 requires NoDVFS baseline records")
    rows = [r for r in all_rows if r.get("policy") in POLICY_ORDER]
    invalid = [
        row
        for row in [*baseline.values(), *rows]
        if any(
            float(row.get(field, 0.0)) <= 0
            for field in ("total_energy_J", "avg_power_W", "total_exe_time_ns")
        )
    ]
    if invalid:
        raise ValueError(
            f"Figure 11 has {len(invalid)} records with missing/non-positive energy, power, or time"
        )
    require_combinations(
        ((row.get("model"), row.get("phase")) for row in baseline.values()),
        ((model, phase) for model in MODELS for phase in PHASES),
        "Figure 11 NoDVFS baseline matrix",
        allow_partial,
    )
    require_combinations(
        (
            (row.get("model"), row.get("phase"), row.get("policy"), threshold_pct(row))
            for row in rows
        ),
        (
            (model, phase, policy, threshold)
            for model in MODELS
            for phase in PHASES
            for policy in POLICY_ORDER
            for threshold in THRESHOLDS
        ),
        "Figure 11 policy matrix",
        allow_partial,
    )
    metrics = [
        ("total_energy_J", True, "Energy\nSaving (%)"),
        ("avg_power_W", True, "Power\nSaving (%)"),
        ("total_exe_time_ns", False, "Performance\nOverhead (%)"),
    ]

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (row.get("model"), row.get("phase"), row.get("policy"), threshold_pct(row))
        ].append(row)

    paper_style(22)
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
    fig, axes = plt.subplots(3, 8, figsize=(24.4, 10.6), squeeze=False)
    for mi, model in enumerate(MODELS):
        for pi, phase in enumerate(PHASES):
            col = mi * 2 + pi
            for ri, (field, saving, ylabel) in enumerate(metrics):
                ax = axes[ri, col]
                for policy in POLICY_ORDER:
                    xs, ys = [], []
                    for threshold in THRESHOLDS:
                        value = _metric(
                            grouped.get((model, phase, policy, threshold), []),
                            baseline,
                            field,
                            saving,
                        )
                        if value is not None:
                            xs.append(threshold)
                            ys.append(value)
                    if xs:
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
                ax.grid(True, linewidth=0.5, alpha=0.5, linestyle=":")
                ax.tick_params(axis="both", labelsize=22)
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda value, _: f"{value:g}")
                )
                if ri == 0:
                    ax.set_title(phase.capitalize(), fontsize=23)
                if ri < 2:
                    ax.set_xticklabels([])
                if col == 0:
                    ax.set_ylabel(ylabel, labelpad=5, y=0.5, fontsize=23)
                    if ri == 1:
                        label = ax.yaxis.label
                        label.set_transform(
                            offset_copy(
                                label.get_transform(),
                                fig=fig,
                                x=0,
                                y=5 / 300,
                                units="inches",
                            )
                        )
    # Recover the source's per-panel scaling. Updated NeuSim measurements can
    # contain real negative savings, so those panels retain a padded negative
    # range instead of silently applying the source's zero lower bound.
    for ri in range(len(metrics)):
        for col in range(len(MODELS) * len(PHASES)):
            ax = axes[ri, col]
            policy_values = [
                float(value)
                for line in ax.lines
                if line.get_label() in POLICY_ORDER
                for value in line.get_ydata()
            ]
            if ri == 2:
                ax.plot(
                    [0, max(THRESHOLDS)],
                    [0, max(THRESHOLDS)],
                    color="gray",
                    linestyle="--",
                    linewidth=1.0,
                    zorder=0,
                )
            if policy_values and min(policy_values) < 0:
                low, high = min(policy_values), max(policy_values)
                pad = max(0.15 * (high - low), 0.1)
                ax.set_ylim(low - pad, max(0.0, high + pad))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                continue
            peak = max(policy_values, default=0.0)
            target = (
                peak * 1.15
                if peak > (0.01 if ri == 2 else 0.0)
                else (0.5 if ri == 2 else 1.0)
            )
            ticks = MaxNLocator(nbins=3, min_n_ticks=4).tick_values(0, target)
            ticks = [tick for tick in ticks if 0 <= tick <= target + 1e-9]
            top_tick = max(ticks, default=target)
            ax.set_yticks(ticks)
            ax.set_ylim(0, top_tick / 0.85 if top_tick > 0 else 1.0)
    crop_bbox = _source_layout(fig, axes)
    fig._micro26_bbox_inches = crop_bbox
    fig._micro26_save_dpi = 300
    mark_incomplete(fig, allow_partial)
    save_figure(fig, output)


def _source_layout(fig, axes) -> Bbox:
    save_dpi = 300
    subplot_w = 2.8
    row_gap = -0.3
    fig_h = 10.6
    fig.subplots_adjust(
        left=0.04,
        right=0.99,
        wspace=0.40,
        hspace=row_gap / subplot_w,
        bottom=1.0 / fig_h,
        top=1.0 - 1.8 / fig_h,
    )
    fig.canvas.draw()
    scale = save_dpi / fig.dpi
    figure_width = fig.get_size_inches()[0] * fig.dpi
    tick_overhang = []
    for column in range(axes.shape[1]):
        maximum = 0.0
        for row in range(axes.shape[0]):
            axis = axes[row, column]
            axis_left = axis.get_position().x0 * figure_width
            boxes = [
                tick.get_window_extent()
                for tick in axis.get_yticklabels()
                if tick.get_text().strip()
            ]
            if boxes:
                maximum = max(
                    maximum,
                    (axis_left - min(box.x0 for box in boxes)) * scale,
                )
        tick_overhang.append(maximum)
    target_gap = 15.0
    total_gap_fraction = sum(
        (tick_overhang[column + 1] + target_gap) / (figure_width * scale)
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
            x_position += adjusted_width + (tick_overhang[column + 1] + target_gap) / (
                figure_width * scale
            )

    square_height = adjusted_width * fig.get_size_inches()[0] / fig.get_size_inches()[1]
    squashed_height = square_height * 0.88
    for row in range(axes.shape[0]):
        row_bottom = axes[row, 0].get_position().y0 + row * (
            square_height - squashed_height
        )
        for column in range(axes.shape[1]):
            axes[row, column].set_position(
                [positions[column], row_bottom, adjusted_width, squashed_height]
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
    order = ["DVFS-C", "eNPU-C", "eNPU-All", "Ideal"]
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
        bbox_to_anchor=(graph_center, 0.98 - 5 / (fig_h * save_dpi)),
        fontsize=21,
        columnspacing=1.0,
        handletextpad=0.3,
    )
    for handle in legend.legend_handles:
        handle.set_markersize(16)
        handle.set_linewidth(3)
    fig.canvas.draw()
    figure_height = fig.get_size_inches()[1] * fig.dpi
    legend_bottom = legend.get_window_extent().y0 / figure_height
    model_name_y = legend_bottom - (29 / scale) / figure_height
    model_texts = []
    for model_index, model in enumerate(MODELS):
        left = axes[0, model_index * 2].get_position()
        right = axes[0, model_index * 2 + 1].get_position()
        model_texts.append(
            fig.text(
                (left.x0 + right.x1) / 2,
                model_name_y,
                SOURCE_MODEL_LABELS[model],
                ha="center",
                va="top",
                fontsize=23,
            )
        )

    fig.canvas.draw()
    title_top = axes[0, 0].title.get_window_extent().y1 / figure_height
    model_bottom = (
        min(text.get_window_extent().y0 for text in model_texts) / figure_height
    )
    shift = title_top - model_bottom + (9 / scale) / figure_height
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
        fontsize=23,
    )

    fig.canvas.draw()
    margin = 15 / save_dpi
    legend_box = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    xlabel_box = xlabel.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ylabel_box = (
        axes[0, 0]
        .yaxis.label.get_window_extent()
        .transformed(fig.dpi_scale_trans.inverted())
    )
    ytick_boxes = [
        tick.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        for row in range(axes.shape[0])
        for tick in axes[row, 0].get_yticklabels()
        if tick.get_text().strip()
    ]
    leftmost = min([ylabel_box.x0, *(box.x0 for box in ytick_boxes)])
    return Bbox(
        [
            [leftmost - margin, xlabel_box.y0 - margin],
            [
                fig.get_size_inches()[0],
                legend_box.y1 + 30 / save_dpi,
            ],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="NeuSim sweep result root or normalized energy table",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally results/figures/figure_11_dvfs_summary.pdf)",
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
