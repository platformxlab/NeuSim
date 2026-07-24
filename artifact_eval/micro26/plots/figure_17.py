#!/usr/bin/env python3
"""Plot Figure 17: sensitivity to V/f-domain count.

Raw domain-sweep values are the default.  ``--paper-presentation-adjustments``
reproduces the legacy plot-only -2 pp four-domain offset and three-domain clamp;
the figure is visibly labeled when that opt-in is used.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from common import (
    MODEL_LABELS,
    MODELS,
    PHASES,
    load_energy_results,
    load_table,
    mark_incomplete,
    paper_style,
    require_combinations,
    save_figure,
    threshold_pct,
)
from matplotlib.ticker import MaxNLocator

MODES = ("dom5", "dom4_savu", "dom3")
DOMAIN_THRESHOLDS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
REFERENCE_THRESHOLDS = [0.0, 2.0, 5.0, 10.0, 20.0, 25.0, 30.0]
MODE_LABELS = {"dom5": "5 Domains", "dom4_savu": "4 Domains", "dom3": "3 Domains"}
SOURCE_MODEL_LABELS = {
    **MODEL_LABELS,
    "deepseekv2-236b": "DeepSeek-V2-236B",
    "deepseekv3-671b": "DeepSeek-V3-671B",
}
MODE_STYLE = {
    "dom5": dict(
        color="#e8750a",
        marker="D",
        linestyle="-",
        markerfacecolor="white",
        markeredgecolor="#e8750a",
        markeredgewidth=1.6,
        markersize=10,
        linewidth=2.2,
    ),
    "dom4_savu": dict(
        color="#1f77b4",
        marker="s",
        linestyle="--",
        markerfacecolor="#1f77b4",
        markeredgecolor="#1f77b4",
        markeredgewidth=0.9,
        markersize=9,
        linewidth=2.2,
    ),
    "dom3": dict(
        color="#7f7f7f",
        marker="v",
        linestyle="-",
        markerfacecolor="#7f7f7f",
        markeredgecolor="#7f7f7f",
        markeredgewidth=0.9,
        markersize=9,
        linewidth=2.2,
    ),
}


def _domain_json(path: Path) -> list[dict]:
    with path.open() as handle:
        doc = json.load(handle)
    if not (isinstance(doc, dict) and isinstance(doc.get("results"), dict)):
        return load_table(path)
    rows = []
    none = doc["none"]
    for key, modes in doc["results"].items():
        model, config, phase, perf = key.split("|")
        baseline = none.get(f"{model}|{config}|{phase}")
        if not baseline or float(baseline[0]) <= 0:
            continue
        for mode, pair in modes.items():
            rows.append(
                {
                    "model": model,
                    "config": config,
                    "phase": phase,
                    "threshold_pct": float(perf) * 100.0,
                    "mode": mode,
                    "energy_saving_pct": (1.0 - float(pair[0]) / float(baseline[0]))
                    * 100.0,
                }
            )
    return rows


def _reference_rows(path: Path | None) -> list[dict]:
    if path is None:
        return []
    rows = [
        row
        for row in load_energy_results(path)
        if row.get("pg_strategy", "NoPG") == "NoPG"
    ]
    baselines = [row for row in rows if row.get("policy") == "NoDVFS"]
    output = []
    for row in rows:
        if row.get("policy") not in ("Ideal", "DVFS-C"):
            continue
        candidates = [
            base
            for base in baselines
            if base.get("model") == row.get("model")
            and base.get("phase") == row.get("phase")
        ]
        if row.get("config") is not None:
            exact = [
                base for base in candidates if base.get("config") == row.get("config")
            ]
            if exact:
                candidates = exact
        energies = [float(base.get("total_energy_J", 0.0)) for base in candidates]
        energies = [energy for energy in energies if energy > 0]
        if energies:
            output.append(
                {
                    **row,
                    "energy_saving_pct": (
                        1.0 - float(row["total_energy_J"]) / mean(energies)
                    )
                    * 100.0,
                }
            )
    return output


def _interp(points: list[tuple[float, float]], x: float) -> float | None:
    if not points:
        return None
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return None


def plot(
    input_path: Path,
    output: Path,
    reference_input: Path | None,
    adjustments: bool,
    allow_partial: bool = False,
) -> None:
    rows = _domain_json(input_path)
    refs = _reference_rows(reference_input)
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (row.get("model"), row.get("phase"), row.get("mode"), threshold_pct(row))
        ].append(float(row["energy_saving_pct"]))
    ref_grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in refs:
        ref_grouped[
            (row.get("model"), row.get("phase"), row.get("policy"), threshold_pct(row))
        ].append(float(row["energy_saving_pct"]))
    if not grouped:
        raise ValueError("Figure 17 needs domain-sweep records")
    require_combinations(
        grouped,
        (
            (model, phase, mode, threshold)
            for model in MODELS
            for phase in PHASES
            for mode in MODES
            for threshold in DOMAIN_THRESHOLDS
        ),
        "Figure 17 domain-count matrix",
        allow_partial,
    )
    require_combinations(
        (
            (model, phase, threshold)
            for model, phase, policy, threshold in ref_grouped
            if policy == "Ideal"
        ),
        (
            (model, phase, threshold)
            for model in MODELS
            for phase in PHASES
            for threshold in REFERENCE_THRESHOLDS
        ),
        "Figure 17 Ideal reference matrix (pass --reference-input)",
        allow_partial,
    )
    if adjustments:
        require_combinations(
            (
                (model, phase, threshold)
                for model, phase, policy, threshold in ref_grouped
                if policy == "DVFS-C"
            ),
            (
                (model, phase, threshold)
                for model in MODELS
                for phase in PHASES
                for threshold in REFERENCE_THRESHOLDS
            ),
            "Figure 17 DVFS-C floor reference",
        )
    thresholds = (
        sorted({key[3] for key in grouped}) if allow_partial else DOMAIN_THRESHOLDS
    )

    paper_style(17)
    plt.rcParams.update(
        {
            "axes.labelsize": 17,
            "axes.titlesize": 16,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 17,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.minor.size": 2,
            "ytick.minor.size": 2,
            "lines.linewidth": 2.2,
            "lines.markersize": 9,
            "legend.handlelength": 2.0,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            "legend.borderpad": 0.4,
        }
    )
    fig_w, fig_h = 3.3 * len(MODELS), 5.6
    size_scale = 0.9
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(fig_w * size_scale, fig_h * size_scale),
        sharex=True,
        sharey="row",
    )
    row_values: list[list[float]] = [[] for _ in PHASES]
    for ri, phase in enumerate(PHASES):
        for ci, model in enumerate(MODELS):
            ax = axes[ri, ci]
            floor_points = [
                (t, mean(ref_grouped[(model, phase, "DVFS-C", t)]))
                for t in REFERENCE_THRESHOLDS
                if ref_grouped.get((model, phase, "DVFS-C", t))
            ]
            for mode in MODES:
                xs, ys = [], []
                for threshold in thresholds:
                    vals = grouped.get((model, phase, mode, threshold), [])
                    if not vals:
                        continue
                    value = mean(vals)
                    if adjustments and mode == "dom4_savu":
                        value -= 2.0
                    if adjustments and mode == "dom3":
                        floor = _interp(floor_points, threshold)
                        if floor is not None:
                            value = max(value, floor)
                    xs.append(threshold)
                    ys.append(value)
                if xs:
                    row_values[ri].extend(ys)
                    ax.plot(xs, ys, label=MODE_LABELS[mode], **MODE_STYLE[mode])
            ideal_x, ideal_y = [], []
            for threshold in sorted({k[3] for k in ref_grouped}):
                vals = ref_grouped.get((model, phase, "Ideal", threshold), [])
                if vals:
                    ideal_x.append(threshold)
                    ideal_y.append(mean(vals))
            if ideal_x:
                row_values[ri].extend(ideal_y)
                ax.plot(
                    ideal_x,
                    ideal_y,
                    color="#8b0000",
                    marker="",
                    linestyle=(0, (1, 1)),
                    markeredgewidth=0.9,
                    markeredgecolor="#8b0000",
                    markerfacecolor="#8b0000",
                    markersize=11,
                    linewidth=2.2,
                    label="Ideal",
                )
            ax.set_xlim(
                -0.03 * max(DOMAIN_THRESHOLDS),
                1.08 * max(DOMAIN_THRESHOLDS),
            )
            ax.set_xticks([0, 10, 20, 30])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(True, linewidth=0.5, alpha=0.5, linestyle=":")
            if ri == 0:
                ax.set_title(SOURCE_MODEL_LABELS[model])
            if ci == 0:
                ax.set_ylabel(
                    f"{phase.capitalize()}\nEnergy\nSaving (%)",
                    linespacing=1.2,
                    labelpad=4,
                )
            else:
                ax.tick_params(axis="y", labelleft=False)
    for row_axes, values in zip(axes, row_values, strict=False):
        if not values:
            continue
        row_min, row_max = min(values), max(values)
        if row_min >= 0.0:
            row_axes[0].set_ylim(0.0, 1.2 * row_max if row_max > 0.0 else 1.0)
        else:
            span = max(
                row_max - row_min,
                0.05 * max(abs(row_min), abs(row_max), 1.0),
            )
            padding = 0.2 * span
            row_axes[0].set_ylim(row_min - padding, row_max + padding)
    left, right = 0.10, 0.995
    graph_center = (left + right) / 2
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        bbox_to_anchor=(graph_center, 1.03),
        columnspacing=1.0,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(0.8)
    for legend_handle in legend.legend_handles:
        legend_handle.set_markersize(9)
        legend_handle.set_linewidth(2.2)
    fig.text(
        graph_center,
        0.10 / fig_h,
        "Performance Degradation Threshold (%)",
        ha="center",
    )
    if adjustments:
        fig.text(
            0.995,
            0.005,
            "Legacy presentation adjustments enabled",
            ha="right",
            va="bottom",
            color="#a00",
            fontsize=7,
        )
    fig.subplots_adjust(
        left=left,
        right=right,
        wspace=0.08,
        hspace=0.16,
        bottom=0.72 / fig_h,
        top=1.0 - 0.92 / fig_h,
    )
    mark_incomplete(fig, allow_partial)
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="domain_count_customall_results.json or normalized domain table",
    )
    parser.add_argument(
        "--reference-input",
        type=Path,
        help="Optional sweep result root/table providing Ideal (and DVFS-C for opt-in clamp)",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally results/figures/figure_17_domain_count.pdf)",
    )
    parser.add_argument(
        "--paper-presentation-adjustments",
        action="store_true",
        help="Opt in to the legacy -2 pp offset and DVFS-C floor; never enabled by default",
    )
    parser.add_argument(
        "--allow-partial",
        "--allow-incomplete",
        action="store_true",
        help="Allow an explicitly reduced quick-run matrix",
    )
    args = parser.parse_args()
    plot(
        args.input,
        args.output,
        args.reference_input,
        args.paper_presentation_adjustments,
        args.allow_partial,
    )


if __name__ == "__main__":
    main()
