#!/usr/bin/env python3
"""Plot Figure 21: component energy for power gating combined with eNPU-All."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from common import (
    component_energy,
    load_energy_results,
    mark_incomplete,
    paper_style,
    require_combinations,
    save_figure,
    threshold_pct,
)
from matplotlib.patches import Patch

THRESHOLDS = [0, 2, 5, 10, 20]
DESIGNS = [
    ("NoDVFS", "NoPG", "None"),
    ("NoDVFS", "Full", "PG-Only"),
    ("eNPU-All", "NoPG", "DVFS-Only"),
    ("eNPU-All", "Full", "PG+DVFS"),
]
STACK_COMPONENTS = ("Other", "SA", "SRAM", "VU", "HBM", "ICI")
LEGEND_COMPONENTS = ("SA", "SRAM", "VU", "HBM", "ICI", "Other")
COMPONENT_COLORS = {
    "Other": "#b0b0b0",
    "SA": "#4878cf",
    "SRAM": "#f7dc6f",
    "VU": "#1f5132",
    "HBM": "#e07070",
    "ICI": "#e8e8e8",
}
COMPONENT_HATCHES = {
    "Other": "/",
    "SA": "/",
    "SRAM": "/",
    "VU": "",
    "HBM": "",
    "ICI": "",
}


def plot(
    input_path: Path,
    output: Path,
    requested_model: str | None,
    allow_partial: bool = False,
) -> None:
    rows = load_energy_results(input_path)
    models = sorted({str(r.get("model")) for r in rows if r.get("model")})
    if requested_model is None:
        if len(models) != 1:
            raise ValueError(
                "Figure 21's paper model is ambiguous; pass --model explicitly"
            )
        model = models[0]
    else:
        from common import canonical_model

        model = canonical_model(requested_model)
    rows = [r for r in rows if r.get("model") == model]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("phase"),
                row.get("policy"),
                row.get("pg_strategy", "NoPG"),
                threshold_pct(row),
            )
        ].append(row)
    expected = []
    for phase in ("prefill", "decode"):
        expected.extend(
            [
                (phase, "NoDVFS", "NoPG", 0.0),
                (phase, "NoDVFS", "Full", 0.0),
            ]
        )
        expected.extend(
            (phase, "eNPU-All", pg, float(threshold))
            for pg in ("NoPG", "Full")
            for threshold in THRESHOLDS
        )
    require_combinations(
        grouped,
        expected,
        f"Figure 21 {model} phase/design/threshold matrix",
        allow_partial,
    )

    paper_style(28)
    plt.rcParams.update(
        {
            "font.size": 36,
            "axes.labelsize": 36,
            "axes.titlesize": 36,
            "xtick.labelsize": 28,
            "ytick.labelsize": 28,
            "legend.fontsize": 36,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "legend.handlelength": 2.0,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            "legend.borderpad": 0.4,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16.8, 6.0), sharey=True)
    group_width = len(DESIGNS) + 1.5
    bar_width = 0.85
    for ax, phase in zip(axes, ("prefill", "decode"), strict=False):
        baseline_rows = grouped.get((phase, "NoDVFS", "NoPG", 0.0), [])
        if not baseline_rows:
            raise ValueError(f"missing {model} {phase} NoDVFS/NoPG baseline")
        baseline = mean(float(r["total_energy_J"]) for r in baseline_rows)
        for ti, threshold in enumerate(THRESHOLDS):
            for di, (policy, pg, _label) in enumerate(DESIGNS):
                rs = grouped.get(
                    (
                        phase,
                        policy,
                        pg,
                        0.0 if policy == "NoDVFS" else float(threshold),
                    ),
                    [],
                )
                x = ti * group_width + di
                bottom = 0.0
                for component in STACK_COMPONENTS:
                    value = (
                        mean(component_energy(r, component) for r in rs) / baseline
                        if rs
                        else 0.0
                    )
                    ax.bar(
                        x,
                        value,
                        bar_width,
                        bottom=bottom,
                        color=COMPONENT_COLORS[component],
                        hatch=COMPONENT_HATCHES[component],
                        edgecolor="black",
                        linewidth=0.4,
                    )
                    bottom += value
        centers = [i * group_width + 1.5 for i in range(len(THRESHOLDS))]
        last_bar_right = (
            (len(THRESHOLDS) - 1) * group_width + (len(DESIGNS) - 1) + bar_width
        )
        ax.set_xlim(-0.8, last_bar_right - 0.06)
        ax.set_xticks(centers)
        ax.set_xticklabels([str(v) for v in THRESHOLDS])
        ax.set_ylim(0, 1.05)
        ax.set_title(phase.capitalize(), fontsize=32)
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.3, alpha=0.4)
    axes[0].set_ylabel("Normalized Energy", fontsize=32)
    axes[1].tick_params(axis="y", labelleft=False, length=0)
    fig.text(
        0.5,
        30 / (fig.get_size_inches()[1] * fig.dpi),
        "Performance Degradation Threshold (%)",
        ha="center",
        fontsize=32,
    )
    handles = [
        Patch(
            facecolor=COMPONENT_COLORS[c],
            hatch=COMPONENT_HATCHES[c],
            edgecolor="black",
            linewidth=0.5,
            label=c,
        )
        for c in LEGEND_COMPONENTS
    ]
    legend = fig.legend(
        handles=handles,
        ncol=6,
        loc="upper center",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        fontsize=28,
        bbox_to_anchor=(0.5, 0.97),
        columnspacing=1.0,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(1.0)
    px50 = 50 / (fig.get_size_inches()[1] * fig.dpi)
    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        wspace=0.05,
        bottom=0.14 + px50,
        top=0.70 + px50,
    )
    mark_incomplete(fig, allow_partial)
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="NeuSim PG/DVFS sweep root or normalized energy table",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally results/figures/figure_21_dvfs_power_gating.pdf)",
    )
    parser.add_argument(
        "--model",
        help="Explicit model identifier; required when input has multiple models",
    )
    parser.add_argument(
        "--allow-partial",
        "--allow-incomplete",
        action="store_true",
        help="Allow an explicitly reduced quick-run matrix",
    )
    args = parser.parse_args()
    plot(args.input, args.output, args.model, args.allow_partial)


if __name__ == "__main__":
    main()
