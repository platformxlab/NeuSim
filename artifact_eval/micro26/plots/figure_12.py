#!/usr/bin/env python3
"""Plot Figure 12: Llama3-70B component-level normalized energy."""

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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

POLICIES = ["NoDVFS", "DVFS-C", "eNPU-C", "eNPU-All", "Ideal"]
THRESHOLDS = {"prefill": [0, 2, 5, 10, 20], "decode": [0, 2]}
STACK_COMPONENTS = ["Other", "SA", "SRAM", "VU", "HBM", "ICI"]
LEGEND_COMPONENTS = ["SA", "SRAM", "VU", "HBM", "ICI", "Other"]
PAPER_COMPONENT_COLORS = {
    "Other": "#b0b0b0",
    "SA": "#4878cf",
    "VU": "#1f5132",
    "SRAM": "#f7dc6f",
    "HBM": "#e07070",
    "ICI": "#e8e8e8",
}
PAPER_COMPONENT_HATCHES = {
    "Other": "/",
    "SA": "/",
    "VU": "",
    "SRAM": "/",
    "HBM": "",
    "ICI": "",
}
GROUP_WIDTH = len(POLICIES) + 1.5
BAR_WIDTH = 0.85


def _xlim_range(num_thresholds: int) -> float:
    """Return the original paper renderer's proportional panel width."""
    last_bar_right = (
        (num_thresholds - 1) * GROUP_WIDTH + (len(POLICIES) - 1) + BAR_WIDTH
    )
    return (last_bar_right - 0.06) - (-0.8)


def plot(input_path: Path, output: Path, allow_partial: bool = False) -> None:
    rows = [
        r
        for r in load_energy_results(input_path)
        if r.get("model") == "llama3-70b" and r.get("pg_strategy", "NoPG") == "NoPG"
    ]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("phase"), row.get("policy"), threshold_pct(row))].append(row)
    expected = []
    for phase in ("prefill", "decode"):
        expected.append((phase, "NoDVFS", 0.0))
        expected.extend(
            (phase, policy, float(threshold))
            for policy in POLICIES
            if policy != "NoDVFS"
            for threshold in THRESHOLDS[phase]
        )
    require_combinations(
        grouped,
        expected,
        "Figure 12 phase/policy/threshold matrix",
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
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
        }
    )
    fig = plt.figure(figsize=(16.8, 6.0))
    spec = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[
            _xlim_range(len(THRESHOLDS["prefill"])),
            _xlim_range(len(THRESHOLDS["decode"])),
        ],
    )
    axes = [fig.add_subplot(spec[0, 0]), fig.add_subplot(spec[0, 1])]
    for column, (ax, phase) in enumerate(
        zip(axes, ("prefill", "decode"), strict=False)
    ):
        base_rows = grouped.get((phase, "NoDVFS", 0.0), [])
        if not base_rows:
            raise ValueError(f"missing Llama3-70B {phase} NoDVFS baseline")
        baseline = mean(float(r["total_energy_J"]) for r in base_rows)
        for ti, threshold in enumerate(THRESHOLDS[phase]):
            for pi, policy in enumerate(POLICIES):
                rs = (
                    base_rows
                    if policy == "NoDVFS"
                    else grouped.get((phase, policy, float(threshold)), [])
                )
                x = ti * GROUP_WIDTH + pi
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
                        BAR_WIDTH,
                        bottom=bottom,
                        color=PAPER_COMPONENT_COLORS[component],
                        hatch=PAPER_COMPONENT_HATCHES[component],
                        edgecolor="black",
                        linewidth=0.4,
                    )
                    bottom += value
        centers = [
            i * GROUP_WIDTH + (len(POLICIES) - 1) / 2
            for i in range(len(THRESHOLDS[phase]))
        ]
        ax.set_xticks(centers)
        ax.set_xticklabels([str(v) for v in THRESHOLDS[phase]])
        ax.set_ylim(0, 1.05)
        last_bar_right = (
            (len(THRESHOLDS[phase]) - 1) * GROUP_WIDTH + (len(POLICIES) - 1) + BAR_WIDTH
        )
        ax.set_xlim(-0.8, last_bar_right - 0.06)
        ax.set_title(phase.capitalize(), fontsize=32)
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.3, alpha=0.4)
        if column == 0:
            ax.set_ylabel("Normalized Energy", fontsize=32)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
    component_handles = [
        Patch(
            facecolor=PAPER_COMPONENT_COLORS[c],
            hatch=PAPER_COMPONENT_HATCHES[c],
            edgecolor="black",
            linewidth=0.5,
            label=c,
        )
        for c in LEGEND_COMPONENTS
    ]
    fig.legend(
        handles=component_handles,
        ncol=6,
        loc="upper center",
        frameon=True,
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        bbox_to_anchor=(0.5, 0.97),
        fontsize=28,
    )
    pixel_50 = 50 / (fig.get_size_inches()[1] * fig.dpi)
    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        wspace=0.05,
        bottom=0.14 + pixel_50,
        top=0.70 + pixel_50,
    )
    fig.text(
        0.5,
        30 / (fig.get_size_inches()[1] * fig.dpi),
        "Performance Degradation Threshold (%)",
        ha="center",
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
        help="NeuSim sweep result root or normalized energy table",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally results/figures/figure_12_component_energy.pdf)",
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
