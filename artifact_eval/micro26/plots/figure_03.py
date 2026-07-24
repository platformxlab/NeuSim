#!/usr/bin/env python3
"""Plot Figure 3: request-level component utilization."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from common import (
    MODEL_LABELS,
    MODELS,
    PHASES,
    load_energy_results,
    mark_incomplete,
    paper_style,
    require_combinations,
    save_figure,
)

SERIES = [
    ("SA Temp.", ("sa_util", "sa_temp_util"), "#c0392b"),
    ("VU Temp.", ("vu_util", "vu_temp_util"), "#e67e22"),
    ("SRAM BW", ("sram_util", "sram_temp_util", "vmem_temp_util"), "#f1c40f"),
    # Source-faithful despite the paper legend: component_utilization.py reads
    # hbm_temp_util for the series labelled "HBM BW". Do not substitute the
    # separately recorded hbm_bw_util without explicitly revising the paper.
    ("HBM BW", ("hbm_temp_util",), "#2471a3"),
    ("ICI BW", ("ici_util", "ici_temp_util"), "#27ae60"),
]


def _value(rows: list[dict], keys: tuple[str, ...]) -> float:
    vals = []
    for row in rows:
        for key in keys:
            if key in row:
                vals.append(float(row[key]))
                break
    if not vals:
        return 0.0
    value = mean(vals)
    return value * 100.0 if abs(value) <= 1.0 else value


def plot(input_path: Path, output: Path, allow_partial: bool = False) -> None:
    rows = load_energy_results(input_path)
    rows = [
        r
        for r in rows
        if r.get("policy", "NoDVFS") == "NoDVFS"
        and r.get("pg_strategy", "NoPG") == "NoPG"
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row.get("model"), row.get("phase"))].append(row)
    if not grouped:
        raise ValueError("Figure 3 needs NoDVFS utilization records")
    require_combinations(
        (
            (model, phase, label)
            for (model, phase), records in grouped.items()
            for label, keys, _ in SERIES
            if any(any(key in row for key in keys) for row in records)
        ),
        (
            (model, phase, label)
            for model in MODELS
            for phase in PHASES
            for label, _, _ in SERIES
        ),
        "Figure 3 utilization matrix",
        allow_partial,
    )

    paper_style(16)
    plt.rcParams["xtick.major.pad"] = 8
    plt.rcParams["ytick.major.pad"] = 8
    fig, axes = plt.subplots(2, 1, figsize=(12.0, 7.0), sharex=False)
    width = 0.2
    group_step = len(SERIES) * width + 0.15
    x = np.arange(len(MODELS), dtype=float) * group_step
    for ax, phase in zip(axes, PHASES, strict=False):
        for index, (label, keys, color) in enumerate(SERIES):
            vals = [_value(grouped.get((model, phase), []), keys) for model in MODELS]
            ax.bar(
                x + (index - 2) * width,
                vals,
                width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.8,
            )
            for xpos, val in zip(x + (index - 2) * width, vals, strict=False):
                ax.text(
                    xpos,
                    val + 3.0,
                    f"{val:.0f}" if val >= 0.01 else "0",
                    ha="center",
                    va="bottom",
                    fontsize=18,
                    rotation=0,
                )
        ax.set_title(phase.capitalize(), fontsize=18, pad=10)
        ax.set_xticks(x)
        labels = [MODEL_LABELS[model].rsplit("-", 1) for model in MODELS]
        ax.set_xticklabels(["\n".join(parts) for parts in labels], fontsize=18)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 130)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.tick_params(axis="y", labelsize=18)
        ax.set_ylabel("Utilization (%)", fontsize=18)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="lightgray", linestyle="solid", linewidth=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=True,
        edgecolor="black",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=18,
        columnspacing=2.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.4)
    mark_incomplete(fig, allow_partial)
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="NeuSim result root or normalized utilization CSV/JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally results/figures/figure_03_component_utilization.pdf)",
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
