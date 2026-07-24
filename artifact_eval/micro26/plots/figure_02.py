#!/usr/bin/env python3
"""Plot Figure 2: normalized NoDVFS request-energy breakdown."""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from common import (
    MODEL_LABELS,
    MODELS,
    PHASES,
    component_energy,
    load_energy_results,
    mark_incomplete,
    paper_style,
    require_combinations,
    save_figure,
)

PAPER_COMPONENTS = ("SA", "VU", "SRAM", "ICI", "HBM")
STACKS = (
    *(
        ("static", component, f"Static {component}", "//")
        for component in PAPER_COMPONENTS
    ),
    *(
        ("dynamic", component, f"Dynamic {component}", "")
        for component in PAPER_COMPONENTS
    ),
    (None, "Other", "Other", ""),
)
SOURCE_COMPONENT_COLORS = {
    "SA": "#D8801C",
    "VU": "#B05050",
    "SRAM": "#F0DC8C",
    "ICI": "#2D4880",
    "HBM": "#95D095",
    "Other": "#F5F5F5",
}


def plot(input_path: Path, output: Path, allow_partial: bool = False) -> None:
    rows = load_energy_results(input_path)
    baseline = [
        r
        for r in rows
        if r.get("policy", "NoDVFS") == "NoDVFS"
        and r.get("pg_strategy", "NoPG") == "NoPG"
    ]
    if not baseline:
        raise ValueError("Figure 2 needs NoDVFS/NoPG energy records")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in baseline:
        grouped[(row.get("model"), row.get("phase"))].append(row)
    invalid = [
        key
        for key, records in grouped.items()
        if any(float(row.get("total_energy_J", 0.0)) <= 0 for row in records)
    ]
    if invalid:
        raise ValueError(f"Figure 2 has non-positive request energy for {invalid}")
    require_combinations(
        grouped,
        ((model, phase) for model in MODELS for phase in PHASES),
        "Figure 2 model/phase matrix",
        allow_partial,
    )

    paper_style(20)
    plt.rcParams["hatch.linewidth"] = 0.4
    plt.rcParams["hatch.color"] = "black"
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 5.0), sharex=True)
    y = (np.arange(len(MODELS)) * 1.5)[::-1]
    for ax, phase in zip(axes, PHASES, strict=False):
        left = np.zeros(len(MODELS))
        for kind, component, label, hatch in STACKS:
            values = []
            for model in MODELS:
                rs = grouped.get((model, phase), [])
                total = mean(float(r["total_energy_J"]) for r in rs) if rs else 0.0
                value = (
                    mean(component_energy(r, component, kind) for r in rs)
                    if rs
                    else 0.0
                )
                values.append(100.0 * value / total if total else 0.0)
            ax.barh(
                y,
                values,
                left=left,
                color=SOURCE_COMPONENT_COLORS[component],
                edgecolor="black",
                linewidth=0.1,
                hatch=hatch,
                label=label,
            )
            left += np.asarray(values)
        ax.set_title(phase.capitalize())
        ax.set_xlim(0, 100)
        ticks = [0, 20, 40, 60, 80, 100]
        ax.set_xticks(ticks, [f"{value}%" for value in ticks], fontsize=18)
        ax.grid(axis="x", color="lightgray", linestyle="solid")
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.set_yticklabels(
            [MODEL_LABELS[m] for m in MODELS] if phase == "prefill" else [],
            fontsize=18,
        )
        ax.tick_params(axis="both", which="major", length=0, pad=4)
        ax.set_ylim(y[-1] - 1.5, y[0] + 1.5)
    handles, labels = axes[0].get_legend_handles_labels()

    def flip(items, ncol):
        return list(itertools.chain(*(items[index::ncol] for index in range(ncol))))

    fig.legend(
        flip(handles[:-1], 5) + handles[-1:],
        flip(labels[:-1], 5) + labels[-1:],
        loc="upper center",
        ncol=6,
        frameon=True,
        edgecolor="black",
        borderaxespad=0.0,
        bbox_to_anchor=(0.5, 0.99),
        columnspacing=0.6,
        handletextpad=0.3,
        fontsize=19,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.subplots_adjust(wspace=0.12)
    mark_incomplete(fig, allow_partial)
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="NeuSim result root or normalized energy CSV/JSON",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally results/figures/figure_02_energy_breakdown.pdf)",
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
