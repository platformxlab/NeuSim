#!/usr/bin/env python3
"""Plot Figure 16: spatial and temporal DVFS granularity.

Energy saving is recomputed from raw request energy and the matching NoDVFS
record. The millisecond-policy rows must come from the epoch-based experiment,
not from a plot-time transformation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

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
    require_fields,
    save_figure,
    threshold_pct,
)
from matplotlib.ticker import MaxNLocator

POLICIES = ["DVFS-C", "DVFS-C-ms", "eNPU-All", "eNPU-ms", "Ideal"]
THRESHOLDS = [0.0, 2.0, 5.0, 10.0, 20.0, 25.0, 30.0]
MATCH_FIELDS = ("input_tokens", "output_tokens", "config")
SOURCE_MODEL_LABELS = {
    **MODEL_LABELS,
    "deepseekv2-236b": "DeepSeek-V2-236B",
    "deepseekv3-671b": "DeepSeek-V3-671B",
}

# Recovered directly from the vector paths embedded for Figure 16 in the
# paper PDF. Keep this local: the millisecond-policy colors and dashes are
# specific to this five-curve comparison.
POLICY_STYLES = {
    "DVFS-C": dict(
        color="#2a9d4e",
        marker="^",
        linestyle="-.",
        markerfacecolor="none",
        markeredgecolor="#2a9d4e",
        markeredgewidth=1.0,
        markersize=8,
        linewidth=2.0,
    ),
    "DVFS-C-ms": dict(
        color="#7a3b9b",
        marker="v",
        linestyle="--",
        markerfacecolor="none",
        markeredgecolor="#7a3b9b",
        markeredgewidth=1.0,
        markersize=8,
        linewidth=2.0,
    ),
    "eNPU-All": dict(
        color="#e8750a",
        marker="D",
        linestyle=(0, (5, 2)),
        markerfacecolor="none",
        markeredgecolor="#e8750a",
        markeredgewidth=1.0,
        markersize=6,
        linewidth=2.0,
    ),
    "eNPU-ms": dict(
        color="#17a2b8",
        marker="s",
        linestyle=(0, (3, 1, 1, 1)),
        markerfacecolor="white",
        markeredgecolor="#17a2b8",
        markeredgewidth=1.0,
        markersize=7,
        linewidth=2.0,
    ),
    "Ideal": dict(
        color="#8b0000",
        marker="",
        linestyle=(0, (1, 1)),
        linewidth=2.0,
    ),
}


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.is_file():
        return load_table(path)
    normalized = path / "energy_records.json"
    if normalized.is_file():
        return load_table(normalized)
    try:
        return load_energy_results(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Figure 16 found neither {normalized} nor NeuSim energy JSONs under {path}"
        ) from exc


def _matching_baselines(
    row: Mapping[str, Any], baselines: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    matches = [
        base
        for base in baselines
        if base.get("model") == row.get("model")
        and base.get("phase") == row.get("phase")
    ]
    for field in MATCH_FIELDS:
        if field in row:
            exact = [
                base
                for base in matches
                if field not in base or base.get(field) == row.get(field)
            ]
            if exact:
                matches = exact
            else:
                return []
    return matches


def _energy_savings(rows: list[dict[str, Any]], source: Path) -> list[dict[str, Any]]:
    require_fields(rows, ("model", "phase", "policy", "total_energy_J"), source)
    rows = [row for row in rows if row.get("pg_strategy", "NoPG") == "NoPG"]
    baselines = [row for row in rows if row.get("policy") == "NoDVFS"]
    if not baselines:
        raise ValueError("Figure 16 requires NoDVFS request-energy records")

    output: list[dict[str, Any]] = []
    missing: set[tuple[Any, Any]] = set()
    for row in rows:
        if row.get("policy") not in POLICIES:
            continue
        candidates = _matching_baselines(row, baselines)
        energies = [float(base["total_energy_J"]) for base in candidates]
        energies = [value for value in energies if value > 0]
        if not energies:
            missing.add((row.get("model"), row.get("phase")))
            continue
        baseline = mean(energies)
        output.append(
            {
                **row,
                "threshold_pct": threshold_pct(row),
                "energy_saving_pct": (1.0 - float(row["total_energy_J"]) / baseline)
                * 100.0,
            }
        )
    if missing:
        cases = ", ".join(f"{model}/{phase}" for model, phase in sorted(missing))
        raise ValueError(
            f"Figure 16 has policy records without matching NoDVFS energy: {cases}"
        )
    if not output:
        raise ValueError(
            "Figure 16 input has no DVFS-C/eNPU/Ideal records with request energy"
        )
    return output


def plot(input_path: Path, output: Path, allow_partial: bool = False) -> None:
    rows = _energy_savings(_load_records(input_path), input_path)
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.get("model"),
                row.get("phase"),
                row.get("policy"),
                row["threshold_pct"],
            )
        ].append(float(row["energy_saving_pct"]))
    require_combinations(
        grouped,
        (
            (model, phase, policy, threshold)
            for model in MODELS
            for phase in PHASES
            for policy in POLICIES
            for threshold in THRESHOLDS
        ),
        "Figure 16 policy matrix",
        allow_partial,
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
    plotted = 0
    row_values: list[list[float]] = [[] for _ in PHASES]
    for row_index, phase in enumerate(PHASES):
        for column_index, model in enumerate(MODELS):
            ax = axes[row_index, column_index]
            for policy in POLICIES:
                points = sorted(
                    (
                        (float(key[3]), mean(values))
                        for key, values in grouped.items()
                        if key[:3] == (model, phase, policy)
                    ),
                    key=lambda point: point[0],
                )
                if not points:
                    continue
                xs, ys = zip(*points, strict=False)
                row_values[row_index].extend(ys)
                ax.plot(xs, ys, label=policy, **POLICY_STYLES[policy])
                plotted += 1
            if not ax.lines:
                ax.text(
                    0.5,
                    0.5,
                    "missing input",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            ax.set_xlim(-0.03 * max(THRESHOLDS), 1.08 * max(THRESHOLDS))
            ax.set_xticks([0, 10, 20, 30])
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.grid(True, linewidth=0.5, alpha=0.5, linestyle=":")
            if row_index == 0:
                ax.set_title(SOURCE_MODEL_LABELS[model])
            if column_index == 0:
                ax.set_ylabel(
                    f"{phase.capitalize()}\nEnergy\nSaving (%)",
                    linespacing=1.2,
                    labelpad=4,
                )
            else:
                ax.tick_params(axis="y", labelleft=False)
    if not plotted:
        plt.close(fig)
        raise ValueError("Figure 16 could not construct any policy curves")

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

    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes.flat:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=False):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    left, right = 0.10, 0.995
    graph_center = (left + right) / 2
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        bbox_to_anchor=(graph_center, 1.03),
        columnspacing=1.0,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(0.8)
    fig.text(
        graph_center,
        0.10 / fig_h,
        "Performance Degradation Threshold (%)",
        ha="center",
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
        help="energy_records.json, normalized table, or temporal-granularity result directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally results/figures/figure_16_dvfs_granularity.pdf)",
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
