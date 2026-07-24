#!/usr/bin/env python3
"""Plot Figure 18: sequence-length sensitivity of eNPU-All.

Cells are raw energy savings relative to a matching NoDVFS request. No
sequence-length-dependent static-energy correction is applied.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from common import (
    MODEL_LABELS,
    load_energy_results,
    load_table,
    mark_incomplete,
    paper_style,
    require_combinations,
    require_fields,
    save_figure,
    threshold_pct,
)

MODELS = ["llama3-70b", "deepseekv3-671b"]
PHASES = ["prefill", "decode"]
INPUT_LENGTHS = [
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]
THRESHOLDS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
MATCH_FIELDS = ("model", "phase", "input_tokens", "output_tokens", "config")
PAPER_COLOR_MIN = 0.0
PAPER_COLOR_MAX = 25.0


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
            f"Figure 18 found neither {normalized} nor NeuSim energy JSONs under {path}"
        ) from exc


def _matching_baselines(
    row: Mapping[str, Any], baselines: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    matches = list(baselines)
    for field in MATCH_FIELDS:
        exact = [base for base in matches if base.get(field) == row.get(field)]
        if exact:
            matches = exact
        elif field in row:
            return []
    return matches


def _saving_rows(
    rows: list[dict[str, Any]], source: Path, output_tokens: int
) -> list[dict[str, Any]]:
    require_fields(
        rows,
        ("model", "phase", "input_tokens", "policy", "total_energy_J"),
        source,
    )
    rows = [
        row
        for row in rows
        if int(row.get("output_tokens", output_tokens)) == output_tokens
        and row.get("model") in MODELS
        and row.get("phase") in PHASES
    ]
    baselines = [row for row in rows if row.get("policy") == "NoDVFS"]
    policies = [row for row in rows if row.get("policy") == "eNPU-All"]
    if not baselines:
        raise ValueError(
            f"Figure 18 requires NoDVFS records at output_tokens={output_tokens}"
        )
    if not policies:
        raise ValueError(
            f"Figure 18 requires eNPU-All records at output_tokens={output_tokens}"
        )

    output: list[dict[str, Any]] = []
    missing: set[tuple[Any, ...]] = set()
    for row in policies:
        candidates = _matching_baselines(row, baselines)
        energies = [float(base["total_energy_J"]) for base in candidates]
        energies = [value for value in energies if value > 0]
        if not energies:
            missing.add((row.get("model"), row.get("phase"), row.get("input_tokens")))
            continue
        output.append(
            {
                **row,
                "threshold_pct": threshold_pct(row),
                "energy_saving_pct": (
                    1.0 - float(row["total_energy_J"]) / mean(energies)
                )
                * 100.0,
            }
        )
    if missing:
        cases = ", ".join(f"{m}/{p}/{int(n)}" for m, p, n in sorted(missing))
        raise ValueError(
            f"Figure 18 has eNPU-All rows without matching NoDVFS energy: {cases}"
        )
    return output


def _length_label(value: int) -> str:
    if value >= 1048576 and value % 1048576 == 0:
        return f"{value // 1048576}M"
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


def plot(
    input_path: Path, output: Path, output_tokens: int, allow_partial: bool = False
) -> None:
    rows = _saving_rows(_load_records(input_path), input_path, output_tokens)
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        length = int(float(row["input_tokens"]))
        threshold = float(row["threshold_pct"])
        if length in INPUT_LENGTHS and threshold in THRESHOLDS:
            grouped[(row.get("model"), row.get("phase"), length, threshold)].append(
                float(row["energy_saving_pct"])
            )
    if not grouped:
        raise ValueError(
            "Figure 18 has no records on the fixed 256-to-1M length and 0%-to-30% threshold grid"
        )
    require_combinations(
        grouped,
        (
            (model, phase, length, threshold)
            for model in MODELS
            for phase in PHASES
            for length in INPUT_LENGTHS
            for threshold in THRESHOLDS
        ),
        "Figure 18 sequence-length matrix",
        allow_partial,
    )

    matrices: dict[tuple[str, str], np.ndarray] = {}
    for model in MODELS:
        for phase in PHASES:
            matrix = np.full((len(INPUT_LENGTHS), len(THRESHOLDS)), np.nan)
            for yi, length in enumerate(INPUT_LENGTHS):
                for xi, threshold in enumerate(THRESHOLDS):
                    values = grouped.get((model, phase, length, threshold), [])
                    if values:
                        matrix[yi, xi] = mean(values)
            matrices[(model, phase)] = matrix

    finite = np.concatenate(
        [matrix[np.isfinite(matrix)] for matrix in matrices.values()]
    )
    if finite.size == 0:
        raise ValueError("Figure 18 could not compute a finite energy-saving cell")
    # The paper fixes one shared normalization across all four heatmaps.
    lower = PAPER_COLOR_MIN
    upper = PAPER_COLOR_MAX

    # Match the paper's compact four-panel layout: its DejaVu Serif text is
    # roughly twice the size, relative to each heatmap, of the generic AE
    # plotting defaults previously used here.
    paper_style(16)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 4.4), sharex=True, sharey=True)
    image = None
    for row_index, model in enumerate(MODELS):
        for column_index, phase in enumerate(PHASES):
            ax = axes[row_index * len(PHASES) + column_index]
            matrix = matrices[(model, phase)]
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                vmin=lower,
                vmax=upper,
            )
            for yi in range(len(INPUT_LENGTHS)):
                for xi in range(len(THRESHOLDS)):
                    value = matrix[yi, xi]
                    if np.isfinite(value):
                        normalized = (value - lower) / (upper - lower)
                        color = "white" if normalized <= 0.62 else "black"
                        ax.text(
                            xi,
                            yi,
                            f"{value:.0f}",
                            ha="center",
                            va="center",
                            fontsize=12,
                            color=color,
                        )
                    else:
                        ax.text(
                            xi,
                            yi,
                            "–",
                            ha="center",
                            va="center",
                            fontsize=12,
                            color="#888888",
                        )
            ax.set_title(phase.capitalize())
            ax.set_xticks(
                range(len(THRESHOLDS)), [f"{value:g}" for value in THRESHOLDS]
            )
            ax.set_yticks(
                range(len(INPUT_LENGTHS)),
                [_length_label(value) for value in INPUT_LENGTHS],
            )
            if row_index == 0 and column_index == 0:
                ax.set_ylabel("Input Sequence Length", fontsize=22)
    assert image is not None
    fig.text(0.27, 0.90, MODEL_LABELS[MODELS[0]], ha="center", fontsize=22)
    fig.text(0.70, 0.90, MODEL_LABELS[MODELS[1]], ha="center", fontsize=22)
    fig.supxlabel(
        "Performance Degradation Threshold (%)", y=0.0, fontsize=22
    )
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.15, top=0.82, wspace=0.08)
    colorbar_axis = fig.add_axes((0.925, 0.15, 0.014, 0.67))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_ticks([0, 5, 10, 15, 20, 25])
    colorbar.set_label("Energy Saving (%)", fontsize=20)
    mark_incomplete(fig, allow_partial)
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="energy_records.json, normalized table, or fixed-sequence result directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally results/figures/figure_18_sequence_length.pdf)",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=512,
        help="Fixed output sequence length represented by the sweep",
    )
    parser.add_argument(
        "--allow-partial",
        "--allow-incomplete",
        action="store_true",
        help="Allow an explicitly reduced quick-run matrix",
    )
    args = parser.parse_args()
    plot(args.input, args.output, args.output_tokens, args.allow_partial)


if __name__ == "__main__":
    main()
