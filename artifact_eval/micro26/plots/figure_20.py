#!/usr/bin/env python3
"""Plot Figure 20: MoE expert-load imbalance and provisioning gap.

The recovered paper script uses a hybrid summary: the left panel reports
absolute savings at 20%, while the right panel reports the mean loss across
0/1/2/5/10/20% with its min/max band. That source-compatible interpretation
is the default. No anchoring or rescaling is provided.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from common import (
    load_table,
    mark_incomplete,
    paper_style,
    require_combinations,
    save_figure,
)
from matplotlib.lines import Line2D

POLICIES = ["DVFS-C", "eNPU-All"]
FACTORS = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
PAPER_MANIFEST = (
    Path(__file__).resolve().parents[1] / "config" / "paper_experiments.json"
)
with PAPER_MANIFEST.open(encoding="utf-8") as _handle:
    _paper = json.load(_handle)
MEAN_THRESHOLDS = [
    float(value)
    for value in _paper["performance_degradation_thresholds_percent"]["figure_20_sweep"]
]
STYLES = {
    "DVFS-C": dict(color="#2a9d4e", marker="^"),
    "eNPU-All": dict(color="#e8750a", marker="D"),
}


def plot(
    input_path: Path,
    output: Path,
    worst_case_experts: int,
    summary: str,
    threshold_pct: float,
    allow_partial: bool = False,
) -> None:
    if input_path.is_dir():
        input_path = input_path / "expert_imbalance_records.json"
    rows = [
        r
        for r in load_table(input_path)
        if str(r.get("phase", "prefill")) == "prefill"
        and int(r.get("nwc", worst_case_experts)) == worst_case_experts
    ]
    rows = [r for r in rows if r.get("policy") in POLICIES]
    if not rows:
        raise ValueError("Figure 20 input has no matching prefill imbalance records")
    for row in rows:
        row["pd_pct"] = (
            float(row.get("pd", 0)) * 100.0
            if abs(float(row.get("pd", 0))) <= 1
            else float(row["pd"])
        )
    expected_thresholds = (
        MEAN_THRESHOLDS if summary in {"hybrid", "mean"} else [threshold_pct]
    )
    require_combinations(
        (
            (row.get("policy"), float(row["real_f"]), float(row["pd_pct"]))
            for row in rows
            if "real_f" in row
        ),
        (
            (policy, factor, threshold)
            for policy in POLICIES
            for factor in FACTORS
            for threshold in expected_thresholds
        ),
        "Figure 20 policy/factor/threshold matrix",
        allow_partial,
    )
    missing_metrics = [
        (row.get("policy"), row.get("real_f"), row.get("pd_pct"))
        for row in rows
        if any(field not in row for field in ("saving_oracle_pct", "saving_wc_pct"))
    ]
    if missing_metrics:
        raise ValueError(
            f"Figure 20 records lack saving metrics: {missing_metrics[:12]}"
        )
    factors = sorted({float(r["real_f"]) for r in rows}) if allow_partial else FACTORS
    at_threshold = [row for row in rows if abs(row["pd_pct"] - threshold_pct) < 1e-8]
    if not at_threshold:
        raise ValueError(f"no records at {threshold_pct:g}%")
    left_rows = rows if summary == "mean" else at_threshold
    right_rows = at_threshold if summary == "at-threshold" else rows
    left_grouped: dict[tuple, list[dict]] = defaultdict(list)
    right_grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in left_rows:
        left_grouped[(row["policy"], float(row["real_f"]))].append(row)
    for row in right_rows:
        right_grouped[(row["policy"], float(row["real_f"]))].append(row)
    paper_style(15)
    plt.rcParams.update(
        {
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.6,
            "lines.markersize": 6,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))
    left_measured_values = [
        float(row[field])
        for row in left_rows
        for field in ("saving_oracle_pct", "saving_wc_pct")
    ]
    right_measured_values = [
        float(row["saving_oracle_pct"]) - float(row["saving_wc_pct"])
        for row in right_rows
    ]
    for policy in POLICIES:
        oracle = [
            mean(float(r["saving_oracle_pct"]) for r in left_grouped[(policy, f)])
            for f in factors
        ]
        worst = [
            mean(float(r["saving_wc_pct"]) for r in left_grouped[(policy, f)])
            for f in factors
        ]
        delta = [
            mean(
                float(r["saving_oracle_pct"]) - float(r["saving_wc_pct"])
                for r in right_grouped[(policy, f)]
            )
            for f in factors
        ]
        style = STYLES[policy]
        axes[0].plot(
            factors,
            oracle,
            linestyle="-",
            linewidth=1.6,
            markersize=7,
            **style,
        )
        axes[0].plot(
            factors,
            worst,
            linestyle=(0, (4, 2)),
            markerfacecolor="white",
            linewidth=1.6,
            markersize=7,
            **style,
        )
        axes[1].plot(
            factors,
            delta,
            linestyle="-",
            linewidth=1.6,
            markersize=8,
            label=policy,
            **style,
        )
        if summary in {"hybrid", "mean"}:
            lows, highs = [], []
            for f in factors:
                vals = [
                    float(r["saving_oracle_pct"]) - float(r["saving_wc_pct"])
                    for r in right_grouped[(policy, f)]
                ]
                lows.append(min(vals))
                highs.append(max(vals))
            axes[1].fill_between(
                factors, lows, highs, color=style["color"], alpha=0.18, linewidth=0
            )
    if left_measured_values and all(
        0.0 <= value <= 32.0 for value in left_measured_values
    ):
        axes[0].set_ylim(0.0, 32.0)
    elif left_measured_values and min(left_measured_values) < 0.0:
        raw_span = max(left_measured_values) - min(left_measured_values)
        axes[0].set_ylim(bottom=min(left_measured_values) - 0.05 * max(raw_span, 1.0))
    if right_measured_values and all(value >= 0.0 for value in right_measured_values):
        axes[1].set_ylim(bottom=0.0)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(factors)
        ax.set_xticklabels([f"{v:g}" for v in factors])
        ax.set_xlabel("Expert Capacity Factor")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    left_label = (
        "mean over thresholds"
        if summary == "mean"
        else f"at {threshold_pct:g}% threshold"
    )
    right_label = (
        "mean over thresholds"
        if summary in {"hybrid", "mean"}
        else f"at {threshold_pct:g}% threshold"
    )
    axes[0].set_title(
        f"Energy Saving at {threshold_pct:g}%\nPerf. Degrad. Threshold"
        if summary == "hybrid"
        else f"Energy Saving ({left_label})"
    )
    axes[0].set_ylabel("Total Energy Saving (%)")
    axes[1].set_title(
        "Loss from\nWorst-Case Provisioning"
        if summary == "hybrid"
        else f"Loss from Worst-Case Provisioning ({right_label})"
    )
    axes[1].set_ylabel(r"$\Delta$ Energy Saving (%)")
    provisioning_handles = [
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle="-",
            markerfacecolor="gray",
            label="Ideal Provisioning",
        ),
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            linestyle=(0, (4, 2)),
            markerfacecolor="white",
            label="Worst-Case Provisioning",
        ),
    ]
    axes[0].legend(
        handles=provisioning_handles,
        loc="lower left",
        fontsize=14,
        framealpha=0.92,
        edgecolor="gray",
        fancybox=False,
    )
    axes[1].legend(
        loc="upper right",
        framealpha=0.92,
        edgecolor="gray",
        fancybox=False,
    )
    fig.tight_layout()
    mark_incomplete(fig, allow_partial)
    save_figure(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Worst-case-provisioning JSON/CSV with result records",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally results/figures/figure_20_expert_imbalance.pdf)",
    )
    parser.add_argument("--worst-case-experts", type=int, default=8)
    parser.add_argument(
        "--summary",
        choices=("hybrid", "mean", "at-threshold"),
        default="hybrid",
        help=(
            "hybrid matches the recovered source: absolute savings at --threshold "
            "on the left and across-threshold loss statistics on the right"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Percent threshold used with --summary at-threshold",
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
        args.worst_case_experts,
        args.summary,
        args.threshold,
        args.allow_partial,
    )


if __name__ == "__main__":
    main()
