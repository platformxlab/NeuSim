#!/usr/bin/env python3
"""Plot MICRO'26 Figure 5 from a freshly reproduced FleetSim request trace.

The artifact's Figure 5 uses the versioned static FleetSim allocation. This
plotter intentionally has no result-directory discovery and no cache: callers
must provide the newly generated ``request_trace.csv`` and its SLO definition.
The plotted statistics use a one-minute trailing window for both panels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

ROLLING_WINDOW_MINUTES = 1.0
SLO_MULTIPLIER = "5x"
REQUIRED_TRACE_COLUMNS = (
    "input_seqlen",
    "output_seqlen",
    "prefill_end_timestamp",
    "decode_end_timestamp",
    "TTFT_ns",
    "TPOT_ns",
)
PAPER_RCPARAMS: dict[str, Any] = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 28,
    "axes.labelsize": 28,
    "legend.fontsize": 28,
    "xtick.labelsize": 28,
    "ytick.labelsize": 28,
}


@dataclass(frozen=True)
class SLOTarget:
    """One sequence-length bucket from the SLO configuration."""

    input_seqlen: int
    ttft_seconds: float
    decode_seqlen: int
    tpot_milliseconds: float


@dataclass(frozen=True)
class RollingSummary:
    """Trailing-window SLO-slack statistics in chronological request order."""

    time_hours: np.ndarray
    mean: np.ndarray
    p1: np.ndarray
    p25: np.ndarray
    p75: np.ndarray
    p99: np.ndarray


def load_slo_targets(
    slo_config: Path, multiplier: str = SLO_MULTIPLIER
) -> list[SLOTarget]:
    """Load sequence-length SLO buckets from a ``determine_slo.py`` JSON."""

    with slo_config.open(encoding="utf-8") as stream:
        document = json.load(stream)
    results = document.get("results") if isinstance(document, dict) else None
    if not isinstance(results, list) or not results:
        raise ValueError(f"{slo_config} must contain a non-empty 'results' list")

    targets: list[SLOTarget] = []
    for row in sorted(results, key=lambda item: float(item["percentile"])):
        try:
            target = SLOTarget(
                input_seqlen=int(row["input_seqlen"]),
                ttft_seconds=float(row["prefill"]["slo_TTFT_sec"][multiplier]),
                decode_seqlen=int(row["decode"]["representative_seqlen"]),
                tpot_milliseconds=float(row["decode"]["slo_TPOT_ms"][multiplier]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{slo_config} has an invalid SLO bucket for multiplier {multiplier!r}"
            ) from exc
        if (
            target.input_seqlen < 0
            or target.decode_seqlen < 0
            or target.ttft_seconds <= 0
            or target.tpot_milliseconds <= 0
        ):
            raise ValueError(f"{slo_config} contains a non-positive SLO target")
        targets.append(target)

    for previous, current in zip(targets, targets[1:], strict=False):
        if (
            current.input_seqlen < previous.input_seqlen
            or current.decode_seqlen < previous.decode_seqlen
        ):
            raise ValueError(
                f"{slo_config} sequence-length buckets are not monotonic by percentile"
            )
    return targets


def assign_ttft_target(input_seqlen: int, targets: list[SLOTarget]) -> float:
    """Assign the first TTFT bucket whose input-length ceiling is sufficient."""

    if not targets:
        raise ValueError("at least one SLO target is required")
    for target in targets:
        if input_seqlen <= target.input_seqlen:
            return target.ttft_seconds
    return targets[-1].ttft_seconds


def assign_tpot_target(total_seqlen: int, targets: list[SLOTarget]) -> float:
    """Assign the first TPOT bucket whose total-length ceiling is sufficient."""

    if not targets:
        raise ValueError("at least one SLO target is required")
    for target in targets:
        if total_seqlen <= target.decode_seqlen:
            return target.tpot_milliseconds
    return targets[-1].tpot_milliseconds


def rolling_slack_summary(
    time_hours: np.ndarray | pd.Series,
    slack_percent: np.ndarray | pd.Series,
) -> RollingSummary:
    """Compute one-minute trailing distribution statistics."""

    times = np.asarray(time_hours, dtype=float)
    slack = np.asarray(slack_percent, dtype=float)
    if times.ndim != 1 or slack.ndim != 1 or len(times) != len(slack):
        raise ValueError(
            "time and slack must be one-dimensional arrays of equal length"
        )
    if not len(times):
        raise ValueError("cannot compute rolling statistics for an empty trace")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(slack)):
        raise ValueError("time and slack values must be finite")

    order = np.argsort(times, kind="stable")
    times = times[order]
    slack = slack[order]
    index = pd.to_timedelta(times, unit="h")
    series = pd.Series(slack, index=index)
    rolling = series.rolling(
        window=pd.Timedelta(minutes=ROLLING_WINDOW_MINUTES),
        min_periods=1,
    )
    return RollingSummary(
        time_hours=times,
        mean=rolling.mean().to_numpy(),
        p1=rolling.quantile(0.01).to_numpy(),
        p25=rolling.quantile(0.25).to_numpy(),
        p75=rolling.quantile(0.75).to_numpy(),
        p99=rolling.quantile(0.99).to_numpy(),
    )


def compute_slack_summaries(
    request_trace: Path, targets: list[SLOTarget]
) -> tuple[RollingSummary, RollingSummary]:
    """Load one fresh FleetSim trace and calculate TTFT and TPOT slack."""

    try:
        frame = pd.read_csv(request_trace, usecols=list(REQUIRED_TRACE_COLUMNS))
    except ValueError as exc:
        raise ValueError(
            f"{request_trace} is missing one or more required timing columns"
        ) from exc
    missing = [column for column in REQUIRED_TRACE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(
            f"{request_trace} is missing required columns: {', '.join(missing)}"
        )
    if frame.empty:
        raise ValueError(f"{request_trace} contains no completed requests")

    numeric = frame.loc[:, REQUIRED_TRACE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        invalid = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(
            f"{request_trace} has non-numeric or missing values in: {', '.join(invalid)}"
        )
    if (numeric.loc[:, REQUIRED_TRACE_COLUMNS] < 0).any().any():
        raise ValueError(f"{request_trace} contains negative length, time, or latency")

    ttft_targets = numeric["input_seqlen"].map(
        lambda value: assign_ttft_target(int(value), targets)
    )
    total_lengths = numeric["input_seqlen"] + numeric["output_seqlen"]
    tpot_targets = total_lengths.map(
        lambda value: assign_tpot_target(int(value), targets)
    )
    ttft_actual = numeric["TTFT_ns"] / 1e9
    tpot_actual = numeric["TPOT_ns"] / 1e6
    ttft_slack = (ttft_targets - ttft_actual) / ttft_targets * 100.0
    tpot_slack = (tpot_targets - tpot_actual) / tpot_targets * 100.0

    nanoseconds_per_hour = 3600.0 * 1e9
    ttft_time = numeric["prefill_end_timestamp"] / nanoseconds_per_hour
    tpot_time = numeric["decode_end_timestamp"] / nanoseconds_per_hour
    return (
        rolling_slack_summary(ttft_time, ttft_slack),
        rolling_slack_summary(tpot_time, tpot_slack),
    )


def _plot_summary(axis: plt.Axes, summary: RollingSummary) -> None:
    """Draw one panel using the exact percentile ordering and line styles."""

    time = summary.time_hours
    axis.plot(time, summary.mean, linewidth=2.5, label="Avg.", rasterized=True)
    axis.plot(
        time,
        summary.p25,
        linewidth=2.5,
        linestyle="-.",
        label="P25",
        rasterized=True,
    )
    axis.plot(
        time,
        summary.p75,
        linewidth=2.5,
        linestyle="--",
        label="P75",
        rasterized=True,
    )
    axis.plot(
        time,
        summary.p1,
        linewidth=2.5,
        linestyle=":",
        label="P1",
        rasterized=True,
    )
    axis.plot(
        time,
        summary.p99,
        linewidth=2.5,
        linestyle=":",
        label="P99",
        rasterized=True,
    )
    axis.fill_between(
        time,
        summary.p1,
        summary.p99,
        alpha=0.15,
        rasterized=True,
    )


def _add_break_marks(figure: Figure, top_axis: plt.Axes, bottom_axis: plt.Axes) -> None:
    """Draw matching diagonal marks on both sides of a broken y-axis pair."""

    figure.canvas.draw()
    top_bounds = top_axis.get_position()
    bottom_bounds = bottom_axis.get_position()
    break_width = 0.015
    break_height = 0.005
    break_gap = 0.003
    break_style = {"color": "k", "clip_on": False, "linewidth": 1.5}
    for axis, y_anchor in (
        (top_axis, top_bounds.y0),
        (bottom_axis, bottom_bounds.y1),
    ):
        transform = blended_transform_factory(axis.transAxes, figure.transFigure)
        for x_anchor in (0.0, 1.0):
            for direction in (-1, 1):
                center = y_anchor + direction * break_gap
                figure.add_artist(
                    Line2D(
                        [x_anchor - break_width, x_anchor + break_width],
                        [center - break_height, center + break_height],
                        transform=transform,
                        **break_style,
                    )
                )


def _configure_broken_axes(
    figure: Figure,
    top_axis: plt.Axes,
    bottom_axis: plt.Axes,
    *,
    top_limits: tuple[float, float],
    top_ticks: list[float],
    bottom_limits: tuple[float, float],
    bottom_ticks: list[float],
    label: str,
) -> None:
    """Configure one slack panel as a zoomed top and compressed lower segment."""

    top_axis.set_ylim(*top_limits)
    top_axis.set_yticks(top_ticks)
    bottom_axis.set_ylim(*bottom_limits)
    bottom_axis.set_yticks(bottom_ticks)
    bottom_axis.axhline(y=0, color="red", linestyle="--", linewidth=2.0)
    bottom_axis.text(
        0.02,
        0.5,
        "SLO",
        transform=bottom_axis.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=28,
        color="red",
    )

    top_axis.spines["bottom"].set_visible(False)
    bottom_axis.spines["top"].set_visible(False)
    top_axis.tick_params(bottom=False, labelbottom=False)
    bottom_axis.tick_params(labelbottom=False)
    _add_break_marks(figure, top_axis, bottom_axis)

    top_axis.grid(True, alpha=0.3)
    bottom_axis.grid(True, alpha=0.3)
    midpoint = (top_axis.get_position().y1 + bottom_axis.get_position().y0) / 2
    figure.text(
        0.04,
        midpoint,
        label,
        va="center",
        ha="center",
        rotation="vertical",
        fontsize=28,
    )


def create_figure(
    ttft: RollingSummary, tpot: RollingSummary
) -> tuple[Figure, tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes]]:
    """Create Figure 5 while preserving the original paper plot geometry."""

    with plt.rc_context(PAPER_RCPARAMS):
        figure = plt.figure(figsize=(16, 9))
        grid = figure.add_gridspec(
            5,
            1,
            height_ratios=[2, 1, 0.25, 2, 1],
            hspace=0.15,
        )
        ttft_top = figure.add_subplot(grid[0])
        ttft_bottom = figure.add_subplot(grid[1], sharex=ttft_top)
        tpot_top = figure.add_subplot(grid[3], sharex=ttft_top)
        tpot_bottom = figure.add_subplot(grid[4], sharex=ttft_top)

        for axis in (ttft_top, ttft_bottom):
            _plot_summary(axis, ttft)
        ttft_top.set_xlim(0, 24)
        _configure_broken_axes(
            figure,
            ttft_top,
            ttft_bottom,
            top_limits=(65, 100),
            top_ticks=[70, 80, 90, 100],
            bottom_limits=(-100, 65),
            bottom_ticks=[-100, -50, 0],
            label="TTFT\nSLO Slack (%)",
        )
        ttft_top.legend(
            loc="lower center",
            ncol=10,
            columnspacing=0.8,
            handletextpad=0.4,
            bbox_to_anchor=(0.5, 1.02),
            edgecolor="black",
        )

        for axis in (tpot_top, tpot_bottom):
            _plot_summary(axis, tpot)
        tpot_top.set_xlim(0, 24)
        _configure_broken_axes(
            figure,
            tpot_top,
            tpot_bottom,
            top_limits=(55, 100),
            top_ticks=[60, 80, 100],
            bottom_limits=(-10, 10),
            bottom_ticks=[-10, 0],
            label="TPOT\nSLO Slack (%)",
        )
        tpot_bottom.tick_params(labelbottom=True)
        tpot_bottom.set_xlabel("Time (hours)")
        return figure, (ttft_top, ttft_bottom, tpot_top, tpot_bottom)


def plot(request_trace: Path, slo_config: Path, output: Path) -> None:
    """Reproduce Figure 5 directly from the specified fresh simulation output."""

    targets = load_slo_targets(slo_config)
    ttft, tpot = compute_slack_summaries(request_trace, targets)
    figure, _ = create_figure(ttft, tpot)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request-trace",
        required=True,
        type=Path,
        help="Fresh FleetSim request_trace.csv",
    )
    parser.add_argument(
        "--slo-config",
        required=True,
        type=Path,
        help="Llama3-70B Azure-Code SLO JSON used by the reproduced run",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output PDF (normally figures/figure_05_slo_slack.pdf)",
    )
    args = parser.parse_args()
    plot(args.request_trace, args.slo_config, args.output)
    print(f"Figure 5 saved to {args.output}")


if __name__ == "__main__":
    main()
