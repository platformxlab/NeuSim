#!/usr/bin/env python3
"""Plot MICRO'26 Figure 22 from three explicit FleetSim request traces.

This is a path-independent adaptation of the paper's original
``fleetsim_dvfs_timeseries.py``. It deliberately preserves the original
five-minute rolling window, one-minute resampling, pointwise baseline
normalization, panel geometry, colors, line styles, and axis labels.

The simulator and service-level DVFS scheduler consume the exact SLO values
from the supplied JSON. For plotted SLO satisfaction only, the original
Figure 22 script rounded each target expressed in seconds to three decimal
places. This adaptation preserves that plotting-only rounding.

The current FleetSim trace records aggregate per-request ``prefill_energy_J``
and ``decode_energy_J`` values. The paper plot's y-axis says ``Joule/Token``,
but its original implementation first converted decode energy-per-token back
to aggregate request energy and then averaged per request. This plotter keeps
that source behavior for reproducibility; it does not divide energy by token
count.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from artifact_eval.micro26.plots.figure_05 import (
    SLO_MULTIPLIER,
    SLOTarget,
    load_slo_targets,
)

RUN_ORDER = ("baseline", "DVFSC", "CustomAll")
LABELS = {
    "baseline": "NoDVFS",
    "DVFSC": "DVFS-C",
    "CustomAll": "eNPU-All",
}
COLORS = {
    "baseline": "#444444",
    "DVFSC": "#ff7f0e",
    "CustomAll": "#2ca02c",
}
LINESTYLES: dict[str, Any] = {
    "baseline": (0, (4, 2)),
    "DVFSC": (0, (1, 1.2)),
    "CustomAll": "-",
}
ALPHAS = {"baseline": 1.0, "DVFSC": 1.0, "CustomAll": 0.75}
ROLLING_WINDOW = "5min"
RESAMPLE_FREQUENCY = "1min"
PAPER_SLO_DECIMAL_PLACES_SECONDS = 3
AXIS_LABEL_FONTSIZE = 12
TITLE_FONTSIZE = 13
PAPER_RCPARAMS: dict[str, Any] = {
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
}
REQUIRED_TRACE_COLUMNS = (
    "enqueue_timestamp",
    "input_seqlen",
    "output_seqlen",
    "TTFT_ns",
    "TPOT_ns",
    "prefill_energy_J",
    "decode_energy_J",
)
ROLLING_COLUMNS = (
    "prefill_slo",
    "decode_slo",
    "prefill_energy",
    "decode_energy",
)
DUMP_COLUMNS = (
    *ROLLING_COLUMNS,
    "prefill_energy_norm",
    "decode_energy_norm",
)


@dataclass(frozen=True)
class Figure22Outputs:
    """Files created by one Figure 22 plotting invocation."""

    pdf: Path
    png: Path
    csv: Path


def round_slo_targets_for_paper_plot(
    targets: list[SLOTarget],
) -> list[SLOTarget]:
    """Apply the original Figure 22 plotting-only SLO precision.

    TTFT is already expressed in seconds. TPOT is stored in milliseconds by
    ``SLOTarget``, so convert it to seconds, round, and convert it back. The
    scheduler does not call this function and continues to use the exact JSON
    values.
    """

    return [
        SLOTarget(
            input_seqlen=target.input_seqlen,
            ttft_seconds=round(target.ttft_seconds, PAPER_SLO_DECIMAL_PLACES_SECONDS),
            decode_seqlen=target.decode_seqlen,
            tpot_milliseconds=round(
                target.tpot_milliseconds / 1000.0,
                PAPER_SLO_DECIMAL_PLACES_SECONDS,
            )
            * 1000.0,
        )
        for target in targets
    ]


def slo_threshold_records(
    targets: list[SLOTarget],
) -> list[dict[str, int | float]]:
    """Return unit-explicit SLO records suitable for review provenance."""

    return [
        {
            "input_seqlen_ceiling": target.input_seqlen,
            "decode_total_seqlen_ceiling": target.decode_seqlen,
            "ttft_seconds": target.ttft_seconds,
            "tpot_seconds": target.tpot_milliseconds / 1000.0,
        }
        for target in targets
    ]


def _bucket_values(
    sequence_lengths: np.ndarray,
    targets: list[SLOTarget],
    *,
    phase: str,
) -> np.ndarray:
    """Map each length to the first sufficient P33/P66/P100 SLO bucket."""

    if not targets:
        raise ValueError("at least one SLO target is required")
    if phase == "prefill":
        ceilings = np.asarray([target.input_seqlen for target in targets])
        values = np.asarray([target.ttft_seconds for target in targets])
    elif phase == "decode":
        ceilings = np.asarray([target.decode_seqlen for target in targets])
        values = np.asarray([target.tpot_milliseconds / 1000.0 for target in targets])
    else:
        raise ValueError(f"unknown phase: {phase}")

    bucket_indices = np.searchsorted(ceilings, sequence_lengths, side="left")
    bucket_indices = np.minimum(bucket_indices, len(values) - 1)
    return values[bucket_indices]


def load_run(path: Path, targets: list[SLOTarget]) -> pd.DataFrame:
    """Load one fresh request trace and extract paper Figure 22 features.

    ``targets`` are the exact scheduler inputs. SLO pass/fail aggregation uses
    the original paper plot's three-decimal precision in seconds.
    """

    try:
        frame = pd.read_csv(path, usecols=list(REQUIRED_TRACE_COLUMNS))
    except ValueError as exc:
        raise ValueError(
            f"{path} is missing one or more required Figure 22 columns: "
            f"{', '.join(REQUIRED_TRACE_COLUMNS)}"
        ) from exc
    if frame.empty:
        raise ValueError(f"{path} contains no completed requests")

    numeric = frame.loc[:, REQUIRED_TRACE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        invalid = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError(
            f"{path} has non-numeric or missing values in: {', '.join(invalid)}"
        )
    if (numeric < 0).any().any():
        raise ValueError(f"{path} contains negative lengths, times, or energy")

    paper_plot_targets = round_slo_targets_for_paper_plot(targets)
    total_seqlen = (
        numeric["input_seqlen"].to_numpy() + numeric["output_seqlen"].to_numpy()
    )
    ttft_slo_seconds = _bucket_values(
        numeric["input_seqlen"].to_numpy(),
        paper_plot_targets,
        phase="prefill",
    )
    tpot_slo_seconds = _bucket_values(total_seqlen, paper_plot_targets, phase="decode")

    features = pd.DataFrame(
        {
            "prefill_pass": numeric["TTFT_ns"].to_numpy() / 1e9 <= ttft_slo_seconds,
            "decode_pass": numeric["TPOT_ns"].to_numpy() / 1e9 <= tpot_slo_seconds,
            "prefill_energy": numeric["prefill_energy_J"].to_numpy(),
            # Aggregate request energy, intentionally matching the paper script.
            "decode_energy": numeric["decode_energy_J"].to_numpy(),
            "t": pd.to_timedelta(numeric["enqueue_timestamp"].to_numpy(), unit="ns"),
        }
    )
    return features.sort_values("t").set_index("t")


def rolling_metrics(frame: pd.DataFrame, window: str = ROLLING_WINDOW) -> pd.DataFrame:
    """Compute rolling phase SLO rates and mean aggregate energy per request."""

    return pd.DataFrame(
        {
            "prefill_slo": frame["prefill_pass"].astype(float).rolling(window).mean(),
            "decode_slo": frame["decode_pass"].astype(float).rolling(window).mean(),
            "prefill_energy": frame["prefill_energy"].rolling(window).mean(),
            "decode_energy": frame["decode_energy"].rolling(window).mean(),
        }
    )


def resample_to_grid(
    series: pd.Series, frequency: str = RESAMPLE_FREQUENCY
) -> pd.Series:
    """Resample one irregular request series onto the paper's minute grid."""

    return series.resample(frequency).mean().interpolate(method="time")


def compute_grids(
    run_paths: Mapping[str, Path],
    targets: list[SLOTarget],
    *,
    window: str = ROLLING_WINDOW,
) -> dict[str, pd.DataFrame]:
    """Load, aggregate, and normalize all three explicit policy traces."""

    missing = [name for name in RUN_ORDER if name not in run_paths]
    extras = [name for name in run_paths if name not in RUN_ORDER]
    if missing or extras:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if extras:
            details.append(f"unexpected {', '.join(extras)}")
        raise ValueError("invalid Figure 22 run set: " + "; ".join(details))

    grid: dict[str, pd.DataFrame] = {}
    # Process one trace at a time so the three full-day request frames do not
    # need to coexist in memory. This does not change the original arithmetic.
    for name in RUN_ORDER:
        raw = load_run(Path(run_paths[name]), targets)
        rolled = rolling_metrics(raw, window=window)
        grid[name] = pd.DataFrame(
            {column: resample_to_grid(rolled[column]) for column in ROLLING_COLUMNS}
        )

    base_prefill = grid["baseline"]["prefill_energy"]
    base_decode = grid["baseline"]["decode_energy"]
    for name in RUN_ORDER:
        grid[name]["prefill_energy_norm"] = grid[name]["prefill_energy"] / base_prefill
        grid[name]["decode_energy_norm"] = grid[name]["decode_energy"] / base_decode
    return grid


def _hours(index: pd.TimedeltaIndex) -> np.ndarray:
    """Convert a timedelta grid to elapsed trace hours."""

    return index.total_seconds().to_numpy() / 3600.0


def create_figure(
    grid: Mapping[str, pd.DataFrame],
) -> tuple[Figure, np.ndarray]:
    """Create the paper's 2x2 SLO/energy Figure 22 layout."""

    with plt.rc_context(PAPER_RCPARAMS):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(6.8, 4.0 * 0.55),
            sharex=True,
            sharey=False,
            gridspec_kw={"hspace": 0.18, "wspace": 0.30},
        )

        def plot_slo(
            axis: plt.Axes,
            key: str,
            title: str,
            *,
            ylim: tuple[float, float],
            yticks: tuple[float, ...],
            decimals: int,
        ) -> None:
            for name in RUN_ORDER:
                series = grid[name][key].dropna()
                axis.plot(
                    _hours(series.index),
                    series.to_numpy() * 100,
                    color=COLORS[name],
                    ls=LINESTYLES[name],
                    lw=1.4,
                    alpha=ALPHAS[name],
                    label=LABELS[name],
                )
            axis.set_ylim(*ylim)
            axis.set_yticks(yticks)
            axis.grid(True, alpha=0.3)
            axis.set_title(title, fontsize=TITLE_FONTSIZE, pad=2)
            axis.yaxis.set_major_formatter(PercentFormatter(decimals=decimals))

        def plot_energy(axis: plt.Axes, key: str) -> None:
            for name in RUN_ORDER:
                series = grid[name][key].dropna()
                axis.plot(
                    _hours(series.index),
                    series.to_numpy(),
                    color=COLORS[name],
                    ls=LINESTYLES[name],
                    lw=1.4,
                    alpha=ALPHAS[name],
                    label=LABELS[name],
                )
            axis.axhline(1.0, color="black", ls=":", lw=0.8)
            axis.grid(True, alpha=0.3)
            axis.set_ylim(0.7, 1.05)
            axis.set_yticks([0.7, 0.8, 0.9, 1.0])

        plot_slo(
            axes[0, 0],
            "prefill_slo",
            "Prefill",
            ylim=(95, 100.3),
            yticks=(95, 96, 97, 98, 99, 100),
            decimals=0,
        )
        plot_slo(
            axes[0, 1],
            "decode_slo",
            "Decode",
            ylim=(99.88, 100.01),
            yticks=(99.90, 100.00),
            decimals=1,
        )
        axes[0, 1].set_yticklabels(("99.9%", "100%"))
        axes[0, 1].set_yticks((99.95,), minor=True)
        axes[0, 1].grid(True, which="minor", alpha=0.3)
        axes[0, 0].set_ylabel("SLO\nSat. Rate", fontsize=AXIS_LABEL_FONTSIZE)

        plot_energy(axes[1, 0], "prefill_energy_norm")
        plot_energy(axes[1, 1], "decode_energy_norm")
        axes[1, 0].set_ylabel("Normalized\nJoule/Token", fontsize=AXIS_LABEL_FONTSIZE)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        legend = figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=3,
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
            fancybox=False,
            fontsize=9,
            handletextpad=0.3,
            columnspacing=0.8,
            handlelength=1.0,
        )
        legend.get_frame().set_linewidth(0.8)
        axes[1, 0].set_xlabel("Time (hours)", fontsize=13)
        axes[1, 1].set_xlabel("Time (hours)", fontsize=13)

        axes[0, 0].yaxis.label.set_y(0.54)
        axes[1, 0].yaxis.label.set_y(0.46)
        for axis in axes.flat:
            axis.set_xlim(12, 24)
            axis.set_xticks(list(range(12, 25, 2)))
        figure.align_ylabels(axes[:, 0])
        return figure, axes


def write_grid_csv(grid: Mapping[str, pd.DataFrame], output: Path) -> None:
    """Write the same compact one-minute data grid used by the plotted lines."""

    merged = pd.DataFrame(
        {
            f"{name}_{column}": grid[name][column]
            for name in RUN_ORDER
            for column in DUMP_COLUMNS
        }
    )
    merged.index = merged.index.total_seconds() / 3600.0
    merged.index.name = "hours"
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output)


def plot(
    baseline_trace: Path,
    dvfsc_trace: Path,
    enpu_all_trace: Path,
    slo_config: Path,
    output_dir: Path,
    *,
    basename: str = "figure_22_fleetsim_dvfs_timeseries",
) -> Figure22Outputs:
    """Build Figure 22 PDF/PNG and its plotted one-minute CSV grid."""

    targets = load_slo_targets(slo_config, multiplier=SLO_MULTIPLIER)
    grid = compute_grids(
        {
            "baseline": baseline_trace,
            "DVFSC": dvfsc_trace,
            "CustomAll": enpu_all_trace,
        },
        targets,
    )
    figure, _ = create_figure(grid)

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{basename}.pdf"
    png = output_dir / f"{basename}.png"
    csv = output_dir / f"{basename}.csv"
    figure.savefig(png, dpi=160, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    write_grid_csv(grid, csv)
    return Figure22Outputs(pdf=pdf, png=png, csv=csv)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-trace",
        required=True,
        type=Path,
        help="Fresh NoDVFS request_trace.csv",
    )
    parser.add_argument(
        "--dvfsc-trace",
        required=True,
        type=Path,
        help="Fresh DVFS-C request_trace.csv",
    )
    parser.add_argument(
        "--enpu-all-trace",
        required=True,
        type=Path,
        help="Fresh eNPU-All request_trace.csv",
    )
    parser.add_argument(
        "--slo-config",
        required=True,
        type=Path,
        help="Llama3-70B Azure-Code SLO JSON used by all three runs",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for Figure 22 PDF, PNG, and plotted CSV grid",
    )
    parser.add_argument(
        "--basename",
        default="figure_22_fleetsim_dvfs_timeseries",
        help="Shared output filename stem",
    )
    args = parser.parse_args()

    outputs = plot(
        args.baseline_trace.expanduser().resolve(strict=True),
        args.dvfsc_trace.expanduser().resolve(strict=True),
        args.enpu_all_trace.expanduser().resolve(strict=True),
        args.slo_config.expanduser().resolve(strict=True),
        args.output_dir.expanduser().resolve(),
        basename=args.basename,
    )
    print(f"Figure 22 PDF: {outputs.pdf}")
    print(f"Figure 22 PNG: {outputs.png}")
    print(f"Figure 22 plotted data: {outputs.csv}")


if __name__ == "__main__":
    main()
