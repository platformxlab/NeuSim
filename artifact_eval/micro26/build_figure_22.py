#!/usr/bin/env python3
"""Build the Figure 22 review bundle from three explicit fresh FleetSim traces.

This helper performs no simulation and discovers no results automatically.
Callers must identify the NoDVFS, DVFS-C, and eNPU-All request traces produced
by their fresh static-fleet runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from artifact_eval.micro26.plots import figure_22  # noqa: E402

AE_ROOT = REPO_ROOT / "artifact_eval" / "micro26"
DEFAULT_SLO_CONFIG = AE_ROOT / "config" / "figure_05_slo_llama3_70b_azure_code.json"
DEFAULT_OUTPUT_DIR = AE_ROOT / "reproduced" / "figure22-standalone"
ALLOCATION = {
    "name": "paper_static",
    "prefill_vpods": 20,
    "decode_vpods": 8,
    "chips_per_vpod": 4,
    "npu": "TPU v5p",
    "batch_size": 1,
    "dp_tp_pp": "1/4/1",
}


def sha256_file(path: Path) -> str:
    """Hash one explicit input or generated output."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """Record an explicit file path, size, and content hash."""

    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def git_record() -> dict[str, Any]:
    """Capture checkout identity without modifying Git state."""

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": git("status", "--short").splitlines(),
    }


def _relative_link(target: Path, report: Path) -> str:
    """Return a portable Markdown link relative to the generated report."""

    return Path(os.path.relpath(target, report.parent)).as_posix()


def render_report(
    output_dir: Path,
    traces: dict[str, Path],
    outputs: figure_22.Figure22Outputs,
    slo_config: Path,
    scheduler_slo_targets: list[figure_22.SLOTarget],
    paper_plot_slo_targets: list[figure_22.SLOTarget],
) -> Path:
    """Write the reviewer-facing Figure 22 index."""

    report = output_dir / "FIGURE_22_REVIEW.md"
    pdf_link = _relative_link(outputs.pdf, report)
    png_link = _relative_link(outputs.png, report)
    csv_link = _relative_link(outputs.csv, report)
    slo_link = _relative_link(slo_config, report)
    plotter_link = _relative_link(AE_ROOT / "plots" / "figure_22.py", report)
    helper_link = _relative_link(Path(__file__), report)
    provenance_link = _relative_link(output_dir / "figure_22_provenance.json", report)
    threshold_rows = []
    for scheduler, plotted in zip(
        scheduler_slo_targets, paper_plot_slo_targets, strict=True
    ):
        threshold_rows.append(
            "| "
            f"{scheduler.input_seqlen} | {scheduler.decode_seqlen} | "
            f"{scheduler.ttft_seconds:g} | {plotted.ttft_seconds:g} | "
            f"{scheduler.tpot_milliseconds / 1000.0:g} | "
            f"{plotted.tpot_milliseconds / 1000.0:g} |"
        )

    lines = [
        "# MICRO'26 Figure 22 review",
        "",
        "- Status: **COMPLETE**",
        "- Static allocation: 20 prefill vPods and 8 decode vPods",
        "- vPod configuration: TPU v5p, 4 chips, " "batch size 1, DP/TP/PP = 1/4/1",
        "- Policies: NoDVFS, DVFS-C, and eNPU-All",
        "- Rolling window: 5 minutes; plotted grid: 1 minute",
        "- Displayed interval: hours 12--24, matching the paper",
        "",
        "## Figure and plotted data",
        "",
        f"- [PDF figure]({pdf_link})",
        f"- [PNG preview]({png_link})",
        f"- [One-minute plotted data CSV]({csv_link})",
        f"- [Full provenance]({provenance_link})",
        "",
        "## Explicit simulation inputs",
        "",
        f"- NoDVFS: [{traces['baseline'].name}]"
        f"({_relative_link(traces['baseline'], report)})",
        f"- DVFS-C: [{traces['DVFSC'].name}]"
        f"({_relative_link(traces['DVFSC'], report)})",
        f"- eNPU-All: [{traces['CustomAll'].name}]"
        f"({_relative_link(traces['CustomAll'], report)})",
        f"- SLO buckets: [{slo_config.name}]({slo_link})",
        "",
        "The helper requires all three trace paths explicitly and does not "
        "discover or consume request-result data from the original "
        "`trace_util` checkout.",
        "",
        "## SLO-threshold precision",
        "",
        "FleetSim and the service-level DVFS scheduler use the exact `5x` "
        "targets in the SLO JSON. The original Figure 22 plotting script "
        "rounded both TTFT and TPOT targets in seconds to three decimal "
        "places before classifying requests. Only the plotted SLO-satisfaction "
        "curves use those rounded values here.",
        "",
        "| Input length <= | Total length <= | Scheduler TTFT (s) | "
        "Plot TTFT (s) | Scheduler TPOT (s) | Plot TPOT (s) |",
        "|---:|---:|---:|---:|---:|---:|",
        *threshold_rows,
        "",
        "## Source-format compatibility note",
        "",
        "The paper axis is labeled `Normalized Joule/Token`, but the original "
        "Figure 22 script plots a five-minute rolling mean of aggregate energy "
        "per completed request: `prefill_energy_J` for prefill and aggregate "
        "`decode_energy_J` for decode. It then normalizes each policy pointwise "
        "against the corresponding NoDVFS minute. This reproduction retains "
        "that exact source behavior; it does not divide by input or output "
        "token count.",
        "",
        "## Scripts",
        "",
        f"- Plotter: [{plotter_link}]({plotter_link})",
        f"- Review-bundle helper: [{helper_link}]({helper_link})",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def build(
    baseline_trace: Path,
    dvfsc_trace: Path,
    enpu_all_trace: Path,
    slo_config: Path,
    output_dir: Path,
) -> tuple[figure_22.Figure22Outputs, Path, Path]:
    """Generate Figure 22 outputs, provenance, and the Markdown review."""

    traces = {
        "baseline": baseline_trace.expanduser().resolve(strict=True),
        "DVFSC": dvfsc_trace.expanduser().resolve(strict=True),
        "CustomAll": enpu_all_trace.expanduser().resolve(strict=True),
    }
    if len(set(traces.values())) != len(traces):
        raise ValueError("Figure 22 requires three distinct policy traces")
    slo_config = slo_config.expanduser().resolve(strict=True)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scheduler_slo_targets = figure_22.load_slo_targets(
        slo_config, multiplier=figure_22.SLO_MULTIPLIER
    )
    paper_plot_slo_targets = figure_22.round_slo_targets_for_paper_plot(
        scheduler_slo_targets
    )

    outputs = figure_22.plot(
        traces["baseline"],
        traces["DVFSC"],
        traces["CustomAll"],
        slo_config,
        output_dir / "figures",
    )
    provenance_path = output_dir / "figure_22_provenance.json"
    provenance = {
        "schema_version": 1,
        "status": "complete",
        "paper_figure": 22,
        "completed_utc": datetime.now(UTC).isoformat(),
        "allocation": ALLOCATION,
        "policies": {
            "baseline": "NoDVFS",
            "DVFSC": "DVFS-C",
            "CustomAll": "eNPU-All",
        },
        "aggregation": {
            "rolling_window": figure_22.ROLLING_WINDOW,
            "resample_frequency": figure_22.RESAMPLE_FREQUENCY,
            "display_hours": [12, 24],
            "energy_quantity": "rolling mean aggregate energy per request",
            "legacy_axis_label": "Normalized Joule/Token",
            "normalization": "pointwise against NoDVFS at each minute",
            "slo_satisfaction": {
                "scheduler": {
                    "source": "exact 5x values in inputs.slo_config",
                    "thresholds": figure_22.slo_threshold_records(
                        scheduler_slo_targets
                    ),
                },
                "paper_plot": {
                    "source": "original fleetsim_dvfs_timeseries.py",
                    "rounding": (
                        "round TTFT and TPOT expressed in seconds to "
                        f"{figure_22.PAPER_SLO_DECIMAL_PLACES_SECONDS} "
                        "decimal places"
                    ),
                    "decimal_places_seconds": (
                        figure_22.PAPER_SLO_DECIMAL_PLACES_SECONDS
                    ),
                    "thresholds": figure_22.slo_threshold_records(
                        paper_plot_slo_targets
                    ),
                },
            },
        },
        "inputs": {
            "request_traces": {
                name: file_record(path) for name, path in traces.items()
            },
            "slo_config": file_record(slo_config),
        },
        "outputs": {
            "pdf": file_record(outputs.pdf),
            "png": file_record(outputs.png),
            "plotted_csv": file_record(outputs.csv),
        },
        "software": {
            "git": git_record(),
            "plotter": file_record(AE_ROOT / "plots" / "figure_22.py"),
            "builder": file_record(Path(__file__)),
        },
        "data_policy": {
            "automatic_result_discovery": False,
            "original_trace_util_result_data_consumed": False,
            "caller_supplied_traces_only": True,
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = render_report(
        output_dir,
        traces,
        outputs,
        slo_config,
        scheduler_slo_targets,
        paper_plot_slo_targets,
    )
    return outputs, report, provenance_path


def main() -> int:
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
        type=Path,
        default=DEFAULT_SLO_CONFIG,
        help="SLO JSON shared by all runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Figure 22 review-bundle output directory",
    )
    args = parser.parse_args()

    outputs, report, provenance = build(
        args.baseline_trace,
        args.dvfsc_trace,
        args.enpu_all_trace,
        args.slo_config,
        args.output_dir,
    )
    print(f"Figure 22 PDF: {outputs.pdf}")
    print(f"Figure 22 PNG: {outputs.png}")
    print(f"Figure 22 plotted data: {outputs.csv}")
    print(f"Figure 22 review: {report}")
    print(f"Figure 22 provenance: {provenance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
