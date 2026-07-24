#!/usr/bin/env python3
"""Run the static FleetSim experiment and plot MICRO'26 Figure 5.

Only caller-supplied request traces and files in this checkout are accepted as
inputs. The runner consumes no precomputed per-request, result, or plot cache.
The static allocation is supplied by a versioned input configuration, and the
run creates a fresh NeuSim op-analysis cache beneath its output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
AE_ROOT = REPO_ROOT / "artifact_eval" / "micro26"
REFERENCE_CHIP_OUTPUT = AE_ROOT / "reference_outputs" / "chip"
SLO_CONFIG = AE_ROOT / "config" / "figure_05_slo_llama3_70b_azure_code.json"
PLOT_SCRIPT = AE_ROOT / "plots" / "figure_05.py"
ALLOCATION_CONFIG = (
    REPO_ROOT / "configs" / "fleetsim" / "figure_05_llama3_70b_tpuv5p_p20d8.json"
)
ALLOCATION_COUNTS = {"prefill": 20, "decode": 8}
RUN_NAME = "NoDVFS"
FIGURE_NAME = "figure_05_slo_slack.pdf"
AZURE_COLUMNS = ("TIMESTAMP", "ContextTokens", "GeneratedTokens")
RESULT_COLUMNS = (
    "enqueue_timestamp",
    "prefill_end_timestamp",
    "decode_end_timestamp",
    "TTFT_ns",
    "TPOT_ns",
)
REFERENCE_TRACE_ROWS = 2_490_144
REFERENCE_TRACE_SPAN_HOURS = 23.99995430611111
REFERENCE_TRACE_SHA256 = (
    "89a0a30da525745540642efc09606bcda1163f766186937141d4ac714390cc9b"
)
PAPER_TRACE_MAX_HOURS = 26.0
DEFAULT_SMOKE_REQUESTS = 32
ROLLING_WINDOW_MINUTES = 1.0


def utc_now() -> str:
    """Return a stable UTC timestamp for provenance records."""

    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """Return path, size, and content hash for one provenance input/output."""

    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def parse_azure_timestamp(value: str) -> datetime:
    """Parse the ISO-8601 timestamps used by the Azure inference trace."""

    normalized = value.strip().replace("Z", "+00:00")
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    return timestamp


def inspect_azure_trace(path: Path) -> dict[str, Any]:
    """Validate an Azure trace and summarize its row count and time span."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [
            column for column in AZURE_COLUMNS if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"{path} is not an Azure request trace; missing: {', '.join(missing)}"
            )
        row_count = 0
        first_timestamp: datetime | None = None
        last_timestamp: datetime | None = None
        for row in reader:
            timestamp = parse_azure_timestamp(row["TIMESTAMP"])
            if first_timestamp is None:
                first_timestamp = timestamp
            if last_timestamp is not None and timestamp < last_timestamp:
                raise ValueError(f"{path} is not sorted by TIMESTAMP")
            last_timestamp = timestamp
            row_count += 1

    if first_timestamp is None or last_timestamp is None:
        raise ValueError(f"{path} contains no requests")
    span_hours = (last_timestamp - first_timestamp).total_seconds() / 3600.0
    return {
        **file_record(path),
        "rows": row_count,
        "first_timestamp": first_timestamp.isoformat(),
        "last_timestamp": last_timestamp.isoformat(),
        "span_hours": span_hours,
    }


def inspect_result_trace(path: Path) -> dict[str, Any]:
    """Summarize a fresh FleetSim request trace without retaining its rows."""

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [
            column for column in RESULT_COLUMNS if column not in reader.fieldnames
        ]
        if missing:
            raise ValueError(
                f"{path} is not a FleetSim request trace; missing: {', '.join(missing)}"
            )
        rows = 0
        first_enqueue_ns: int | None = None
        last_enqueue_ns: int | None = None
        last_completion_ns: int | None = None
        for row in reader:
            enqueue_ns = int(row["enqueue_timestamp"])
            completion_ns = int(row["decode_end_timestamp"])
            first_enqueue_ns = (
                enqueue_ns
                if first_enqueue_ns is None
                else min(first_enqueue_ns, enqueue_ns)
            )
            last_enqueue_ns = (
                enqueue_ns
                if last_enqueue_ns is None
                else max(last_enqueue_ns, enqueue_ns)
            )
            last_completion_ns = (
                completion_ns
                if last_completion_ns is None
                else max(last_completion_ns, completion_ns)
            )
            rows += 1

    if (
        first_enqueue_ns is None
        or last_enqueue_ns is None
        or last_completion_ns is None
    ):
        raise ValueError(f"{path} contains no completed requests")
    nanoseconds_per_hour = 3600.0 * 1e9
    return {
        **file_record(path),
        "rows": rows,
        "first_enqueue_hours": first_enqueue_ns / nanoseconds_per_hour,
        "last_enqueue_hours": last_enqueue_ns / nanoseconds_per_hour,
        "last_completion_hours": last_completion_ns / nanoseconds_per_hour,
    }


def source_tree_record() -> dict[str, Any]:
    """Hash the simulator and Figure 5 code that determine this run."""

    paths = {
        Path(__file__).resolve(),
        PLOT_SCRIPT,
        SLO_CONFIG,
        ALLOCATION_CONFIG,
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "configs" / "models" / "llama3-70b.json",
        REPO_ROOT / "configs" / "chips" / "tpuv5p.json",
        REPO_ROOT / "configs" / "systems" / "system_config.json",
    }
    paths.update(
        path
        for path in (REPO_ROOT / "neusim").rglob("*.py")
        if "tests" not in path.relative_to(REPO_ROOT).parts
        and "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in sorted(paths):
        relative = path.relative_to(REPO_ROOT).as_posix()
        content_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
        records.append({"path": relative, "sha256": content_hash})
    return {"sha256": digest.hexdigest(), "files": records}


def validate_reference_trace(trace_record: dict[str, Any]) -> None:
    """Require the exact one-day Azure Code trace used for Figure 5."""

    mismatches: list[str] = []
    if trace_record["rows"] != REFERENCE_TRACE_ROWS:
        mismatches.append(
            f"rows={trace_record['rows']:,} (expected {REFERENCE_TRACE_ROWS:,})"
        )
    if trace_record["sha256"] != REFERENCE_TRACE_SHA256:
        mismatches.append(
            f"sha256={trace_record['sha256']} (expected {REFERENCE_TRACE_SHA256})"
        )
    if abs(trace_record["span_hours"] - REFERENCE_TRACE_SPAN_HOURS) > 1e-9:
        mismatches.append(
            f"span={trace_record['span_hours']:.9f}h "
            f"(expected {REFERENCE_TRACE_SPAN_HOURS:.9f}h)"
        )
    if mismatches:
        raise ValueError(
            "full Figure 5 requires the exact unsampled Azure Code one-day trace; "
            + "; ".join(mismatches)
            + ". Use --smoke or --max-requests for a workflow-only diagnostic."
        )


def git_record() -> dict[str, Any]:
    """Capture the exact checkout state without changing it."""

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": git("status", "--short").splitlines(),
    }


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically update a provenance JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_and_tee(
    command: list[str],
    *,
    log_path: Path,
    environment: dict[str, str],
) -> None:
    """Run one command while preserving both reviewer-visible and file logs."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"+ {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def validate_static_vpod_stats(static_vpod_path: Path) -> dict[str, Any]:
    """Prove that FleetSim kept the requested static vPod allocation."""

    document = json.loads(static_vpod_path.read_text(encoding="utf-8"))
    expected = ALLOCATION_COUNTS
    observed: dict[str, int] = {}
    for phase, expected_count in expected.items():
        entries = document.get(phase)
        if not isinstance(entries, list):
            raise ValueError(f"{static_vpod_path} has no {phase!r} vPod list")
        observed[phase] = len(entries)
        if len(entries) != expected_count:
            raise ValueError(
                f"the allocation created {len(entries)} {phase} vPods; "
                f"expected {expected_count}"
            )
        for entry in entries:
            if (
                entry.get("npu_type") != "5p"
                or int(entry.get("num_chips", -1)) != 4
                or entry.get("pcfg") != "bs1-dp1-tp4-pp1"
            ):
                raise ValueError(
                    f"{static_vpod_path} contains a non-paper {phase} vPod: {entry}"
                )
    return {"expected": expected, "observed": observed}


def prepare_output_directory(path: Path) -> Path:
    """Create a new output root while protecting the reference outputs."""

    output = path.expanduser().resolve()
    reference_output = REFERENCE_CHIP_OUTPUT.resolve()
    if output == reference_output or reference_output in output.parents:
        raise ValueError(
            "Figure 5 must use a separate output directory; refusing to modify "
            f"the reference output at {reference_output}"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"{output} is not empty. Choose a fresh --output-dir so provenance "
            "cannot be mixed with an earlier run."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def render_report(
    output_dir: Path,
    provenance: dict[str, Any],
    run_records: list[dict[str, Any]],
) -> Path:
    """Write the reviewer-facing index for this standalone experiment."""

    mode = provenance["mode"]
    lines = [
        "# MICRO'26 Figure 5 review",
        "",
        f"- Status: **{provenance['status'].upper()}**",
        f"- Run mode: `{mode}`",
        "- Static allocation: 20 prefill and 8 decode vPods",
        "- NPU configuration per vPod: TPU v5p, 4 chips, DP/TP/PP = 1/4/1, batch size 1",
        f"- Rolling window: {ROLLING_WINDOW_MINUTES:g} minute",
        "- SLO: 5x the single-request four-chip latency, bucketed at P33/P66/P100",
        "- Precomputed per-request/result/plot cache: **not consumed**",
        "- NeuSim op-analysis cache: freshly generated inside each run directory",
        "",
    ]
    if mode != "full":
        lines.extend(
            [
                "> This is a workflow check with a request limit. It is not a "
                "paper-scale Figure 5 reproduction.",
                "",
            ]
        )
    lines.extend(
        [
            "## Fresh input provenance",
            "",
            f"- Trace: `{provenance['inputs']['trace']['path']}`",
            f"- Trace SHA-256: `{provenance['inputs']['trace']['sha256']}`",
            f"- Trace rows: {provenance['inputs']['trace']['rows']:,}",
            f"- Trace span: {provenance['inputs']['trace']['span_hours']:.3f} hours",
            "- Full provenance: "
            "[figure_05_provenance.json](figure_05_provenance.json)",
            "",
            "## Outputs",
            "",
        ]
    )
    for record in run_records:
        policy = record["policy"]
        relative_figure = Path(record["figure"]["path"]).relative_to(output_dir)
        relative_trace = Path(record["request_trace"]["path"]).relative_to(output_dir)
        relative_stats = Path(record["stats"]["path"]).relative_to(output_dir)
        relative_log = Path(record["simulation_log"]["path"]).relative_to(output_dir)
        lines.extend(
            [
                f"### {policy}",
                "",
                f"- Figure: [{relative_figure}]({relative_figure.as_posix()})",
                f"- Fresh request trace: [{relative_trace}]({relative_trace.as_posix()})",
                f"- Summary statistics: [{relative_stats}]({relative_stats.as_posix()})",
                f"- Simulation log: [{relative_log}]({relative_log.as_posix()})",
                f"- Completed requests: {record['request_trace']['rows']:,}",
                f"- Last arrival: {record['request_trace']['last_enqueue_hours']:.3f} hours",
                f"- Last completion: {record['request_trace']['last_completion_hours']:.3f} hours",
                "",
            ]
        )
    lines.extend(
        [
            "The plotting script is "
            "[artifact_eval/micro26/plots/figure_05.py]"
            "(../plots/figure_05.py). It reads only the fresh request trace "
            "listed above and the versioned SLO configuration.",
            "",
        ]
    )
    report = output_dir / "FIGURE_05_REVIEW.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-file",
        required=True,
        type=Path,
        help="Explicit AzureLLMInferenceTrace_code_1day.csv input path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Fresh output directory (defaults to reproduced/figure05-standalone)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Run only {DEFAULT_SMOKE_REQUESTS} requests unless --max-requests is set",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        help="Limit requests for diagnosis; any limit marks the report non-paper-scale",
    )
    args = parser.parse_args()

    if args.max_requests is not None and args.max_requests <= 0:
        parser.error("--max-requests must be positive")
    max_requests = args.max_requests
    if args.smoke and max_requests is None:
        max_requests = DEFAULT_SMOKE_REQUESTS
    mode = "full" if max_requests is None else ("smoke" if args.smoke else "limited")

    trace_path = args.trace_file.expanduser().resolve(strict=True)
    if not trace_path.is_file():
        parser.error(f"--trace-file is not a file: {trace_path}")
    trace_record = inspect_azure_trace(trace_path)
    if mode == "full":
        try:
            validate_reference_trace(trace_record)
        except ValueError as exc:
            parser.error(str(exc))

    if args.output_dir is None:
        output_path = (
            Path("/tmp") / f"neusim-micro26-figure05-{mode}-{os.getpid()}"
            if mode != "full"
            else AE_ROOT / "reproduced" / "figure05-standalone"
        )
    else:
        output_path = args.output_dir
    output_dir = prepare_output_directory(output_path)
    provenance_path = output_dir / "figure_05_provenance.json"

    for required in [SLO_CONFIG, PLOT_SCRIPT, ALLOCATION_CONFIG]:
        if not required.is_file():
            raise FileNotFoundError(f"required Figure 5 input is missing: {required}")

    provenance: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        "mode": mode,
        "paper_figure": 5,
        "static_allocation": file_record(ALLOCATION_CONFIG),
        "max_requests": max_requests,
        "rolling_window_minutes": ROLLING_WINDOW_MINUTES,
        "slo_multiplier": "5x",
        "inputs": {
            "trace": {
                **trace_record,
                "reference_trace_match": mode == "full",
            },
            "slo_config": file_record(SLO_CONFIG),
            "allocation_config": file_record(ALLOCATION_CONFIG),
        },
        "software": {"git": git_record(), "source_tree": source_tree_record()},
        "cache_policy": {
            "precomputed_request_result_plot_cache_input": None,
            "precomputed_request_result_plot_cache_consumed": False,
            "npusim_backend_op_analysis_cache": ("fresh inside this output directory"),
        },
        "runs": [],
    }
    write_json(provenance_path, provenance)

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT), environment.get("PYTHONPATH", "")])
    )
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"

    try:
        for policy in (RUN_NAME,):
            run_dir = output_dir / "runs" / policy
            run_dir.mkdir(parents=True)
            backend_cache = run_dir / ".cache" / "npusim_backend"
            environment["NEUSIM_RESULTS_DIR"] = str(run_dir)

            simulation_command = [
                sys.executable,
                "-m",
                "neusim.run_scripts.fleetsim_main",
                "--model=llama3-70b",
                "--request_pattern=trace",
                "--trace=Azure-Code",
                f"--trace_file={trace_path}",
                f"--configs_path={REPO_ROOT / 'configs'}",
                "--request_rate=1.0",
                f"--max_timestamp_hours={PAPER_TRACE_MAX_HOURS:g}",
                f"--static_vpod_allocation={ALLOCATION_CONFIG}",
                f"--output_dir={run_dir}",
                f"--npusim_backend_cache_dir={backend_cache}",
                "--tqdm=false",
                "--enable_dvfs_power_model=true",
            ]
            if max_requests is not None:
                simulation_command.append(f"--max_num_requests={max_requests}")

            simulation_log = run_dir / "simulation.log"
            run_and_tee(
                simulation_command,
                log_path=simulation_log,
                environment=environment,
            )
            request_trace = run_dir / "request_trace.csv"
            stats_path = run_dir / "stats.json"
            static_vpod_path = run_dir / "static_vpod_stats.json"
            for expected_output in (request_trace, stats_path, static_vpod_path):
                if not expected_output.is_file():
                    raise FileNotFoundError(
                        f"FleetSim did not produce {expected_output}"
                    )
            request_record = inspect_result_trace(request_trace)
            expected_requests = (
                min(trace_record["rows"], max_requests)
                if max_requests is not None
                else trace_record["rows"]
            )
            if request_record["rows"] != expected_requests:
                raise RuntimeError(
                    f"FleetSim completed {request_record['rows']:,} requests; "
                    f"expected {expected_requests:,}"
                )
            static_vpod_record = validate_static_vpod_stats(static_vpod_path)

            figure_name = FIGURE_NAME
            figure_path = output_dir / "figures" / figure_name
            plot_command = [
                sys.executable,
                str(PLOT_SCRIPT),
                f"--request-trace={request_trace}",
                f"--slo-config={SLO_CONFIG}",
                f"--output={figure_path}",
            ]
            plot_log = run_dir / "plot.log"
            run_and_tee(
                plot_command,
                log_path=plot_log,
                environment=environment,
            )

            run_record = {
                "policy": policy,
                "simulation_command": simulation_command,
                "plot_command": plot_command,
                "allocation_config": file_record(ALLOCATION_CONFIG),
                "request_trace": request_record,
                "stats": file_record(stats_path),
                "static_vpods": {
                    **file_record(static_vpod_path),
                    **static_vpod_record,
                },
                "figure": file_record(figure_path),
                "simulation_log": file_record(simulation_log),
                "plot_log": file_record(plot_log),
                "npusim_backend_op_analysis_cache": str(backend_cache.resolve()),
            }
            provenance["runs"].append(run_record)
            write_json(provenance_path, provenance)

        provenance["status"] = "complete"
        provenance["completed_utc"] = utc_now()
        write_json(provenance_path, provenance)
        report = render_report(output_dir, provenance, provenance["runs"])
        print(f"Figure 5 review: {report}")
        print(f"Figure 5 provenance: {provenance_path}")
        return 0
    except BaseException as exc:
        provenance["status"] = "failed"
        provenance["completed_utc"] = utc_now()
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        write_json(provenance_path, provenance)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
