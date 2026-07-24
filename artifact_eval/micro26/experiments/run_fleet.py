#!/usr/bin/env python3
"""Run the shared fresh static-fleet reproductions for MICRO'26 Figures 5 and 22.

The default first wave runs the NoDVFS Figure 5 replay while validating the
artifact's supplied DVFS-C and eNPU-All lookup caches.  With
``--regenerate-lookup-cache``, both caches are instead rebuilt in parallel with
the NoDVFS replay.  The second wave replays the
two DVFS policies in parallel.  Figure 22 shares the fresh NoDVFS request trace
from Figure 5; no request result or lookup cache from ``trace_util`` is read.

A nonempty output root is accepted only with ``--resume``.  Reuse is
provenance-bound: the exact Azure trace, current source tree, complete result
schema, versioned static allocation, lookup-cache coverage, and cache-hit counters
are validated before a completed stage is skipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from artifact_eval.micro26.experiments import run_figure_05  # noqa: E402

AE_ROOT = REPO_ROOT / "artifact_eval" / "micro26"
DEFAULT_OUTPUT_DIR = AE_ROOT / "reproduced" / "fleet"
DEFAULT_TRACE_FILE = AE_ROOT / "data" / "azure" / "AzureLLMInferenceTrace_code_1day.csv"
DEFAULT_LOOKUP_CACHE_DIR = AE_ROOT / "data" / "dvfs_lookup"
FIGURE05_RUNNER = AE_ROOT / "experiments" / "run_figure_05.py"
CACHE_GENERATOR = AE_ROOT / "experiments" / "generate_figure_22_dvfs_cache.py"
FIGURE22_BUILDER = AE_ROOT / "build_figure_22.py"
ALLOCATION_CONFIG = run_figure_05.ALLOCATION_CONFIG
ALLOCATION_ID = "paper_static"
BASELINE_RUN_NAME = run_figure_05.RUN_NAME
SLO_CONFIG = run_figure_05.SLO_CONFIG
POLICIES = ("DVFSC", "CustomAll")
POLICY_LABELS = {"DVFSC": "DVFS-C", "CustomAll": "eNPU-All"}


def default_cache_worker_counts(cpu_count: int | None = None) -> tuple[int, int]:
    """Split cache CPUs 72/28 after reserving FleetSim and host headroom."""

    count = max(2, cpu_count or os.cpu_count() or 2)
    # On the 48-core reference host this reserves eight cores plus the
    # concurrent NoDVFS process, leaving 39 cache workers: 28 DVFSC + 11
    # CustomAll. Both generator CLIs independently enforce a 32-worker cap.
    host_headroom = max(2, round(count / 6))
    cache_pool = max(2, count - host_headroom - 1)
    dvfsc = min(32, max(1, round(cache_pool * 0.72)))
    customall = min(32, max(1, cache_pool - dvfsc))
    return dvfsc, customall


(
    DEFAULT_DVFSC_CACHE_WORKERS,
    DEFAULT_CUSTOMALL_CACHE_WORKERS,
) = default_cache_worker_counts()
EXPECTED_REQUEST_ROWS = run_figure_05.REFERENCE_TRACE_ROWS
EXPECTED_RESULT_COLUMNS = (
    "input_seqlen",
    "output_seqlen",
    "enqueue_timestamp",
    "prefill_start_timestamp",
    "prefill_end_timestamp",
    "decode_start_timestamp",
    "decode_end_timestamp",
    "prefill_queuing_delay_ns",
    "prefill_latency_ns",
    "TTFT_ns",
    "decode_queuing_delay_per_iteration_ns",
    "TPOT_ns",
    "effective_TPOT_ns",
    "prefill_energy_J",
    "decode_energy_J",
    "decode_energy_per_token_J",
    "prefill_cost_dollars",
    "decode_cost_dollars",
)
POSITIVE_RESULT_COLUMNS = (
    "input_seqlen",
    "output_seqlen",
    "prefill_end_timestamp",
    "decode_end_timestamp",
    "prefill_latency_ns",
    "TTFT_ns",
    "TPOT_ns",
    "effective_TPOT_ns",
    "prefill_energy_J",
    "decode_energy_J",
    "decode_energy_per_token_J",
    "prefill_cost_dollars",
    "decode_cost_dollars",
)


class WorkflowValidationError(RuntimeError):
    """Raised when a resumable artifact is stale, mixed, or incomplete."""


class ParallelStageError(RuntimeError):
    """One process in a parallel wave failed; siblings were stopped."""

    def __init__(
        self,
        returncodes: Mapping[str, int],
        log_paths: Mapping[str, Path],
    ):
        self.returncodes = dict(returncodes)
        self.log_paths = dict(log_paths)
        details = "; ".join(
            f"{name} exited {code}; see {self.log_paths[name]}"
            for name, code in self.returncodes.items()
            if code != 0
        )
        super().__init__(details)


@dataclass
class RunningCommand:
    """One subprocess and its open combined-output log."""

    name: str
    command: list[str]
    log_path: Path
    process: subprocess.Popen[str]
    log_stream: Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(document: Any) -> str:
    encoded = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowValidationError(f"cannot read {path}: {error}") from error
    if not isinstance(document, dict):
        raise WorkflowValidationError(f"{path} is not a JSON object")
    return document


def write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_record() -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "status_porcelain": git("status", "--short").splitlines(),
    }


def common_environment() -> dict[str, str]:
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
    return environment


def launch_commands(
    specifications: Sequence[tuple[str, list[str], Path]],
    environment: Mapping[str, str],
) -> None:
    """Launch one wave concurrently and stop sibling process groups on failure."""

    running: list[RunningCommand] = []
    try:
        for name, command, log_path in specifications:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_stream = log_path.open("w", encoding="utf-8")
            print(f"[launch:{name}] {shlex.join(command)}", flush=True)
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                env=dict(environment),
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            running.append(RunningCommand(name, command, log_path, process, log_stream))
    except BaseException:
        for stage in running:
            try:
                os.killpg(stage.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            stage.process.wait()
            stage.log_stream.close()
        raise

    pending = {stage.name: stage for stage in running}
    returncodes: dict[str, int] = {}
    log_paths = {stage.name: stage.log_path for stage in running}
    terminate_deadline: float | None = None
    while pending:
        made_progress = False
        for name, stage in list(pending.items()):
            returncode = stage.process.poll()
            if returncode is None:
                continue
            made_progress = True
            stage.log_stream.close()
            del pending[name]
            returncodes[name] = returncode
            if returncode == 0:
                print(f"[complete:{name}] {stage.log_path}", flush=True)
                continue
            print(
                f"[failed:{name}] exit={returncode}; see {stage.log_path}",
                flush=True,
            )
            if terminate_deadline is None:
                terminate_deadline = time.monotonic() + 10.0
                for sibling in pending.values():
                    try:
                        os.killpg(sibling.process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

        if terminate_deadline is not None and time.monotonic() >= terminate_deadline:
            for sibling in pending.values():
                try:
                    os.killpg(sibling.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            terminate_deadline = math.inf
        if pending and not made_progress:
            time.sleep(0.2)

    if any(returncode != 0 for returncode in returncodes.values()):
        raise ParallelStageError(returncodes, log_paths)


def prepare_output_root(path: Path, *, resume: bool) -> Path:
    output = path.expanduser().resolve()
    reference_root = (AE_ROOT / "reference_outputs").resolve()
    if output == reference_root or reference_root in output.parents:
        raise ValueError(
            "Figures 5/22 require a separate output directory; refusing to "
            f"modify the reference-output tree at {reference_root}"
        )
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(
            f"{output} is not empty. Pass --resume to validate and reuse "
            "completed stages, or choose a fresh --output-dir."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def validate_exact_trace(path: Path) -> dict[str, Any]:
    record = run_figure_05.inspect_azure_trace(path)
    run_figure_05.validate_reference_trace(record)
    return record


def validate_result_trace(
    path: Path, *, expected_rows: int = EXPECTED_REQUEST_ROWS
) -> dict[str, Any]:
    """Stream-validate every expected numeric cell in a FleetSim trace."""

    path = path.expanduser().resolve(strict=True)
    rows = 0
    first_enqueue_ns: int | None = None
    last_enqueue_ns: int | None = None
    last_completion_ns: int | None = None
    minima = {name: math.inf for name in POSITIVE_RESULT_COLUMNS}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as error:
            raise WorkflowValidationError(f"{path} is empty") from error
        missing = [column for column in EXPECTED_RESULT_COLUMNS if column not in header]
        if missing:
            raise WorkflowValidationError(
                f"{path} is missing result columns: {', '.join(missing)}"
            )
        indices = {column: header.index(column) for column in EXPECTED_RESULT_COLUMNS}
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise WorkflowValidationError(
                    f"{path}:{line_number} has {len(row)} fields; "
                    f"expected {len(header)}"
                )
            values: dict[str, float] = {}
            for column, index in indices.items():
                raw = row[index]
                if raw == "":
                    raise WorkflowValidationError(
                        f"{path}:{line_number} has an empty {column}"
                    )
                try:
                    value = float(raw)
                except ValueError as error:
                    raise WorkflowValidationError(
                        f"{path}:{line_number} has nonnumeric {column}={raw!r}"
                    ) from error
                if not math.isfinite(value):
                    raise WorkflowValidationError(
                        f"{path}:{line_number} has non-finite {column}={raw!r}"
                    )
                values[column] = value
            for column in POSITIVE_RESULT_COLUMNS:
                minima[column] = min(minima[column], values[column])
            enqueue_ns = int(values["enqueue_timestamp"])
            completion_ns = int(values["decode_end_timestamp"])
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

    if rows != expected_rows:
        raise WorkflowValidationError(
            f"{path} contains {rows:,} completed requests; "
            f"expected {expected_rows:,}"
        )
    nonpositive = {column: value for column, value in minima.items() if value <= 0}
    if nonpositive:
        raise WorkflowValidationError(
            f"{path} has non-positive completed-request values: {nonpositive}"
        )
    assert (
        first_enqueue_ns is not None
        and last_enqueue_ns is not None
        and last_completion_ns is not None
    )
    ns_per_hour = 3600.0 * 1e9
    return {
        **file_record(path),
        "rows": rows,
        "columns": header,
        "all_expected_cells_numeric_and_finite": True,
        "positive_column_minima": minima,
        "first_enqueue_hours": first_enqueue_ns / ns_per_hour,
        "last_enqueue_hours": last_enqueue_ns / ns_per_hour,
        "last_completion_hours": last_completion_ns / ns_per_hour,
    }


def _require_record_matches(
    path: Path, recorded: Mapping[str, Any], label: str
) -> dict[str, Any]:
    actual = file_record(path)
    for field in ("path", "bytes", "sha256"):
        if recorded.get(field) != actual[field]:
            raise WorkflowValidationError(
                f"{label} {field} does not match its provenance record"
            )
    return actual


def validate_figure05_bundle(
    figure05_root: Path,
    trace_record: Mapping[str, Any],
    current_source_sha256: str,
) -> dict[str, Any]:
    provenance_path = figure05_root / "figure_05_provenance.json"
    provenance = read_json(provenance_path)
    if provenance.get("status") != "complete":
        raise WorkflowValidationError(f"{provenance_path} is not marked complete")
    if provenance.get("mode") != "full":
        raise WorkflowValidationError(f"{provenance_path} is not a full experiment")
    allocation_input = provenance.get("static_allocation", {})
    if allocation_input.get("sha256") != file_record(ALLOCATION_CONFIG)["sha256"]:
        raise WorkflowValidationError(
            f"{provenance_path} does not bind the current allocation input"
        )
    input_trace = provenance.get("inputs", {}).get("trace", {})
    if (
        input_trace.get("sha256") != trace_record["sha256"]
        or input_trace.get("rows") != EXPECTED_REQUEST_ROWS
        or input_trace.get("reference_trace_match") is not True
    ):
        raise WorkflowValidationError(
            f"{provenance_path} does not bind the exact Azure one-day trace"
        )
    recorded_source = (
        provenance.get("software", {}).get("source_tree", {}).get("sha256")
    )
    if recorded_source != current_source_sha256:
        raise WorkflowValidationError(
            f"{provenance_path} source hash differs from the current simulator"
        )
    runs = provenance.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise WorkflowValidationError(f"{provenance_path} must contain exactly one run")
    run = runs[0]
    if run.get("policy") != BASELINE_RUN_NAME:
        raise WorkflowValidationError(
            f"{provenance_path} run is not the NoDVFS baseline"
        )
    request_trace = figure05_root / "runs" / BASELINE_RUN_NAME / "request_trace.csv"
    request_record = validate_result_trace(request_trace)
    if run.get("request_trace", {}).get("sha256") != request_record["sha256"]:
        raise WorkflowValidationError(
            "Figure 5 request trace hash differs from its provenance"
        )
    static_path = figure05_root / "runs" / BASELINE_RUN_NAME / "static_vpod_stats.json"
    static = run_figure_05.validate_static_vpod_stats(static_path)
    _require_record_matches(static_path, run.get("static_vpods", {}), "static vPods")
    stats_path = figure05_root / "runs" / BASELINE_RUN_NAME / "stats.json"
    _require_record_matches(stats_path, run.get("stats", {}), "Figure 5 stats")
    figure_path = Path(run.get("figure", {}).get("path", ""))
    _require_record_matches(figure_path, run.get("figure", {}), "Figure 5 PDF")
    report_path = figure05_root / "FIGURE_05_REVIEW.md"
    if not report_path.is_file():
        raise WorkflowValidationError(f"missing {report_path}")
    backend_cache = (
        figure05_root / "runs" / BASELINE_RUN_NAME / ".cache" / "npusim_backend"
    )
    if not backend_cache.is_dir() or not any(backend_cache.rglob("metadata.json")):
        raise WorkflowValidationError(
            f"fresh Figure 5 backend cache is incomplete: {backend_cache}"
        )
    return {
        "provenance": file_record(provenance_path),
        "request_trace": request_record,
        "static_vpods": {**file_record(static_path), **static},
        "stats": file_record(stats_path),
        "figure": file_record(figure_path),
        "report": file_record(report_path),
        "backend_cache": str(backend_cache.resolve()),
    }


def validate_cache_manifest(
    cache_root: Path, policy: str, trace_record: Mapping[str, Any]
) -> dict[str, Any]:
    manifest_path = cache_root / policy / "manifest.json"
    manifest = read_json(manifest_path)
    identity = manifest.get("identity", {})
    if not isinstance(identity, dict) or identity.get("policy") != policy:
        raise WorkflowValidationError(f"{manifest_path} policy mismatch")
    if canonical_json_sha256(identity) != manifest.get("identity_sha256"):
        raise WorkflowValidationError(
            f"{manifest_path} identity hash does not match its contents"
        )
    expected_topology = {
        "prefill_vpods": 20,
        "decode_vpods": 8,
        "num_chips": 4,
        "batch_size": 1,
        "dp": 1,
        "tp": 4,
        "pp": 1,
        "ep": 1,
    }
    if (
        identity.get("model") != "llama3-70b"
        or identity.get("version") != "5p"
        or identity.get("phases") != ["prefill", "decode"]
        or identity.get("budgets") != [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        or identity.get("topology") != expected_topology
    ):
        raise WorkflowValidationError(
            f"{manifest_path} does not match the paper model/topology/budgets"
        )
    identity_trace = identity.get("trace", {})
    if (
        identity_trace.get("sha256") != trace_record["sha256"]
        or identity_trace.get("rows") != EXPECTED_REQUEST_ROWS
    ):
        raise WorkflowValidationError(
            f"{manifest_path} does not bind the exact Azure trace"
        )
    execution = manifest.get("execution", {})
    coverage = manifest.get("coverage", {})
    if execution.get("state") != "complete" or coverage.get("status") != "complete":
        raise WorkflowValidationError(f"{manifest_path} is not complete")
    if (
        coverage.get("shape_pairs") != 787
        or coverage.get("grouped_files") != 1574
        or coverage.get("budgets_per_file") != 9
    ):
        raise WorkflowValidationError(
            f"{manifest_path} does not have exact 787-shape/two-phase coverage"
        )
    if (
        manifest.get("identity", {})
        .get("algorithm", {})
        .get("detailed_dvfs_power_model")
        is not True
        or coverage.get("detailed_dvfs_power_model") is not True
        or not isinstance(coverage.get("files_with_budget_variation"), int)
        or coverage["files_with_budget_variation"] <= 0
        or not isinstance(coverage.get("files_with_positive_energy_saving"), int)
        or coverage["files_with_positive_energy_saving"] <= 0
    ):
        raise WorkflowValidationError(
            f"{manifest_path} does not certify a nondegenerate detailed-power cache"
        )
    policy_root = cache_root / policy
    expected_files = coverage["grouped_files"]
    if sum(1 for _ in policy_root.rglob("bs1.json")) != expected_files:
        raise WorkflowValidationError(
            f"{policy_root} file count differs from cache manifest"
        )
    output_tree = coverage.get("output_tree", {})
    records = output_tree.get("files") if isinstance(output_tree, dict) else None
    if not isinstance(records, list) or len(records) != expected_files:
        raise WorkflowValidationError(
            f"{manifest_path} has no complete output-tree inventory"
        )
    tree_digest = hashlib.sha256()
    recorded_paths: set[str] = set()
    for record in sorted(records, key=lambda item: str(item.get("path", ""))):
        if not isinstance(record, dict):
            raise WorkflowValidationError(
                f"{manifest_path} contains a non-object cache file record"
            )
        relative_text = record.get("path")
        if not isinstance(relative_text, str):
            raise WorkflowValidationError(f"{manifest_path} has an invalid cache path")
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.name != "bs1.json"
        ):
            raise WorkflowValidationError(
                f"{manifest_path} contains an unsafe cache path: {relative_text}"
            )
        target = (policy_root / relative).resolve(strict=True)
        if policy_root.resolve() not in target.parents:
            raise WorkflowValidationError(
                f"{manifest_path} cache path escapes its policy root: {relative_text}"
            )
        actual_sha256 = sha256_file(target)
        if target.stat().st_size != record.get("bytes") or actual_sha256 != record.get(
            "sha256"
        ):
            raise WorkflowValidationError(
                f"{target} differs from its cache manifest record"
            )
        recorded_paths.add(relative.as_posix())
        tree_digest.update(relative.as_posix().encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(actual_sha256.encode("ascii"))
        tree_digest.update(b"\n")
    discovered_paths = {
        path.relative_to(policy_root).as_posix()
        for path in policy_root.rglob("bs1.json")
    }
    if recorded_paths != discovered_paths or tree_digest.hexdigest() != output_tree.get(
        "sha256"
    ):
        raise WorkflowValidationError(
            f"{policy_root} content tree differs from its cache manifest"
        )

    data_policy = manifest.get("identity", {}).get("data_policy", {})
    if (
        data_policy.get("original_trace_util_result_data_consumed") is not False
        or data_policy.get("preexisting_dvfs_lookup_cache_consumed") is not False
    ):
        raise WorkflowValidationError(
            f"{manifest_path} does not certify fresh-only cache generation"
        )
    return {
        "manifest": file_record(manifest_path),
        "identity_sha256": manifest.get("identity_sha256"),
        "coverage": coverage,
        "policy_root": str(policy_root.resolve()),
    }


def policy_command(
    *,
    policy: str,
    trace_path: Path,
    run_dir: Path,
    backend_cache: Path,
    lookup_cache: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "neusim.run_scripts.fleetsim_main",
        "--model=llama3-70b",
        "--request_pattern=trace",
        "--trace=Azure-Code",
        f"--trace_file={trace_path}",
        f"--configs_path={REPO_ROOT / 'configs'}",
        "--request_rate=1.0",
        f"--max_timestamp_hours={run_figure_05.PAPER_TRACE_MAX_HOURS:g}",
        f"--static_vpod_allocation={ALLOCATION_CONFIG}",
        f"--output_dir={run_dir}",
        f"--npusim_backend_cache_dir={backend_cache}",
        "--npusim_backend_cache_use_mmap=true",
        "--enable_dvfs_power_model=true",
        "--enable_dvfs=true",
        f"--dvfs_policy={policy}",
        "--dvfs_max_perf_degrad=1.0",
        "--dvfs_safeguard_window_minutes=5",
        "--dvfs_safeguard_violation_threshold=0.007",
        f"--dvfs_lookup_cache_dir={lookup_cache}",
        "--dvfs_require_cache_hit=true",
        f"--slo_json_path={SLO_CONFIG}",
        "--slo_multiplier=5x",
        "--tqdm=false",
    ]


def initial_policy_provenance(
    *,
    policy: str,
    command: Sequence[str],
    trace_record: Mapping[str, Any],
    source_sha256: str,
    cache_record: Mapping[str, Any],
    baseline_cache: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        "policy": policy,
        "paper_label": POLICY_LABELS[policy],
        "allocation": ALLOCATION_ID,
        "static_topology": {
            "prefill_vpods": 20,
            "decode_vpods": 8,
            "npu": "TPU v5p",
            "chips_per_vpod": 4,
            "batch_size": 1,
            "dp_tp_pp": "1/4/1",
        },
        "command": list(command),
        "service_dvfs": {
            "maximum_performance_degradation": 1.0,
            "safeguard_window_minutes": 5,
            "safeguard_violation_threshold": 0.007,
            "slo_multiplier": "5x",
            "strict_lookup_cache_hits": True,
        },
        "inputs": {
            "trace": dict(trace_record),
            "allocation": file_record(ALLOCATION_CONFIG),
            "slo_config": file_record(SLO_CONFIG),
            "lookup_cache": dict(cache_record),
            "baseline_backend_cache_seed": str(baseline_cache.resolve()),
            "original_trace_util_result_or_cache": None,
        },
        "software_source_tree_sha256": source_sha256,
        "outputs": {},
    }


def finalize_policy_provenance(run_dir: Path) -> None:
    """Validate a successful replay and atomically mark it resumable."""

    provenance_path = run_dir / "policy_run_provenance.json"
    provenance = read_json(provenance_path)
    request_record = validate_result_trace(run_dir / "request_trace.csv")
    static_path = run_dir / "static_vpod_stats.json"
    static = run_figure_05.validate_static_vpod_stats(static_path)
    stats_path = run_dir / "stats.json"
    stats = read_json(stats_path)
    lookup = stats.get("dvfs_lookup")
    if (
        not isinstance(lookup, dict)
        or lookup.get("misses") != 0
        or not isinstance(lookup.get("hits"), int)
        or lookup["hits"] <= 0
    ):
        raise WorkflowValidationError(
            f"{stats_path} does not certify strict zero-miss lookups: {lookup}"
        )
    if not isinstance(lookup.get("plans_applied"), int) or lookup["plans_applied"] <= 0:
        raise WorkflowValidationError(
            f"{stats_path} reports no applied DVFS plans: {lookup}"
        )
    provenance["status"] = "complete"
    provenance["completed_utc"] = utc_now()
    provenance.pop("error", None)
    provenance["outputs"] = {
        "request_trace": request_record,
        "static_vpods": {**file_record(static_path), **static},
        "stats": file_record(stats_path),
        "simulation_log": file_record(run_dir / "simulation.log"),
        "dvfs_lookup": lookup,
    }
    write_json(provenance_path, provenance)


def mark_policy_failed(run_dir: Path, error: BaseException) -> None:
    """Record an interrupted/failed replay without touching successful peers."""

    provenance_path = run_dir / "policy_run_provenance.json"
    provenance = read_json(provenance_path)
    provenance["status"] = "failed"
    provenance["completed_utc"] = utc_now()
    provenance["error"] = f"{type(error).__name__}: {error}"
    write_json(provenance_path, provenance)


def validate_policy_run(
    *,
    run_dir: Path,
    policy: str,
    expected_command: Sequence[str],
    trace_record: Mapping[str, Any],
    source_sha256: str,
    cache_record: Mapping[str, Any],
) -> dict[str, Any]:
    provenance_path = run_dir / "policy_run_provenance.json"
    provenance = read_json(provenance_path)
    expected = {
        "status": "complete",
        "policy": policy,
        "allocation": ALLOCATION_ID,
        "command": list(expected_command),
        "software_source_tree_sha256": source_sha256,
    }
    mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatches:
        raise WorkflowValidationError(
            f"{provenance_path} does not match this run: {mismatches}"
        )
    inputs = provenance.get("inputs", {})
    if inputs.get("trace", {}).get("sha256") != trace_record["sha256"]:
        raise WorkflowValidationError(f"{provenance_path} trace mismatch")
    if (
        inputs.get("lookup_cache", {}).get("manifest", {}).get("sha256")
        != cache_record["manifest"]["sha256"]
    ):
        raise WorkflowValidationError(
            f"{provenance_path} lookup-cache manifest mismatch"
        )

    request_record = validate_result_trace(run_dir / "request_trace.csv")
    recorded_request = provenance.get("outputs", {}).get("request_trace", {})
    if recorded_request.get("sha256") != request_record["sha256"]:
        raise WorkflowValidationError(f"{provenance_path} request trace hash mismatch")
    static_path = run_dir / "static_vpod_stats.json"
    static = run_figure_05.validate_static_vpod_stats(static_path)
    stats_path = run_dir / "stats.json"
    stats = read_json(stats_path)
    lookup = stats.get("dvfs_lookup")
    if not isinstance(lookup, dict):
        raise WorkflowValidationError(f"{stats_path} has no DVFS lookup stats")
    if lookup.get("misses") != 0 or not isinstance(lookup.get("hits"), int):
        raise WorkflowValidationError(
            f"{stats_path} does not certify zero lookup misses: {lookup}"
        )
    if lookup["hits"] <= 0:
        raise WorkflowValidationError(f"{stats_path} reports no DVFS lookup hits")
    if not isinstance(lookup.get("plans_applied"), int) or lookup["plans_applied"] <= 0:
        raise WorkflowValidationError(
            f"{stats_path} reports no applied DVFS plans: {lookup}"
        )
    for output_name, output_path in (
        ("static_vpods", static_path),
        ("stats", stats_path),
        ("simulation_log", run_dir / "simulation.log"),
    ):
        recorded = provenance.get("outputs", {}).get(output_name, {})
        _require_record_matches(output_path, recorded, output_name)
    return {
        "provenance": file_record(provenance_path),
        "request_trace": request_record,
        "static_vpods": {**file_record(static_path), **static},
        "stats": file_record(stats_path),
        "dvfs_lookup": lookup,
        "simulation_log": file_record(run_dir / "simulation.log"),
    }


def validate_figure22_bundle(
    figure22_root: Path, policy_records: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    provenance_path = figure22_root / "figure_22_provenance.json"
    provenance = read_json(provenance_path)
    if provenance.get("status") != "complete":
        raise WorkflowValidationError(f"{provenance_path} is not complete")
    if provenance.get("allocation", {}).get("name") != ALLOCATION_ID:
        raise WorkflowValidationError(
            f"{provenance_path} does not match the paper static input"
        )
    inputs = provenance.get("inputs", {}).get("request_traces", {})
    expected_hashes = {
        "baseline": policy_records["baseline"]["request_trace"]["sha256"],
        "DVFSC": policy_records["DVFSC"]["request_trace"]["sha256"],
        "CustomAll": policy_records["CustomAll"]["request_trace"]["sha256"],
    }
    if len(set(expected_hashes.values())) != len(expected_hashes):
        duplicates = {name: digest for name, digest in expected_hashes.items()}
        raise WorkflowValidationError(
            f"Figure 22 policy traces are byte-identical: {duplicates}"
        )
    for name, expected_hash in expected_hashes.items():
        if inputs.get(name, {}).get("sha256") != expected_hash:
            raise WorkflowValidationError(
                f"{provenance_path} {name} trace hash mismatch"
            )
    output_records: dict[str, Any] = {}
    for name, record in provenance.get("outputs", {}).items():
        if not isinstance(record, dict) or "path" not in record:
            raise WorkflowValidationError(
                f"{provenance_path} has invalid output record {name}"
            )
        output_records[name] = _require_record_matches(
            Path(record["path"]), record, f"Figure 22 {name}"
        )
    report_path = figure22_root / "FIGURE_22_REVIEW.md"
    if not report_path.is_file():
        raise WorkflowValidationError(f"missing {report_path}")
    return {
        "provenance": file_record(provenance_path),
        "report": file_record(report_path),
        "outputs": output_records,
    }


def relative_link(target: Path, report: Path) -> str:
    return Path(os.path.relpath(target, report.parent)).as_posix()


def render_root_review(
    output_root: Path,
    workflow: Mapping[str, Any],
    figure05: Mapping[str, Any],
    figure22: Mapping[str, Any],
    policy_records: Mapping[str, Mapping[str, Any]],
) -> Path:
    report = output_root / "FIGURE_REVIEW.md"
    figure05_pdf = Path(figure05["figure"]["path"])
    figure22_pdf = Path(figure22["outputs"]["pdf"]["path"])
    lines = [
        "# MICRO'26 Figures 5 and 22 review",
        "",
        "- Status: **COMPLETE**",
        "- Static allocation: 20 prefill vPods and 8 decode vPods",
        "- Per-vPod topology: TPU v5p, 4 chips, BS1, DP/TP/PP = 1/4/1",
        "- Exact Azure one-day requests completed by every replay: "
        f"**{EXPECTED_REQUEST_ROWS:,}**",
        "- Figure 22 lookup cache mode: "
        f"**{workflow['inputs']['lookup_cache']['mode']}**",
        "- Original `trace_util` request results and lookup caches consumed: "
        "**none**",
        "",
        "## Figures",
        "",
        f"- Figure 5 PDF: [{figure05_pdf.name}]"
        f"({relative_link(figure05_pdf, report)})",
        f"- Figure 5 review: [FIGURE_05_REVIEW.md]"
        f"({relative_link(Path(figure05['report']['path']), report)})",
        f"- Figure 5 plotter: [figure_05.py]"
        f"({relative_link(AE_ROOT / 'plots' / 'figure_05.py', report)})",
        f"- Figure 22 PDF: [{figure22_pdf.name}]"
        f"({relative_link(figure22_pdf, report)})",
        f"- Figure 22 review: [FIGURE_22_REVIEW.md]"
        f"({relative_link(Path(figure22['report']['path']), report)})",
        f"- Figure 22 plotter: [figure_22.py]"
        f"({relative_link(AE_ROOT / 'plots' / 'figure_22.py', report)})",
        "",
        "## Raw evidence and cache provenance",
        "",
        f"- NoDVFS request trace: [request_trace.csv]"
        f"({relative_link(Path(figure05['request_trace']['path']), report)})",
        f"- DVFS-C request trace: [request_trace.csv]"
        f"({relative_link(Path(policy_records['DVFSC']['request_trace']['path']), report)})",
        f"- eNPU-All request trace: [request_trace.csv]"
        f"({relative_link(Path(policy_records['CustomAll']['request_trace']['path']), report)})",
        f"- DVFS-C cache manifest: [manifest.json]"
        f"({relative_link(Path(workflow['caches']['DVFSC']['manifest']['path']), report)})",
        f"- eNPU-All cache manifest: [manifest.json]"
        f"({relative_link(Path(workflow['caches']['CustomAll']['manifest']['path']), report)})",
        f"- Workflow provenance: [workflow_provenance.json]"
        f"({relative_link(output_root / 'workflow_provenance.json', report)})",
        "",
        "Both policy replays required strict cache hits. Their final FleetSim "
        "statistics report zero lookup misses. All expected request-trace "
        "columns were present, numeric, finite, and nonempty for every one of "
        "the 2,490,144 completed requests.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=DEFAULT_TRACE_FILE,
        help="Exact unsampled Azure one-day trace (default: packaged artifact input)",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--lookup-cache-dir",
        type=Path,
        default=DEFAULT_LOOKUP_CACHE_DIR,
        help="Supplied DVFS lookup root containing DVFSC/ and CustomAll/",
    )
    cache_group.add_argument(
        "--regenerate-lookup-cache",
        action="store_true",
        help=(
            "Ignore the supplied cache and regenerate both policies in the "
            "output root"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Fresh/restartable combined result root",
    )
    parser.add_argument(
        "--dvfsc-cache-workers",
        type=int,
        default=DEFAULT_DVFSC_CACHE_WORKERS,
        help=(
            "Bounded DVFS-C cache processes "
            f"(default: {DEFAULT_DVFSC_CACHE_WORKERS})"
        ),
    )
    parser.add_argument(
        "--customall-cache-workers",
        type=int,
        default=DEFAULT_CUSTOMALL_CACHE_WORKERS,
        help=(
            "Bounded eNPU-All cache processes "
            f"(default: {DEFAULT_CUSTOMALL_CACHE_WORKERS})"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and reuse complete hash-bound stages in a nonempty root",
    )
    args = parser.parse_args(argv)
    for option in ("dvfsc_cache_workers", "customall_cache_workers"):
        if not 1 <= getattr(args, option) <= 32:
            parser.error(f"--{option.replace('_', '-')} must be in [1, 32]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        trace_path = args.trace_file.expanduser().resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"supplied Azure trace is not extracted at {args.trace_file}; "
            "run the supplementary download and extraction commands in "
            "micro26ae.md Installation"
        ) from error
    trace_record = validate_exact_trace(trace_path)
    output_root = prepare_output_root(args.output_dir, resume=args.resume)
    logs_root = output_root / "logs"
    figure05_root = output_root / "figure05"
    figure22_root = output_root / "figure22"
    if args.regenerate_lookup_cache:
        cache_mode = "regenerated"
        cache_root = output_root / "dvfs_lookup"
    else:
        cache_mode = "supplied"
        try:
            cache_root = args.lookup_cache_dir.expanduser().resolve(strict=True)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"supplied lookup cache is not extracted at {args.lookup_cache_dir}; "
                "run the supplementary download and extraction commands in "
                "micro26ae.md Installation"
            ) from error
        if not cache_root.is_dir():
            raise NotADirectoryError(
                f"--lookup-cache-dir is not a directory: {cache_root}"
            )
    source_record = run_figure_05.source_tree_record()
    workflow_path = output_root / "workflow_provenance.json"
    cache_input = {"mode": cache_mode, "path": str(cache_root)}
    if workflow_path.exists():
        if not args.resume:
            raise WorkflowValidationError(
                f"{workflow_path} exists but --resume was not supplied"
            )
        workflow = read_json(workflow_path)
        if (
            workflow.get("inputs", {}).get("trace", {}).get("sha256")
            != trace_record["sha256"]
            or workflow.get("software_source_tree_sha256") != source_record["sha256"]
            or workflow.get("inputs", {}).get("lookup_cache") != cache_input
        ):
            raise WorkflowValidationError(
                "existing workflow provenance does not match the exact trace, "
                "lookup cache, and current simulator; use a fresh output directory"
            )
    else:
        workflow = {
            "schema_version": 1,
            "status": "running",
            "started_utc": utc_now(),
            "completed_utc": None,
            "figures": [5, 22],
            "current_static_allocation": ALLOCATION_ID,
            "parallelism": {
                "wave1": (
                    "one single-process NoDVFS replay"
                    if cache_mode == "supplied"
                    else "one NoDVFS replay plus two policy-cache generators"
                ),
                "lookup_cache_mode": cache_mode,
                "dvfsc_cache_workers": args.dvfsc_cache_workers,
                "customall_cache_workers": args.customall_cache_workers,
                "wave2": "two independent single-process FleetSim replays",
                "numeric_library_threads_per_process": 1,
                "worker_allocation_basis": "72/28 measured-cost split",
            },
            "inputs": {
                "trace": trace_record,
                "lookup_cache": cache_input,
                "allocation": file_record(ALLOCATION_CONFIG),
                "slo_config": file_record(SLO_CONFIG),
                "original_trace_util_result_or_cache": None,
            },
            "software_source_tree_sha256": source_record["sha256"],
            "git": git_record(),
            "stages": {},
            "caches": {},
            "policy_runs": {},
            "outputs": {},
        }
        write_json(workflow_path, workflow)

    environment = common_environment()

    # Wave 1: fresh NoDVFS replay and both independent lookup-cache policies.
    wave1: list[tuple[str, list[str], Path]] = []
    try:
        figure05_record = validate_figure05_bundle(
            figure05_root, trace_record, source_record["sha256"]
        )
        print("[resume:figure05] validated complete static-fleet bundle", flush=True)
    except (FileNotFoundError, WorkflowValidationError) as error:
        if figure05_root.exists() and any(figure05_root.iterdir()):
            raise WorkflowValidationError(
                f"{figure05_root} is nonempty but not safely resumable; "
                "use a fresh output root"
            ) from error
        wave1.append(
            (
                "figure05",
                [
                    sys.executable,
                    str(FIGURE05_RUNNER),
                    f"--trace-file={trace_path}",
                    f"--output-dir={figure05_root}",
                ],
                logs_root / "figure05_driver.log",
            )
        )

    cache_workers = {
        "DVFSC": args.dvfsc_cache_workers,
        "CustomAll": args.customall_cache_workers,
    }
    for policy in POLICIES:
        try:
            validate_cache_manifest(cache_root, policy, trace_record)
            print(
                f"[resume:cache_{policy}] validated complete {cache_mode} cache",
                flush=True,
            )
        except (FileNotFoundError, WorkflowValidationError) as error:
            if cache_mode == "supplied":
                raise WorkflowValidationError(
                    f"supplied {policy} cache failed validation: {cache_root / policy}"
                ) from error
            wave1.append(
                (
                    f"cache_{policy}",
                    [
                        sys.executable,
                        str(CACHE_GENERATOR),
                        f"--trace={trace_path}",
                        f"--output-dir={cache_root}",
                        f"--policy={policy}",
                        f"--workers={cache_workers[policy]}",
                    ],
                    logs_root / f"cache_{policy}.log",
                )
            )
    launch_commands(wave1, environment)

    figure05_record = validate_figure05_bundle(
        figure05_root, trace_record, source_record["sha256"]
    )
    cache_records = {
        policy: validate_cache_manifest(cache_root, policy, trace_record)
        for policy in POLICIES
    }
    workflow["stages"]["wave1"] = {
        "status": "complete",
        "completed_utc": utc_now(),
        "figure05": figure05_record,
    }
    workflow["caches"] = cache_records
    write_json(workflow_path, workflow)

    # Wave 2: seed isolated backend caches from the fresh NoDVFS cache, then
    # run both policy replays concurrently. Isolated copies avoid concurrent
    # joblib writers while retaining the reusable baseline analyses.
    baseline_cache = Path(figure05_record["backend_cache"])
    policy_commands: dict[str, list[str]] = {}
    wave2: list[tuple[str, list[str], Path]] = []
    policy_records: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        run_dir = figure22_root / "runs" / policy
        backend_cache = run_dir / ".cache" / "npusim_backend"
        command = policy_command(
            policy=policy,
            trace_path=trace_path,
            run_dir=run_dir,
            backend_cache=backend_cache,
            lookup_cache=cache_root / policy,
        )
        policy_commands[policy] = command
        try:
            policy_records[policy] = validate_policy_run(
                run_dir=run_dir,
                policy=policy,
                expected_command=command,
                trace_record=trace_record,
                source_sha256=source_record["sha256"],
                cache_record=cache_records[policy],
            )
            print(f"[resume:{policy}] validated complete replay", flush=True)
            continue
        except (FileNotFoundError, WorkflowValidationError) as error:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise WorkflowValidationError(
                    f"{run_dir} is nonempty but not safely resumable; "
                    "use a fresh output root"
                ) from error
        run_dir.mkdir(parents=True)
        shutil.copytree(baseline_cache, backend_cache)
        policy_provenance = initial_policy_provenance(
            policy=policy,
            command=command,
            trace_record=trace_record,
            source_sha256=source_record["sha256"],
            cache_record=cache_records[policy],
            baseline_cache=baseline_cache,
        )
        write_json(run_dir / "policy_run_provenance.json", policy_provenance)
        wave2.append((policy, command, run_dir / "simulation.log"))

    try:
        launch_commands(wave2, environment)
    except ParallelStageError as error:
        for policy, _command, _log in wave2:
            run_dir = figure22_root / "runs" / policy
            if error.returncodes.get(policy) == 0:
                finalize_policy_provenance(run_dir)
            else:
                mark_policy_failed(run_dir, error)
        raise
    except BaseException as error:
        for policy, _command, _log in wave2:
            mark_policy_failed(figure22_root / "runs" / policy, error)
        raise

    for policy, _command, _log in wave2:
        finalize_policy_provenance(figure22_root / "runs" / policy)

    for policy in POLICIES:
        policy_records[policy] = validate_policy_run(
            run_dir=figure22_root / "runs" / policy,
            policy=policy,
            expected_command=policy_commands[policy],
            trace_record=trace_record,
            source_sha256=source_record["sha256"],
            cache_record=cache_records[policy],
        )
    workflow["stages"]["wave2"] = {
        "status": "complete",
        "completed_utc": utc_now(),
    }
    workflow["policy_runs"] = policy_records
    write_json(workflow_path, workflow)

    # NoDVFS is intentionally the one shared fresh Figure 5 replay.
    figure22_policy_records: dict[str, Mapping[str, Any]] = {
        "baseline": figure05_record,
        **policy_records,
    }
    try:
        figure22_record = validate_figure22_bundle(
            figure22_root, figure22_policy_records
        )
        print("[resume:figure22] validated plots and provenance", flush=True)
    except (FileNotFoundError, WorkflowValidationError):
        build_command = [
            sys.executable,
            str(FIGURE22_BUILDER),
            f"--baseline-trace={figure05_record['request_trace']['path']}",
            f"--dvfsc-trace={policy_records['DVFSC']['request_trace']['path']}",
            f"--enpu-all-trace={policy_records['CustomAll']['request_trace']['path']}",
            f"--slo-config={SLO_CONFIG}",
            f"--output-dir={figure22_root}",
        ]
        launch_commands(
            [("plot_figure22", build_command, logs_root / "figure22_plot.log")],
            environment,
        )
        figure22_record = validate_figure22_bundle(
            figure22_root, figure22_policy_records
        )

    workflow["stages"]["plot_and_review"] = {
        "status": "complete",
        "completed_utc": utc_now(),
    }
    workflow["outputs"] = {
        "figure05": figure05_record,
        "figure22": figure22_record,
    }
    workflow["status"] = "complete"
    workflow["completed_utc"] = utc_now()
    write_json(workflow_path, workflow)
    report = render_root_review(
        output_root,
        workflow,
        figure05_record,
        figure22_record,
        policy_records,
    )
    workflow["outputs"]["root_review"] = file_record(report)
    write_json(workflow_path, workflow)
    print(f"Combined review: {report}", flush=True)
    print(f"Figure 5 PDF: {figure05_record['figure']['path']}", flush=True)
    print(
        f"Figure 22 PDF: {figure22_record['outputs']['pdf']['path']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
