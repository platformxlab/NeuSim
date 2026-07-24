#!/usr/bin/env python3
"""Orchestrate the MICRO 2026 eNPU artifact experiments and plots.

The pipeline keeps experiment discovery, native NeuSim execution, plotting,
validation, and reviewer-report generation separate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PIPELINE_MANIFEST = HERE / "config" / "pipeline.json"
PAPER_MANIFEST = HERE / "config" / "paper_experiments.json"
DEFAULT_BUNDLE = HERE / "reproduced" / "chip"
STAGES = ("prepare", "experiments", "plot", "validate", "review")
GROUP_ORDER = (
    "standard_sweep",
    "domain_count",
    "temporal_granularity",
    "fixed_sequence_sweep",
    "expert_imbalance",
    "power_gating",
)
PARALLEL_TRACE_GROUPS = frozenset(
    {
        "standard_sweep",
        "domain_count",
        "temporal_granularity",
        "fixed_sequence_sweep",
        "power_gating",
    }
)
SERIAL_TRACE_GROUPS = frozenset(set(GROUP_ORDER) - PARALLEL_TRACE_GROUPS)


class PipelineError(RuntimeError):
    """A user-actionable artifact-pipeline error."""


def _parse_group_trace_worker_assignment(value: str) -> tuple[str, int]:
    """Parse and validate one repeatable ``GROUP=N`` worker override."""
    if value.count("=") != 1:
        raise argparse.ArgumentTypeError(
            "expected GROUP=N (for example fixed_sequence_sweep=24)"
        )
    raw_group, raw_workers = value.split("=", 1)
    group = raw_group.strip()
    workers_text = raw_workers.strip()
    if group not in GROUP_ORDER:
        raise argparse.ArgumentTypeError(
            f"unknown experiment group {group!r}; choose from: "
            + ", ".join(GROUP_ORDER)
        )
    if not workers_text.isdecimal() or int(workers_text) < 1:
        raise argparse.ArgumentTypeError(
            f"worker count for {group} must be an integer of at least 1"
        )
    workers = int(workers_text)
    if group in SERIAL_TRACE_GROUPS and workers != 1:
        raise argparse.ArgumentTypeError(
            f"{group} is intentionally isolated and only accepts {group}=1"
        )
    return group, workers


class _UniqueGroupTraceWorkers(argparse.Action):
    """Collect worker overrides while rejecting ambiguous duplicates."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: tuple[str, int],
        option_string: str | None = None,
    ) -> None:
        del option_string
        assignments = tuple(getattr(namespace, self.dest, ()) or ())
        group, _ = values
        if any(existing_group == group for existing_group, _ in assignments):
            parser.error(f"--group-trace-workers repeats group {group!r}")
        setattr(namespace, self.dest, (*assignments, values))


@dataclass(frozen=True)
class Command:
    label: str
    argv: tuple[str, ...]
    cwd: Path
    env: dict[str, str]


@dataclass(frozen=True)
class Context:
    config: dict[str, Any]
    paper: dict[str, Any]
    selected: tuple[str, ...]
    stages: tuple[str, ...]
    results_dir: Path
    output_dir: Path
    paper_pdf: Path | None
    python: str
    jobs: int
    trace_workers: int
    group_trace_worker_overrides: tuple[tuple[str, int], ...]
    allow_current_ideal: bool
    verbose_simulator: bool
    quick: bool
    dry_run: bool
    resume: bool


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise PipelineError(f"required manifest is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PipelineError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError(f"manifest root must be an object: {path}")
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@cache
def _path_fingerprint(encoded_path: str) -> str:
    """Hash one file or a directory tree for safe resume decisions."""
    path = Path(encoded_path)
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(b"file\0")
        digest.update(_sha256_file(path).encode())
        return digest.hexdigest()
    digest.update(b"directory\0")
    for candidate in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = candidate.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(_sha256_file(candidate).encode())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _code_fingerprint() -> str:
    """Hash simulator/artifact sources that can change normalized results."""
    suffixes = {".py", ".json", ".sh"}
    roots = (
        HERE / "config",
        HERE / "experiments",
        HERE / "plots",
        REPO_ROOT / "neusim",
        REPO_ROOT / "configs",
    )
    files = [
        path for path in HERE.iterdir() if path.is_file() and path.suffix in suffixes
    ]
    for root in roots:
        if not root.is_dir():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in suffixes
            and "__pycache__" not in path.parts
        )
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _resume_signature(
    context: Context,
    *,
    commands: Sequence[Command],
    inputs: Sequence[Path] = (),
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "code_sha256": _code_fingerprint(),
        "pipeline_manifest_sha256": _sha256_file(PIPELINE_MANIFEST),
        "paper_manifest_sha256": _sha256_file(PAPER_MANIFEST),
        "paper_pdf_sha256": (
            _sha256_file(context.paper_pdf)
            if context.paper_pdf is not None and context.paper_pdf.is_file()
            else None
        ),
        "quick": context.quick,
        "allow_current_ideal": context.allow_current_ideal,
        "jobs": context.jobs,
        # The resolved cap is encoded only in this group's command. Do not put
        # the full override mapping here: unrelated overrides must not stale it.
        "dvfs_ms_candidate_batch_size": os.environ.get("DVFS_MS_CANDIDATE_BATCH_SIZE"),
        "commands": [_command_text(command) for command in commands],
        "inputs": {
            str(path.resolve()): _path_fingerprint(str(path.resolve()))
            for path in inputs
        },
    }


def parse_selection(specification: str, valid: set[str]) -> tuple[str, ...]:
    """Parse comma-separated figure numbers and inclusive ranges."""
    selected: set[str] = set()
    for raw_token in specification.split(","):
        token = raw_token.strip().lower().replace("figure", "")
        if not token:
            continue
        if token == "all":
            selected.update(valid)
            continue
        if "-" in token:
            pieces = token.split("-", 1)
            if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
                raise PipelineError(f"invalid figure range {raw_token!r}")
            first, last = (int(piece) for piece in pieces)
            if last < first:
                raise PipelineError(
                    f"descending figure range {raw_token!r} is not supported"
                )
            selected.update(str(number) for number in range(first, last + 1))
            continue
        if not token.isdigit():
            raise PipelineError(f"invalid figure selector {raw_token!r}")
        selected.add(str(int(token)))
    if not selected:
        raise PipelineError("--figures selected no figures")
    unknown = selected - valid
    if unknown:
        raise PipelineError(
            "figures not in the MICRO26 artifact manifest: "
            + ", ".join(sorted(unknown))
        )
    return tuple(
        sorted(
            selected,
            key=int,
        )
    )


def _selected_groups(context: Context) -> tuple[str, ...]:
    requested: set[str] = set()
    for key in context.selected:
        requested.update(context.config["figures"][key].get("experiment_groups", []))
    return tuple(group for group in GROUP_ORDER if group in requested)


def _group_trace_worker_overrides(context: Context) -> dict[str, int]:
    """Return a validated mapping for programmatic as well as CLI contexts."""
    overrides: dict[str, int] = {}
    for group, workers in context.group_trace_worker_overrides:
        if group not in GROUP_ORDER:
            raise PipelineError(f"unknown trace-worker override group: {group}")
        if group in overrides:
            raise PipelineError(f"duplicate trace-worker override group: {group}")
        if workers < 1:
            raise PipelineError(f"trace-worker override for {group} must be at least 1")
        if group in SERIAL_TRACE_GROUPS and workers != 1:
            raise PipelineError(
                f"{group} is intentionally isolated and requires a worker cap of 1"
            )
        overrides[group] = workers
    return overrides


def _requested_group_trace_worker_cap(context: Context, group: str) -> int:
    if group not in GROUP_ORDER:
        raise PipelineError(f"unknown experiment group: {group}")
    overrides = _group_trace_worker_overrides(context)
    if group in SERIAL_TRACE_GROUPS:
        return overrides.get(group, 1)
    return overrides.get(group, context.trace_workers)


def _resolved_group_trace_worker_cap(context: Context, group: str) -> int:
    """Resolve one group's configured cap after the global Ray-slot limit."""
    requested = _requested_group_trace_worker_cap(context, group)
    return min(requested, context.jobs) if group in PARALLEL_TRACE_GROUPS else 1


def _group_trace_task_count(context: Context, group: str) -> int:
    """Mirror the native runner's deterministic outer-task cardinality."""
    if group in {
        "standard_sweep",
        "domain_count",
        "temporal_granularity",
    }:
        model_count = 1 if context.quick else len(context.paper["models"])
        return model_count * 2
    if group == "fixed_sequence_sweep":
        sequence = context.paper["sequence_lengths_tokens"]
        lengths = [int(value) for value in sequence["input_sweep"]]
        if context.quick:
            lengths = [lengths[0], int(sequence["default_input"])]
        return 4 * len(lengths)
    if group == "expert_imbalance":
        return 2 if context.quick else len(context.paper["expert_capacity_factors"])
    if group == "power_gating":
        return 2
    return 1


def _effective_group_trace_workers(context: Context, group: str) -> int:
    if context.quick or group not in PARALLEL_TRACE_GROUPS:
        return 1
    return min(
        _resolved_group_trace_worker_cap(context, group),
        _group_trace_task_count(context, group),
    )


def _trace_worker_resolution(context: Context) -> dict[str, dict[str, Any]]:
    """Describe requested, configured, and effective caps for every known group."""
    overrides = _group_trace_worker_overrides(context)
    return {
        group: {
            "source": "group_override" if group in overrides else "default",
            "requested_worker_cap": _requested_group_trace_worker_cap(context, group),
            "configured_worker_cap": _resolved_group_trace_worker_cap(context, group),
            "effective_workers": _effective_group_trace_workers(context, group),
            "task_count": _group_trace_task_count(context, group),
            "parallel_safe": group in PARALLEL_TRACE_GROUPS,
        }
        for group in GROUP_ORDER
    }


def _temporal_standard_reuse_input(context: Context, group: str) -> Path | None:
    if group == "temporal_granularity" and "standard_sweep" in _selected_groups(
        context
    ):
        return context.results_dir / "raw" / "standard_sweep" / "energy_records.json"
    return None


def build_group_commands(context: Context, group: str) -> list[Command]:
    output = context.results_dir / "raw" / group
    args = [
        context.python,
        str(HERE / "experiments" / "run_native.py"),
        "--group",
        group,
        "--output-dir",
        str(output),
        "--repo-root",
        str(REPO_ROOT),
        "--paper-manifest",
        str(PAPER_MANIFEST),
        "--pipeline-manifest",
        str(PIPELINE_MANIFEST),
        "--jobs",
        str(context.jobs),
        "--trace-workers",
        str(_resolved_group_trace_worker_cap(context, group)),
    ]
    if context.paper_pdf is not None:
        args.extend(["--paper-pdf", str(context.paper_pdf)])
    if context.quick:
        args.append("--quick")
    if context.resume:
        args.append("--resume")
    if context.allow_current_ideal:
        args.append("--allow-current-ideal")
    if context.verbose_simulator:
        args.append("--verbose-simulator")
    reuse_input = _temporal_standard_reuse_input(context, group)
    if reuse_input is not None:
        args.extend(["--standard-sweep-energy-records", str(reuse_input)])
    return [Command(f"native NeuSim {group}", tuple(args), REPO_ROOT, {})]


def preflight(context: Context) -> None:
    problems: list[str] = []
    if not PIPELINE_MANIFEST.is_file():
        problems.append(f"pipeline manifest not found: {PIPELINE_MANIFEST}")
    if not PAPER_MANIFEST.is_file():
        problems.append(f"paper experiment manifest not found: {PAPER_MANIFEST}")
    if context.paper_pdf is not None and not context.paper_pdf.is_file():
        problems.append(f"paper PDF does not exist: {context.paper_pdf}")

    if "experiments" in context.stages:
        groups = _selected_groups(context)
        driver = HERE / "experiments" / "run_native.py"
        if not driver.is_file():
            problems.append(f"native experiment driver is missing: {driver}")
        ideal_groups = {"standard_sweep", "temporal_granularity"} & set(groups)
        if ideal_groups and not context.quick and not context.allow_current_ideal:
            problems.append(
                "full native matrices include the potentially expensive current "
                "Ideal request search. After per-operator budget filtering, the "
                "backend reduces raw candidate products above 2,000,000 to "
                "per-voltage frequency extrema and, when still over the limit, a "
                "deterministic balanced endpoint-preserving sample; enumeration is "
                "lazy and it does not enumerate the complete "
                "48,147,400-state theoretical lattice. Rerun with "
                "--allow-current-ideal to opt in, or use --quick only for a visibly "
                "labeled operator-local smoke test"
            )
        for package in ("numpy", "pandas"):
            if importlib.util.find_spec(package) is None:
                problems.append(
                    f"Python package {package!r} is missing from {sys.executable}; install NeuSim dependencies"
                )

    if "plot" in context.stages:
        for key in context.selected:
            item = context.config["figures"][key]
            script = HERE / "plots" / item["plot_script"]
            if not script.is_file():
                problems.append(f"Figure {key} plot adapter is missing: {script}")
        if importlib.util.find_spec("matplotlib") is None:
            problems.append(
                f"Python package 'matplotlib' is missing from {sys.executable}; install NeuSim dependencies"
            )

        # A plot-only invocation must start from existing raw data. When the
        # experiments stage is also selected, those inputs are produced later.
        if "experiments" not in context.stages:
            for key in context.selected:
                item = context.config["figures"][key]
                relative = item.get("input")
                if relative and not (context.results_dir / relative).exists():
                    problems.append(
                        f"Figure {key} input is missing: {context.results_dir / relative}; "
                        "run the experiments stage or point --results-dir at collected data"
                    )

    if "validate" in context.stages and "plot" not in context.stages:
        for key in context.selected:
            output = context.output_dir / context.config["figures"][key]["output"]
            if not output.is_file():
                problems.append(f"expected output is missing: {output}")

    if problems:
        detail = "\n".join(f"  - {problem}" for problem in problems)
        raise PipelineError(f"preflight failed:\n{detail}")


def _command_text(command: Command) -> str:
    return shlex.join(command.argv)


def _nonempty_path(path: Path) -> bool:
    if path.is_file():
        return path.stat().st_size > 0
    if path.is_dir():
        return any(
            item.is_file() and item.stat().st_size > 0 for item in path.rglob("*")
        )
    return False


def _path_fingerprints(paths: Sequence[Path]) -> dict[str, str]:
    _path_fingerprint.cache_clear()
    return {
        str(path.resolve()): _path_fingerprint(str(path.resolve())) for path in paths
    }


def _group_expected_outputs(context: Context, group: str) -> tuple[Path, ...]:
    relative_paths: set[str] = {f"raw/{group}/provenance.json"}
    for key in context.selected:
        item = context.config["figures"][key]
        if group not in item.get("experiment_groups", []):
            continue
        for field in ("input", "reference_input"):
            relative = item.get(field)
            if isinstance(relative, str) and relative.startswith(f"raw/{group}/"):
                relative_paths.add(relative)
    return tuple(context.results_dir / relative for relative in sorted(relative_paths))


def _experiment_marker_matches(
    marker: Path,
    signature: dict[str, Any],
    expected_outputs: Sequence[Path],
) -> bool:
    if not marker.is_file() or any(
        not _nonempty_path(path) for path in expected_outputs
    ):
        return False
    try:
        with marker.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == 2
        and payload.get("status") == "complete"
        and payload.get("signature") == signature
        and payload.get("output_fingerprints") == _path_fingerprints(expected_outputs)
    )


def _run_command(context: Context, command: Command) -> None:
    print(f"[{command.label}] cwd={command.cwd}")
    print(f"  {_command_text(command)}")
    if context.dry_run:
        return
    command.cwd.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(command.env)
    try:
        subprocess.run(command.argv, cwd=command.cwd, env=environment, check=True)
    except FileNotFoundError as exc:
        raise PipelineError(
            f"command executable was not found: {command.argv[0]}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise PipelineError(
            f"{command.label} failed with exit status {exc.returncode}"
        ) from exc


def prepare(context: Context) -> None:
    directories = (
        context.results_dir / "raw",
        context.results_dir / ".micro26" / "complete",
        context.output_dir / "figures",
    )
    for directory in directories:
        print(f"[prepare] mkdir -p {directory}")
        if not context.dry_run:
            directory.mkdir(parents=True, exist_ok=True)

    run_manifest = {
        "schema_version": 1,
        "created_at": _now(),
        "repo_root": str(REPO_ROOT),
        "selected": list(context.selected),
        "stages": list(context.stages),
        "quick": context.quick,
        "results_dir": str(context.results_dir),
        "output_dir": str(context.output_dir),
        "paper_pdf": str(context.paper_pdf) if context.paper_pdf else None,
        "paper_manifest": str(PAPER_MANIFEST),
        "pipeline_manifest": str(PIPELINE_MANIFEST),
        "runner": "native",
        "jobs": context.jobs,
        "trace_workers": context.trace_workers,
        "group_trace_worker_overrides": _group_trace_worker_overrides(context),
        "trace_worker_resolution": _trace_worker_resolution(context),
        "allow_current_ideal": context.allow_current_ideal,
        "ideal_search_semantics": {
            "raw_candidate_reduction_threshold": 2_000_000,
            "unreduced_table_lattice_states": 48_147_400,
            "unreduced_table_lattice_is_theoretical": True,
            "full_theoretical_lattice_enumerated": False,
        },
        "verbose_simulator": context.verbose_simulator,
    }
    destination = context.results_dir / ".micro26" / "resolved_run.json"
    print(f"[prepare] write {destination}")
    if not context.dry_run:
        _atomic_json(destination, run_manifest)


def run_experiments(context: Context) -> None:
    groups = _selected_groups(context)
    if not groups:
        print("[experiments] no simulation is required for the selection")
        return
    for group in groups:
        mode = "quick" if context.quick else "full"
        marker = context.results_dir / ".micro26" / "complete" / f"{group}.{mode}.json"
        commands = build_group_commands(context, group)
        signature_inputs: tuple[Path, ...] = ()
        reuse_input = _temporal_standard_reuse_input(context, group)
        if reuse_input is not None:
            signature_inputs = (reuse_input,)
        signature = _resume_signature(
            context, commands=commands, inputs=signature_inputs
        )
        expected_outputs = _group_expected_outputs(context, group)
        if context.resume and _experiment_marker_matches(
            marker, signature, expected_outputs
        ):
            print(f"[experiments] resume: matching {group} marker ({marker})")
            continue
        if context.resume and marker.is_file():
            print(
                f"[experiments] resume: stale/incomplete {group} marker; rerunning ({marker})"
            )
        if not context.dry_run:
            _atomic_json(
                marker,
                {
                    "schema_version": 2,
                    "status": "in_progress",
                    "group": group,
                    "mode": mode,
                    "started_at": _now(),
                    "signature": signature,
                },
            )
        for command in commands:
            _run_command(context, command)
        if not context.dry_run:
            missing = [path for path in expected_outputs if not _nonempty_path(path)]
            if missing:
                raise PipelineError(
                    f"{group} did not produce non-empty expected output(s): "
                    + ", ".join(str(path) for path in missing)
                )
            _path_fingerprint.cache_clear()
            _atomic_json(
                marker,
                {
                    "schema_version": 2,
                    "status": "complete",
                    "group": group,
                    "mode": mode,
                    "completed_at": _now(),
                    "commands": [_command_text(command) for command in commands],
                    "signature": signature,
                    "outputs": [str(path.resolve()) for path in expected_outputs],
                    "output_fingerprints": _path_fingerprints(expected_outputs),
                },
            )


def _plot_command(context: Context, key: str) -> tuple[Command, Path]:
    item = context.config["figures"][key]
    output = context.output_dir / item["output"]
    source = context.results_dir / item["input"]
    script = HERE / "plots" / item["plot_script"]
    args = [
        context.python,
        str(script),
        "--input",
        str(source),
        "--output",
        str(output),
    ]
    args.extend(str(value) for value in item.get("plot_args", []))
    reference = item.get("reference_input")
    if reference:
        args.extend(["--reference-input", str(context.results_dir / reference)])
    if context.quick:
        args.append("--allow-incomplete")
    return Command(f"Figure {key}", tuple(args), HERE, {}), output


def _plot_completion_marker(context: Context, key: str) -> Path:
    mode = "quick" if context.quick else "full"
    return context.results_dir / ".micro26" / "complete" / f"plot-{key}.{mode}.json"


def _plot_marker_matches(
    context: Context,
    key: str,
    output: Path,
    signature: dict[str, Any],
) -> bool:
    marker = _plot_completion_marker(context, key)
    if not marker.is_file() or not output.is_file() or output.stat().st_size == 0:
        return False
    try:
        with marker.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    mode = "quick" if context.quick else "full"
    return (
        isinstance(payload, dict)
        and payload.get("schema_version") == 2
        and payload.get("status") == "complete"
        and payload.get("key") == key
        and payload.get("mode") == mode
        and payload.get("output") == str(output.resolve())
        and payload.get("signature") == signature
        and payload.get("output_sha256") == _sha256_file(output)
    )


def plot(context: Context) -> None:
    for key in context.selected:
        command, output = _plot_command(context, key)
        marker = _plot_completion_marker(context, key)
        item = context.config["figures"][key]
        inputs = []
        for field in ("input", "reference_input"):
            relative = item.get(field)
            if isinstance(relative, str):
                inputs.append(context.results_dir / relative)
        signature = _resume_signature(
            context, commands=(command,), inputs=tuple(inputs)
        )
        if context.resume and _plot_marker_matches(context, key, output, signature):
            print(f"[plot] resume: matching completion marker ({marker})")
            continue
        if not context.dry_run:
            output.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(
                marker,
                {
                    "schema_version": 2,
                    "status": "in_progress",
                    "key": key,
                    "mode": "quick" if context.quick else "full",
                    "started_at": _now(),
                    "signature": signature,
                },
            )
        _run_command(context, command)
        if not context.dry_run and (not output.is_file() or output.stat().st_size == 0):
            raise PipelineError(
                f"{command.label} did not produce a non-empty output: {output}"
            )
        if not context.dry_run:
            _atomic_json(
                marker,
                {
                    "schema_version": 2,
                    "status": "complete",
                    "key": key,
                    "mode": "quick" if context.quick else "full",
                    "output": str(output.resolve()),
                    "output_sha256": _sha256_file(output),
                    "completed_at": _now(),
                    "command": _command_text(command),
                    "signature": signature,
                },
            )


def _validate_pdf(path: Path) -> list[str]:
    errors = []
    if not path.is_file():
        return ["file does not exist"]
    if path.stat().st_size < 100:
        errors.append("file is unexpectedly small")
    with path.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            errors.append("file does not start with a PDF header")
    return errors


def validate(context: Context) -> None:
    if context.dry_run:
        for key in context.selected:
            path = context.output_dir / context.config["figures"][key]["output"]
            print(f"[validate] would check {key}: {path}")
        return
    report: dict[str, Any] = {
        "schema_version": 1,
        "validated_at": _now(),
        "outputs": {},
        "ok": True,
    }
    failures = []
    for key in context.selected:
        path = context.output_dir / context.config["figures"][key]["output"]
        errors = _validate_pdf(path)
        report["outputs"][key] = {
            "path": str(path),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "errors": errors,
        }
        if errors:
            failures.append(f"{key}: " + "; ".join(errors))
        else:
            print(f"[validate] OK {key}: {path}")
    report["ok"] = not failures
    destination = context.output_dir / "validation_report.json"
    if not context.dry_run:
        _atomic_json(destination, report)
    if failures:
        raise PipelineError(
            "output validation failed:\n"
            + "\n".join(f"  - {item}" for item in failures)
        )


def review(context: Context) -> None:
    """Write numerical and visual review notes for the selected figures."""
    destination = context.output_dir / "FIGURE_REVIEW.md"
    command = Command(
        "Numerical figure review",
        (
            context.python,
            str(HERE / "generate_review_report.py"),
            "--results-dir",
            str(context.results_dir),
            "--output-dir",
            str(context.output_dir),
            "--pipeline-manifest",
            str(PIPELINE_MANIFEST),
            "--paper-manifest",
            str(PAPER_MANIFEST),
            "--figures",
            ",".join(context.selected),
            "--output",
            str(destination),
            "--previews",
            "auto",
            *(("--quick",) if context.quick else ()),
            *(
                ("--paper-pdf", str(context.paper_pdf))
                if context.paper_pdf is not None
                else ()
            ),
        ),
        HERE,
        {},
    )
    _run_command(context, command)
    if not context.dry_run and (
        not destination.is_file() or destination.stat().st_size == 0
    ):
        raise PipelineError(
            f"review generator did not produce a non-empty report: {destination}"
        )


def list_registry(config: dict[str, Any]) -> None:
    for key in sorted(config["figures"], key=int):
        item = config["figures"][key]
        print(f"Figure {key:>2s} READY    {item['title']}")


def build_parser(config: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run, plot, validate, and review the MICRO26 eNPU artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ./run_all.sh --allow-current-ideal --resume\n"
            "  ./run_all.sh --figures 2,11 --quick "
            "--results-dir /tmp/neusim-micro26-quick "
            "--output-dir /tmp/neusim-micro26-quick\n"
            "  ./run_all.sh plot validate review"
        ),
    )
    parser.add_argument(
        "stages",
        nargs="*",
        metavar="STAGE",
        help="ordered subset of: " + ", ".join(STAGES) + "; omitted means all stages",
    )
    parser.add_argument(
        "--figures",
        default=config["default_selection"],
        help="comma-separated figures/ranges (for example 2,3,11-13,16-18,20-21)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke-test a reduced model/threshold matrix",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved commands without writing or executing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip work only with matching quick/full completion markers",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="self-contained experiment-data root (raw inputs and markers)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="figure/report root; defaults to --results-dir",
    )
    parser.add_argument(
        "--paper-pdf",
        type=Path,
        help="optional paper PDF to hash; a workspace-relative sibling is auto-detected",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(24, os.cpu_count() or 1)),
        help="Ray CPU slots for native optimizer fan-out",
    )
    parser.add_argument(
        "--trace-workers",
        type=int,
        default=None,
        help=(
            "isolated outer trace workers for race-free native groups; defaults "
            "to min(4, jobs)"
        ),
    )
    parser.add_argument(
        "--group-trace-workers",
        action=_UniqueGroupTraceWorkers,
        type=_parse_group_trace_worker_assignment,
        default=(),
        metavar="GROUP=N",
        help=(
            "repeatable per-group override; overrides --trace-workers for that "
            "group, rejects duplicates, and only permits =1 for isolated groups"
        ),
    )

    parser.add_argument(
        "--allow-current-ideal",
        action="store_true",
        help=(
            "permit the current bounded/reduced Ideal request search; this does not "
            "enumerate the complete 48.1M-state theoretical lattice"
        ),
    )
    parser.add_argument(
        "--verbose-simulator",
        action="store_true",
        help="show verbose NeuSim generator output",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list registered figures, then exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = _load_json(PIPELINE_MANIFEST)
        paper = _load_json(PAPER_MANIFEST)
        parser = build_parser(config)
        args = parser.parse_args(argv)
        if args.list:
            list_registry(config)
            return 0
        invalid_stages = [stage for stage in args.stages if stage not in STAGES]
        if invalid_stages:
            parser.error("unknown stages: " + ", ".join(invalid_stages))
        if len(set(args.stages)) != len(args.stages):
            parser.error("each stage may be specified at most once")
        if args.jobs < 1:
            parser.error("--jobs must be at least 1")
        selected = parse_selection(args.figures, set(config["figures"]))
        if args.trace_workers is not None and args.trace_workers < 1:
            parser.error("--trace-workers must be at least 1")
        trace_workers = min(args.trace_workers or 4, args.jobs)
        stages = tuple(args.stages) if args.stages else STAGES
        results_dir = args.results_dir.expanduser().resolve()
        output_dir = (args.output_dir or args.results_dir).expanduser().resolve()
        paper_pdf = args.paper_pdf.expanduser().resolve() if args.paper_pdf else None
        if paper_pdf is None:
            candidate = REPO_ROOT.parent / "eNPU_micro26ae" / "NPU_DVFS_paper.pdf"
            paper_pdf = candidate if candidate.is_file() else None
        context = Context(
            config=config,
            paper=paper,
            selected=selected,
            stages=stages,
            results_dir=results_dir,
            output_dir=output_dir,
            paper_pdf=paper_pdf,
            python=sys.executable,
            jobs=args.jobs,
            quick=args.quick,
            dry_run=args.dry_run,
            trace_workers=trace_workers,
            group_trace_worker_overrides=tuple(args.group_trace_workers),
            resume=args.resume,
            allow_current_ideal=args.allow_current_ideal,
            verbose_simulator=args.verbose_simulator,
        )

        print(f"MICRO26 selection: {', '.join(context.selected)}")
        print(f"Stages: {', '.join(context.stages)}")
        if context.quick:
            print("Mode: QUICK smoke test (not paper-scale reproduction)")
        if context.dry_run:
            print(
                "Mode: DRY RUN (no commands will execute and no files will be written)"
            )
        preflight(context)
        for stage in context.stages:
            print(f"\n=== {stage} ===")
            if stage == "prepare":
                prepare(context)
            elif stage == "experiments":
                run_experiments(context)
            elif stage == "plot":
                plot(context)
            elif stage == "validate":
                validate(context)
            elif stage == "review":
                review(context)
        print("\nMICRO26 pipeline completed successfully.")
        return 0
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
