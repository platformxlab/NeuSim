#!/usr/bin/env python3
"""Generate a fresh, provenance-bound DVFS lookup cache for Figure 22.

This driver intentionally does not consume a pre-existing lookup cache.  It
extracts every padded request shape from an explicit Azure trace, generates
current NeuSim operators for that shape, and invokes
``analyze_all_operator_energy`` independently for every paper budget.

The resulting policy directory is directly consumable by FleetSim's DVFS
scheduler::

    <output>/<policy>/llama3-70b/<input>_<output>/5p/<phase>/bs1.json

Each grouped ``bs1.json`` contains all nine audited paper budgets. A manifest binds
every resumable file to the trace, simulator source, configuration, runtime,
and exact generation semantics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neusim.configs.models.LLMConfig import LLMConfig  # noqa: E402
from neusim.configs.workloads.LLMInferenceWorkloadConfig import (  # noqa: E402
    LLMInferenceWorkloadConfig,
)
from neusim.fleetsim.util import pad_seqlen  # noqa: E402
from neusim.npusim.frontend.llm_ops_generator import LLMOpsGenerator  # noqa: E402
from neusim.npusim.frontend.power_analysis_lib import (  # noqa: E402
    analyze_all_operator_energy,
)

SCHEMA_VERSION = 2
PRODUCER = "NeuSim MICRO26 Figure 22 fresh DVFS cache generator"
MODEL = "llama3-70b"
VERSION = "5p"
POLICIES = ("DVFSC", "CustomAll")
PHASES = ("prefill", "decode")
BUDGETS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
TOPOLOGY = {
    "prefill_vpods": 20,
    "decode_vpods": 8,
    "num_chips": 4,
    "batch_size": 1,
    "dp": 1,
    "tp": 4,
    "pp": 1,
    "ep": 1,
}
ALLOCATION_CONFIG = (
    REPO_ROOT / "configs" / "fleetsim" / "figure_05_llama3_70b_tpuv5p_p20d8.json"
)
MAX_WORKERS = min(32, os.cpu_count() or 1)


class CacheValidationError(RuntimeError):
    """Raised when a cache or its provenance manifest is incomplete."""


@dataclass(frozen=True)
class PaddingSchedule:
    """The sequence padding schedule consumed by current FleetSim."""

    input_factors: tuple[int, ...]
    input_steps: tuple[int, ...]
    output_factors: tuple[int, ...]
    output_steps: tuple[int, ...]


@dataclass(frozen=True)
class CacheTask:
    """One independently generated padded-shape/phase grouped cache file."""

    repo_root: str
    policy_root: str
    policy: str
    input_seqlen: int
    output_seqlen: int
    phase: str
    manifest_identity_sha256: str
    budgets: tuple[float, ...] = BUDGETS


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _default_list(field_name: str) -> tuple[int, ...]:
    field = LLMInferenceWorkloadConfig.model_fields[field_name]
    if field.default_factory is None:
        raise RuntimeError(f"FleetSim field {field_name!r} has no default factory")
    return tuple(int(value) for value in field.default_factory())


def current_padding_schedule() -> PaddingSchedule:
    """Read padding values from FleetSim rather than maintaining a second copy."""

    return PaddingSchedule(
        input_factors=_default_list("input_seqlen_padding_factors"),
        input_steps=_default_list("input_seqlen_padding_steps"),
        output_factors=_default_list("output_seqlen_padding_factors"),
        output_steps=_default_list("output_seqlen_padding_steps"),
    )


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


def atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Durably replace one JSON file without exposing a partial document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise


def extract_padded_shapes(
    trace_path: Path,
    padding: PaddingSchedule | None = None,
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    """Extract every FleetSim-normalized padded shape from an Azure trace."""

    trace_path = trace_path.expanduser().resolve(strict=True)
    padding = padding or current_padding_schedule()
    required = ("TIMESTAMP", "ContextTokens", "GeneratedTokens")
    shapes: set[tuple[int, int]] = set()
    rows = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    min_input: int | None = None
    max_input: int | None = None
    min_output: int | None = None
    max_output: int | None = None

    with trace_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [name for name in required if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(
                f"{trace_path} is not an Azure inference trace; missing columns: "
                + ", ".join(missing)
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                input_tokens = int(row["ContextTokens"])
                generated_tokens = int(row["GeneratedTokens"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{trace_path}:{line_number} has non-integer token lengths"
                ) from error
            if input_tokens <= 0:
                raise ValueError(
                    f"{trace_path}:{line_number} has non-positive ContextTokens"
                )
            # This is the exact normalization in LoadGenerator's Azure path.
            output_tokens = max(2, generated_tokens)
            input_padded = pad_seqlen(
                input_tokens, padding.input_factors, padding.input_steps
            )
            output_padded = pad_seqlen(
                output_tokens, padding.output_factors, padding.output_steps
            )
            shapes.add((input_padded, output_padded))
            rows += 1
            timestamp = row["TIMESTAMP"].strip()
            first_timestamp = timestamp if first_timestamp is None else first_timestamp
            last_timestamp = timestamp
            min_input = (
                input_tokens if min_input is None else min(min_input, input_tokens)
            )
            max_input = (
                input_tokens if max_input is None else max(max_input, input_tokens)
            )
            min_output = (
                output_tokens if min_output is None else min(min_output, output_tokens)
            )
            max_output = (
                output_tokens if max_output is None else max(max_output, output_tokens)
            )

    if rows == 0:
        raise ValueError(f"{trace_path} contains no requests")
    ordered = sorted(shapes)
    record = {
        "path": str(trace_path),
        "bytes": trace_path.stat().st_size,
        "sha256": sha256_file(trace_path),
        "rows": rows,
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "raw_input_token_range": [min_input, max_input],
        "fleet_normalized_output_token_range": [min_output, max_output],
        "unique_padded_shapes": len(ordered),
        "padded_shapes_sha256": canonical_json_sha256(ordered),
    }
    return ordered, record


def _source_paths(repo_root: Path) -> list[Path]:
    paths: set[Path] = {
        Path(__file__).resolve(),
        repo_root / "pyproject.toml",
        repo_root / "configs" / "models" / f"{MODEL}.json",
        repo_root / "configs" / "chips" / f"tpuv{VERSION}.json",
        repo_root / "configs" / "systems" / "system_config.json",
        repo_root / "configs" / "fleetsim" / "figure_05_llama3_70b_tpuv5p_p20d8.json",
    }
    paths.update(
        path
        for path in (repo_root / "neusim").rglob("*.py")
        if "tests" not in path.relative_to(repo_root).parts
        and "__pycache__" not in path.parts
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "source provenance is incomplete: "
            + ", ".join(str(path) for path in sorted(missing))
        )
    return sorted(paths)


def source_tree_record(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Hash all current simulator/config sources capable of affecting the cache."""

    records: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for path in _source_paths(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        content_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
        records.append({"path": relative, "sha256": content_hash})
    return {"sha256": digest.hexdigest(), "files": records}


def runtime_record() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pydantic", "scipy"):
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def git_record(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "status_porcelain": git("status", "--short").splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def validate_paper_allocation(path: Path = ALLOCATION_CONFIG) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        allocation = json.load(stream)
    expected_counts = {"prefill": 20, "decode": 8}
    for phase in PHASES:
        entry = allocation.get(phase)
        if not isinstance(entry, dict):
            raise ValueError(f"{path} has no {phase!r} allocation")
        expected = {
            "count": expected_counts[phase],
            "npu_type": VERSION,
            "num_chips": 4,
            "batch_size": 1,
            "dp": 1,
            "tp": 4,
            "pp": 1,
        }
        mismatches = {
            key: (entry.get(key), value)
            for key, value in expected.items()
            if entry.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"{path} does not match the paper static TPUv5p TP4 BS1 allocation: {mismatches}"
            )
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": sha256_file(path),
        "allocation": allocation,
    }


def build_identity(
    *,
    policy: str,
    trace_record: Mapping[str, Any],
    shapes: Sequence[tuple[int, int]],
    source_record: Mapping[str, Any],
    runtime: Mapping[str, Any],
    padding: PaddingSchedule,
    allocation: Mapping[str, Any],
    coverage_scope: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Build the immutable portion of the run manifest and hash it."""

    basis = {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "model": MODEL,
        "version": VERSION,
        "policy": policy,
        "budgets": list(BUDGETS),
        "phases": list(PHASES),
        "topology": TOPOLOGY,
        "padding": asdict(padding),
        "trace": {
            key: trace_record[key]
            for key in (
                "sha256",
                "bytes",
                "rows",
                "padded_shapes_sha256",
                "unique_padded_shapes",
            )
        },
        "padded_shapes": [list(shape) for shape in shapes],
        "coverage_scope": coverage_scope,
        "source_tree_sha256": source_record["sha256"],
        "allocation_sha256": allocation["sha256"],
        "runtime": runtime,
        "algorithm": {
            "peak_operator_source": "fresh current LLMOpsGenerator",
            "energy_entrypoint": "analyze_all_operator_energy",
            "invocation_semantics": "independent cold invocation per budget",
            "candidate_points_shared_across_budgets": False,
            "shared_envelope_heuristic": False,
            "dvfsc_seed": 42,
            "dvfs_ga_vectorized": False,
            "dvfs_pareto_serial_within_worker": True,
            "decode_cache_unit": "one generated-token iteration",
            "detailed_dvfs_power_model": True,
        },
        "cache_key_coverage": {
            "batch_size": 1,
            "shape_set": "FleetSim-padded request pairs observed in trace",
            "configuration_specific_evidence": (
                "fresh paper-allocation peak-window backend cache audit found 787 BS1 "
                "keys matching the exact trace's 787 padded request pairs"
            ),
            "strict_full_replay_cache_miss_validation_required": True,
        },
        "data_policy": {
            "original_trace_util_result_data_consumed": False,
            "preexisting_dvfs_lookup_cache_consumed": False,
            "paper_values_used_as_simulator_output": False,
            "request_trace_consumed": True,
        },
    }
    return basis, canonical_json_sha256(basis)


def cache_file_path(task: CacheTask) -> Path:
    return (
        Path(task.policy_root)
        / MODEL
        / f"{task.input_seqlen}_{task.output_seqlen}"
        / VERSION
        / task.phase
        / "bs1.json"
    )


def build_llama_config(
    repo_root: Path, input_seqlen: int, output_seqlen: int
) -> LLMConfig:
    merged: dict[str, Any] = {}
    for path in (
        repo_root / "configs" / "models" / f"{MODEL}.json",
        repo_root / "configs" / "chips" / f"tpuv{VERSION}.json",
        repo_root / "configs" / "systems" / "system_config.json",
    ):
        with path.open(encoding="utf-8") as stream:
            merged.update(json.load(stream))
    merged.update(
        {
            "name": VERSION,
            "input_seqlen": input_seqlen,
            "output_seqlen": output_seqlen,
            "num_chips": TOPOLOGY["num_chips"],
            "global_batch_size": TOPOLOGY["batch_size"],
            "microbatch_size_ici": TOPOLOGY["batch_size"],
            "microbatch_size_dcn": TOPOLOGY["batch_size"],
            "data_parallelism_degree": TOPOLOGY["dp"],
            "tensor_parallelism_degree": TOPOLOGY["tp"],
            "pipeline_parallelism_degree": TOPOLOGY["pp"],
            "num_data_parallel_axes": 0,
            "num_tensor_parallel_axes": 2,
            "num_pipeline_parallel_axes": 1,
            # Prevent optimizer visualization side files and cross-worker races.
            "output_file_path": "",
            # Apply the same detailed V/f-table power model as the fresh
            # NoDVFS denominator and the two service-policy replays.
            "enable_dvfs": True,
        }
    )
    return LLMConfig.model_validate(merged)


def _budget_key(value: float) -> str:
    return str(float(value))


def _policy_string(policy: str, budget: float) -> str:
    return policy if budget == 0 else f"{policy}_{_budget_key(budget)}"


def grouped_document_for_task(task: CacheTask) -> dict[str, Any]:
    """Compute all independent budget points for one shape and phase."""

    # The outer ProcessPool owns parallelism. These settings prevent nested Ray
    # fan-out and preserve the original scalar-exact DVFS-C GA implementation.
    os.environ["DVFS_PARETO_SERIAL"] = "1"
    os.environ["DVFS_GA_VECTORIZED"] = "0"

    config = build_llama_config(
        Path(task.repo_root), task.input_seqlen, task.output_seqlen
    )
    if config.enable_dvfs is not True:
        raise ValueError("Figure 22 cache generation requires detailed DVFS power")
    generated = LLMOpsGenerator(config).generate(
        dump_to_file=False,
        separate_prefill_decode=True,
        analyze_energy=True,
    )
    if not isinstance(generated, tuple) or len(generated) != 3:
        raise TypeError("LLMOpsGenerator did not return split operator lists")
    _, prefill_ops, decode_ops = generated
    if not isinstance(prefill_ops, list) or not isinstance(decode_ops, list):
        raise TypeError("LLMOpsGenerator returned non-list phase operators")

    peak_ops = prefill_ops if task.phase == "prefill" else decode_ops
    phase_config = config.model_copy(deep=True)
    if task.phase == "decode":
        for op in peak_ops:
            count = op.stats.count
            if count < config.output_seqlen or count % config.output_seqlen:
                raise ValueError(
                    f"decode count {count} is not a positive multiple of "
                    f"output_seqlen {config.output_seqlen}: {op}"
                )
            op.stats.count = count // config.output_seqlen
        # Counts are now per iteration; retain identical optimizer weighting.
        phase_config.output_seqlen = 1

    peak_time_ns = sum(op.stats.execution_time_ns * op.stats.count for op in peak_ops)
    peak_energy_J = sum(op.stats.total_energy_J * op.stats.count for op in peak_ops)
    if (
        peak_time_ns <= 0
        or not math.isfinite(float(peak_energy_J))
        or peak_energy_J <= 0
    ):
        raise ValueError("fresh peak operators have invalid time or energy totals")

    points: dict[str, dict[str, Any]] = {}
    for budget in task.budgets:
        # Both the operator list and mutable config are cold copies. In
        # particular, no candidate envelope or GA population is reused.
        ops = deepcopy(peak_ops)
        budget_config = phase_config.model_copy(deep=True)
        timing: dict[str, Any] = {}
        started = time.perf_counter()
        analyze_all_operator_energy(
            ops,
            budget_config,
            pg_config=None,
            dvfs_config=_policy_string(task.policy, budget),
            timing_result=timing,
        )
        wall_seconds = time.perf_counter() - started
        total_time_ns = sum(op.stats.execution_time_ns * op.stats.count for op in ops)
        total_energy_J = sum(op.stats.total_energy_J * op.stats.count for op in ops)
        if (
            total_time_ns <= 0
            or not math.isfinite(float(total_energy_J))
            or total_energy_J <= 0
        ):
            raise ValueError(
                f"invalid {task.policy} point at budget {budget}: "
                f"time={total_time_ns}, energy={total_energy_J}"
            )
        points[_budget_key(budget)] = {
            "time_ns_per_stage": int(total_time_ns),
            "energy_J_per_chip": float(total_energy_J),
            "requested_perf_degrad": budget,
            "actual_perf_degrad_vs_peak": float(total_time_ns / peak_time_ns - 1),
            "energy_saving_vs_peak": float(1 - total_energy_J / peak_energy_J),
            "optimizer_wall_time_seconds": wall_seconds,
        }

    document = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            "producer": PRODUCER,
            "model": MODEL,
            "input_seqlen": task.input_seqlen,
            "output_seqlen": task.output_seqlen,
            "version": VERSION,
            "phase": task.phase,
            "batch_size": TOPOLOGY["batch_size"],
            "dvfs_policy": task.policy,
            "budgets": list(task.budgets),
            "num_chips": TOPOLOGY["num_chips"],
            "dp": TOPOLOGY["dp"],
            "tp": TOPOLOGY["tp"],
            "pp": TOPOLOGY["pp"],
            "ep": TOPOLOGY["ep"],
            "manifest_identity_sha256": task.manifest_identity_sha256,
            "generated_at": utc_now(),
            "peak_time_ns_per_stage": int(peak_time_ns),
            "peak_energy_J_per_chip": float(peak_energy_J),
            "independent_optimizer_invocations": len(task.budgets),
            "candidate_points_shared_across_budgets": False,
            "dvfs_ga_vectorized": False,
            "detailed_dvfs_power_model": True,
        },
        "points": points,
    }
    errors = validate_grouped_document(document, task)
    if errors:
        raise CacheValidationError("; ".join(errors))
    return document


def compute_and_write_task(task: CacheTask) -> dict[str, Any]:
    path = cache_file_path(task)
    document = grouped_document_for_task(task)
    atomic_write_json(path, document)
    return {
        "path": str(path),
        "input_seqlen": task.input_seqlen,
        "output_seqlen": task.output_seqlen,
        "phase": task.phase,
    }


def validate_grouped_document(document: Any, task: CacheTask) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["root is not an object"]
    metadata = document.get("metadata")
    points = document.get("points")
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if not isinstance(metadata, dict):
        return errors + ["metadata is not an object"]
    expected_metadata = {
        "model": MODEL,
        "input_seqlen": task.input_seqlen,
        "output_seqlen": task.output_seqlen,
        "version": VERSION,
        "phase": task.phase,
        "batch_size": TOPOLOGY["batch_size"],
        "dvfs_policy": task.policy,
        "num_chips": TOPOLOGY["num_chips"],
        "dp": TOPOLOGY["dp"],
        "tp": TOPOLOGY["tp"],
        "pp": TOPOLOGY["pp"],
        "manifest_identity_sha256": task.manifest_identity_sha256,
        "candidate_points_shared_across_budgets": False,
        "dvfs_ga_vectorized": False,
        "detailed_dvfs_power_model": True,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            errors.append(
                f"metadata.{key}={metadata.get(key)!r}, expected {expected!r}"
            )
    if metadata.get("budgets") != list(task.budgets):
        errors.append("metadata.budgets mismatch")
    expected_keys = {_budget_key(value) for value in task.budgets}
    if not isinstance(points, dict):
        return errors + ["points is not an object"]
    if set(points) != expected_keys:
        errors.append(
            f"point budgets={sorted(points)}, expected={sorted(expected_keys)}"
        )
    for key in sorted(expected_keys & set(points)):
        point = points[key]
        if not isinstance(point, dict):
            errors.append(f"points.{key} is not an object")
            continue
        for field in (
            "time_ns_per_stage",
            "energy_J_per_chip",
            "actual_perf_degrad_vs_peak",
            "energy_saving_vs_peak",
        ):
            value = point.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
            ):
                errors.append(f"points.{key}.{field} is not finite")
            elif field in ("time_ns_per_stage", "energy_J_per_chip") and value <= 0:
                errors.append(f"points.{key}.{field} is not positive")
        if point.get("requested_perf_degrad") != float(key):
            errors.append(f"points.{key}.requested_perf_degrad mismatch")
    return errors


def validate_cache_file(path: Path, task: CacheTask) -> tuple[bool, list[str]]:
    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        return False, [str(error)]
    errors = validate_grouped_document(document, task)
    return not errors, errors


def build_tasks(
    *,
    repo_root: Path,
    policy_root: Path,
    policy: str,
    shapes: Iterable[tuple[int, int]],
    identity_sha256: str,
) -> list[CacheTask]:
    return [
        CacheTask(
            repo_root=str(repo_root),
            policy_root=str(policy_root),
            policy=policy,
            input_seqlen=input_seqlen,
            output_seqlen=output_seqlen,
            phase=phase,
            manifest_identity_sha256=identity_sha256,
        )
        for input_seqlen, output_seqlen in sorted(shapes)
        for phase in PHASES
    ]


def output_tree_record(policy_root: Path, paths: Sequence[Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for path in sorted(paths):
        relative = path.relative_to(policy_root).as_posix()
        content_hash = sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content_hash.encode("ascii"))
        digest.update(b"\n")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": content_hash,
            }
        )
    return {"sha256": digest.hexdigest(), "files": records}


def validate_coverage(
    policy_root: Path,
    tasks: Sequence[CacheTask],
    *,
    complete_trace: bool,
) -> dict[str, Any]:
    """Require exact shape/phase coverage and validate every grouped point."""

    expected = {cache_file_path(task).resolve(): task for task in tasks}
    discovered = {path.resolve() for path in policy_root.rglob("bs1.json")}
    missing = sorted(set(expected) - discovered)
    unexpected = sorted(discovered - set(expected))
    invalid: list[dict[str, Any]] = []
    files_with_budget_variation = 0
    files_with_positive_energy_saving = 0
    for path, task in expected.items():
        if path not in discovered:
            continue
        valid, errors = validate_cache_file(path, task)
        if not valid:
            invalid.append({"path": str(path), "errors": errors})
            continue
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
        points = document["points"].values()
        time_energy_pairs = {
            (
                int(point["time_ns_per_stage"]),
                float(point["energy_J_per_chip"]),
            )
            for point in points
        }
        if len(time_energy_pairs) > 1:
            files_with_budget_variation += 1
        if any(
            float(point["energy_saving_vs_peak"]) > 0
            for point in document["points"].values()
        ):
            files_with_positive_energy_saving += 1
    if missing or unexpected or invalid:
        raise CacheValidationError(
            json.dumps(
                {
                    "missing": [str(path) for path in missing],
                    "unexpected": [str(path) for path in unexpected],
                    "invalid": invalid,
                },
                indent=2,
            )
        )
    if files_with_budget_variation == 0 or files_with_positive_energy_saving == 0:
        raise CacheValidationError(
            "DVFS cache is globally degenerate: "
            f"varying_files={files_with_budget_variation}, "
            f"positive_saving_files={files_with_positive_energy_saving}"
        )
    tree = output_tree_record(policy_root, sorted(expected))
    return {
        "status": "complete" if complete_trace else "diagnostic_incomplete",
        "complete_trace_coverage": complete_trace,
        "grouped_files": len(expected),
        "shape_pairs": len(expected) // len(PHASES),
        "phases": list(PHASES),
        "budgets_per_file": len(BUDGETS),
        "optimizer_invocations": len(expected) * len(BUDGETS),
        "detailed_dvfs_power_model": True,
        "files_with_budget_variation": files_with_budget_variation,
        "files_with_positive_energy_saving": (files_with_positive_energy_saving),
        "output_tree": tree,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise CacheValidationError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(document, dict):
        raise CacheValidationError(f"manifest {path} is not an object")
    return document


def assert_manifest_matches(
    manifest: Mapping[str, Any],
    identity_basis: Mapping[str, Any],
    identity_sha256: str,
) -> None:
    errors: list[str] = []
    if manifest.get("identity_sha256") != identity_sha256:
        errors.append("identity_sha256")
    # JSON round-trips tuple-valued padding schedules as lists. Compare the
    # canonical serialized documents so a valid manifest remains resumable.
    if canonical_json_sha256(manifest.get("identity")) != canonical_json_sha256(
        identity_basis
    ):
        errors.append("identity document")
    provenance = manifest.get("provenance", {})
    if (
        not isinstance(provenance, dict)
        or provenance.get("trace", {}).get("sha256")
        != identity_basis["trace"]["sha256"]
    ):
        errors.append("trace hash")
    if (
        not isinstance(provenance, dict)
        or provenance.get("source_tree", {}).get("sha256")
        != identity_basis["source_tree_sha256"]
    ):
        errors.append("source hash")
    if errors:
        raise CacheValidationError(
            "existing cache manifest does not match current "
            + ", ".join(errors)
            + "; use a new --output-dir instead of mixing result sources"
        )


def initial_manifest(
    *,
    identity_basis: Mapping[str, Any],
    identity_sha256: str,
    trace_record: Mapping[str, Any],
    source_record: Mapping[str, Any],
    allocation_record: Mapping[str, Any],
    git: Mapping[str, Any],
    workers: int,
    grouped_files: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": PRODUCER,
        "identity_sha256": identity_sha256,
        "identity": identity_basis,
        "provenance": {
            "trace": trace_record,
            "source_tree": source_record,
            "allocation": allocation_record,
            "git": git,
            "data_policy": identity_basis["data_policy"],
        },
        "execution": {
            "state": "in_progress",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "workers": workers,
            "max_workers_guard": MAX_WORKERS,
            "grouped_files_planned": grouped_files,
            "optimizer_invocations_planned": grouped_files * len(BUDGETS),
        },
    }


def _collect_run_inputs(
    trace: Path, policy: str, max_pairs: int | None
) -> tuple[
    PaddingSchedule,
    list[tuple[int, int]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
]:
    padding = current_padding_schedule()
    all_shapes, trace_record = extract_padded_shapes(trace, padding)
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    complete_trace = max_pairs is None or max_pairs >= len(all_shapes)
    shapes = all_shapes if complete_trace else all_shapes[:max_pairs]
    coverage_scope = {
        "mode": "full_trace" if complete_trace else "diagnostic_sorted_prefix",
        "complete_trace_coverage": complete_trace,
        "full_trace_shape_pairs": len(all_shapes),
        "selected_shape_pairs": len(shapes),
        "max_pairs_argument": max_pairs,
        "selection": (
            "all" if complete_trace else "sorted(input_seqlen, output_seqlen) prefix"
        ),
    }
    sources = source_tree_record(REPO_ROOT)
    runtime = runtime_record()
    allocation = validate_paper_allocation()
    identity, identity_sha256 = build_identity(
        policy=policy,
        trace_record=trace_record,
        shapes=shapes,
        source_record=sources,
        runtime=runtime,
        padding=padding,
        allocation=allocation,
        coverage_scope=coverage_scope,
    )
    return (
        padding,
        shapes,
        trace_record,
        sources,
        allocation,
        identity,
        identity_sha256,
    )


def generate_cache(
    *,
    trace: Path,
    output_dir: Path,
    policy: str,
    workers: int,
    max_pairs: int | None = None,
    validate_only: bool = False,
) -> tuple[Path, dict[str, Any]]:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"workers must be in [1, {MAX_WORKERS}]")

    (
        _padding,
        shapes,
        trace_record,
        sources,
        allocation,
        identity,
        identity_sha256,
    ) = _collect_run_inputs(trace, policy, max_pairs)
    complete_trace = bool(identity["coverage_scope"]["complete_trace_coverage"])
    policy_root = output_dir.expanduser().resolve() / policy
    manifest_path = policy_root / "manifest.json"
    tasks = build_tasks(
        repo_root=REPO_ROOT,
        policy_root=policy_root,
        policy=policy,
        shapes=shapes,
        identity_sha256=identity_sha256,
    )

    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
        assert_manifest_matches(manifest, identity, identity_sha256)
    else:
        existing = list(policy_root.rglob("bs1.json")) if policy_root.exists() else []
        if existing:
            raise CacheValidationError(
                f"{policy_root} has cache files but no matching manifest; "
                "use a new --output-dir"
            )
        if validate_only:
            raise CacheValidationError(f"manifest does not exist: {manifest_path}")
        manifest = initial_manifest(
            identity_basis=identity,
            identity_sha256=identity_sha256,
            trace_record=trace_record,
            source_record=sources,
            allocation_record=allocation,
            git=git_record(),
            workers=workers,
            grouped_files=len(tasks),
        )
        atomic_write_json(manifest_path, manifest)

    if validate_only:
        coverage = validate_coverage(policy_root, tasks, complete_trace=complete_trace)
        return manifest_path, coverage

    manifest["execution"].update(
        {
            "state": "in_progress",
            "updated_at": utc_now(),
            "workers": workers,
        }
    )
    manifest.pop("coverage", None)
    atomic_write_json(manifest_path, manifest)

    pending: list[CacheTask] = []
    resumed = 0
    for task in tasks:
        valid, _errors = validate_cache_file(cache_file_path(task), task)
        if valid:
            resumed += 1
        else:
            pending.append(task)

    print(
        f"{policy}: {len(shapes)} shapes, {len(tasks)} grouped files, "
        f"{len(tasks) * len(BUDGETS)} independent optimizer invocations; "
        f"{resumed} validated files resumed, {len(pending)} files pending; "
        f"workers={workers}",
        flush=True,
    )

    completed = 0
    try:
        if pending:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=context
            ) as executor:
                futures = {
                    executor.submit(compute_and_write_task, task): task
                    for task in pending
                }
                progress_interval = max(1, len(pending) // 100)
                for future in as_completed(futures):
                    future.result()
                    completed += 1
                    if completed == len(pending) or completed % progress_interval == 0:
                        print(
                            f"{policy}: completed {completed}/{len(pending)} "
                            "pending grouped files",
                            flush=True,
                        )

        # Detect trace or source mutation during a long-running sweep.
        (
            _padding_after,
            shapes_after,
            trace_after,
            sources_after,
            allocation_after,
            identity_after,
            identity_sha256_after,
        ) = _collect_run_inputs(trace, policy, max_pairs)
        if (
            identity_sha256_after != identity_sha256
            or identity_after != identity
            or shapes_after != shapes
        ):
            raise CacheValidationError(
                "trace, simulator source, runtime, or configuration changed "
                "while the cache was being generated; rerun into a new output directory"
            )
        # Keep detailed post-run records reachable for manual provenance review.
        manifest["provenance"]["trace"] = trace_after
        manifest["provenance"]["source_tree"] = sources_after
        manifest["provenance"]["allocation"] = allocation_after
        coverage = validate_coverage(policy_root, tasks, complete_trace=complete_trace)
    except BaseException as error:
        manifest["execution"].update(
            {
                "state": "failed",
                "updated_at": utc_now(),
                "files_completed_this_launch": completed,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise

    manifest["execution"].update(
        {
            "state": ("complete" if complete_trace else "diagnostic_incomplete"),
            "updated_at": utc_now(),
            "completed_at": utc_now(),
            "files_resumed_this_launch": resumed,
            "files_completed_this_launch": completed,
        }
    )
    manifest["execution"].pop("error", None)
    manifest["coverage"] = coverage
    atomic_write_json(manifest_path, manifest)
    return manifest_path, coverage


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Explicit Azure request-trace CSV; no result/cache data are read.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh cache root. The policy name is appended automatically.",
    )
    parser.add_argument("--policy", required=True, choices=POLICIES)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, MAX_WORKERS),
        help=f"Bounded outer process parallelism (1-{MAX_WORKERS}; default: %(default)s).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help=(
            "Diagnostic only: generate the first K shapes in deterministic "
            "sorted order. Its manifest is marked incomplete and cannot be "
            "resumed or validated as a full-trace cache."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Recompute provenance and validate an existing cache against the same requested scope.",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must be in [1, {MAX_WORKERS}]")
    if args.max_pairs is not None and args.max_pairs < 1:
        parser.error("--max-pairs must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest, coverage = generate_cache(
        trace=args.trace,
        output_dir=args.output_dir,
        policy=args.policy,
        workers=args.workers,
        max_pairs=args.max_pairs,
        validate_only=args.validate_only,
    )
    print(
        f"Validated {coverage['grouped_files']} grouped files; "
        f"manifest: {manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
