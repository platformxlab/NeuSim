#!/usr/bin/env python3
"""Run one MICRO'26 experiment group directly against this NeuSim checkout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
REPO_ROOT = SCRIPT.parents[3]
sys.path.insert(0, str(REPO_ROOT))
if __package__ in (None, ""):
    sys.path.insert(0, str(SCRIPT.parent))
    from neusim_adapter import (  # noqa: E402
        AUTHORITATIVE_MS_SOURCE,
        CURRENT_IDEAL_RAW_CONFIG_LIMIT,
        CURRENT_IDEAL_SEARCH_EXECUTION,
        MODEL_FILES,
        Analysis,
        NativeExperimentError,
        PhaseTrace,
        analyze_dvfsc_all_budgets,
        analyze_enpu_ms_all_budgets,
        analyze_policy_all_budgets,
        analyze_trace,
        atomic_json,
        build_phase_trace,
        current_state_space,
        evaluate_configured_trace,
        legacy_csv_path,
        load_json,
        provenance,
        transplant_modal_schedule,
        write_operator_csv,
    )
else:
    from .neusim_adapter import (  # noqa: E402
        AUTHORITATIVE_MS_SOURCE,
        CURRENT_IDEAL_RAW_CONFIG_LIMIT,
        CURRENT_IDEAL_SEARCH_EXECUTION,
        MODEL_FILES,
        Analysis,
        NativeExperimentError,
        PhaseTrace,
        analyze_dvfsc_all_budgets,
        analyze_enpu_ms_all_budgets,
        analyze_policy_all_budgets,
        analyze_trace,
        atomic_json,
        build_phase_trace,
        current_state_space,
        evaluate_configured_trace,
        legacy_csv_path,
        load_json,
        provenance,
        transplant_modal_schedule,
        write_operator_csv,
    )


GROUPS = (
    "standard_sweep",
    "domain_count",
    "temporal_granularity",
    "fixed_sequence_sweep",
    "expert_imbalance",
    "power_gating",
)
RAY_GROUPS = frozenset(GROUPS)
SCOPED_GROUPS = frozenset(
    {"standard_sweep", "domain_count", "temporal_granularity"}
)
PHASES = ("prefill", "decode")
POLICIES = ("DVFS-C", "eNPU-C", "eNPU-All", "Ideal")

STANDARD_CHECKPOINT_SCHEMA_VERSION = 1
STANDARD_CHECKPOINT_NAME = ".standard_sweep_resume.json"
DEFAULT_TRACE_WORKERS = 4
# expert_imbalance remains serial because its factor and hot/cold searches share
# generated/search filenames; parallelizing it would introduce output races.
PARALLEL_TRACE_GROUPS = frozenset(set(GROUPS) - {"expert_imbalance"})
TEMPORAL_REUSED_POLICIES = ("NoDVFS", "DVFS-C", "eNPU-All", "Ideal")
TEMPORAL_COMPUTED_POLICIES = ("DVFS-C-ms", "eNPU-ms")


@dataclass(frozen=True)
class TemporalStandardReuse:
    source: Path
    source_sha256: str
    rows: dict[tuple[str, str, str, float], dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ideal_search_provenance() -> dict[str, Any]:
    state_space = current_state_space()
    return {
        "implementation": "filtered_exhaustive_with_balanced_endpoint_cap",
        "raw_candidate_reduction_threshold": CURRENT_IDEAL_RAW_CONFIG_LIMIT,
        "unreduced_table_lattice_states": int(state_space["exact_product"]),
        "unreduced_table_lattice_is_theoretical": True,
        "full_theoretical_lattice_enumerated": False,
        "reduction": (
            "raw filtered products above the threshold retain the minimum and "
            "maximum frequency at each voltage; if the product remains above "
            "the threshold, deterministic frequency-ordered balanced samples "
            "preserve true V/f endpoints and enforce the cap"
        ),
        "enumeration": "lazy per operator with sequential outer exhaustive scheduling",
        "analysis_execution": dict(CURRENT_IDEAL_SEARCH_EXECUTION),
    }


def _schedule_sha256(ops: Sequence[Any]) -> str:
    fields = (
        "dvfs_sa",
        "dvfs_vu",
        "dvfs_sram",
        "dvfs_hbm_mc",
        "dvfs_hbm_die",
        "dvfs_hbm_io",
        "dvfs_ici_mc",
        "dvfs_ici_phy",
    )
    rows = []
    for op in ops:
        rows.append(
            {
                "name": op.name,
                "count": int(op.stats.count),
                "domains": {
                    name: {
                        "frequency_GHz": getattr(
                            getattr(op, name), "frequency_GHz", None
                        ),
                        "voltage_V": getattr(getattr(op, name), "voltage_V", None),
                    }
                    for name in fields
                },
            }
        )
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _host_provenance(args: argparse.Namespace) -> dict[str, Any]:
    cpu_model = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                cpu_model = line.split(":", 1)[1].strip()
                break
    memory_bytes = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                memory_bytes = int(line.split()[1]) * 1024
                break
    flags = (
        "DVFS_GA_VECTORIZED",
        "DVFS_GA_EXACT_BATCH_SIZE",
        "DVFS_MS_CANDIDATE_BATCH_SIZE",
        "DVFS_PARETO_SERIAL",
        "DVFS_PARETO_BATCH_SIZE",
        "DVFS_PARETO_MAX_INFLIGHT_BATCHES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "ray_version": _package_version("ray"),
        "numpy_version": _package_version("numpy"),
        "requested_jobs": args.jobs,
        "trace_workers": int(getattr(args, "trace_workers", 1)),
        "optimizer_environment": {name: os.environ.get(name) for name in flags},
        "paper_host_reference": {
            "cpu": "24 Xeon Gold 6136 cores",
            "memory": "128 GB",
            "current_timing_directly_comparable": False,
        },
    }


def _models(paper: Mapping[str, Any], quick: bool) -> list[str]:
    values = [str(entry["id"]) for entry in paper["models"]]
    return values[:1] if quick else values


def _selected_models(
    args: argparse.Namespace, paper: Mapping[str, Any]
) -> list[str]:
    """Return the manifest-ordered model scope for this invocation."""
    requested = tuple(getattr(args, "models", ()) or ())
    if not requested:
        return _models(paper, args.quick)
    requested_set = set(requested)
    return [
        str(entry["id"])
        for entry in paper["models"]
        if str(entry["id"]) in requested_set
    ]


def _selected_phases(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the canonical-order phase scope for this invocation."""
    requested = tuple(getattr(args, "phases", ()) or ())
    if not requested:
        return PHASES
    requested_set = set(requested)
    return tuple(phase for phase in PHASES if phase in requested_set)


def _thresholds(paper: Mapping[str, Any], key: str, quick: bool) -> list[float]:
    values = [
        float(value)
        for value in paper["performance_degradation_thresholds_percent"][key]
    ]
    return values[:1] if quick else values


def _tokens(paper: Mapping[str, Any]) -> tuple[int, int]:
    request = paper["paper"]["default_request_tokens"]
    return int(request["input"]), int(request["output"])


def _table_3_configuration_contract(paper: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate the Table 3 topology encoded by the manifest."""
    configuration_source = paper.get(
        "configuration_source", "paper_experiments.json Table 3 transcription"
    )
    if not isinstance(configuration_source, str) or not configuration_source.strip():
        raise NativeExperimentError(
            "paper manifest configuration_source must be a non-empty string"
        )
    semantics = paper.get("table_3_configuration_semantics")
    if not isinstance(semantics, Mapping):
        raise NativeExperimentError(
            "paper manifest lacks table_3_configuration_semantics"
        )
    models = paper.get("models")
    if not isinstance(models, list):
        raise NativeExperimentError("paper manifest models must be a list")

    def positive_int(value: Any, context: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise NativeExperimentError(f"{context} must be a positive integer")
        return value

    input_tokens, output_tokens = _tokens(paper)
    input_tokens = positive_int(input_tokens, "default input token count")
    output_tokens = positive_int(output_tokens, "default output token count")
    normalized: dict[str, Any] = {}
    for entry in models:
        if not isinstance(entry, Mapping):
            raise NativeExperimentError("paper manifest model entries must be objects")
        model_id = str(entry.get("id", ""))
        if model_id not in MODEL_FILES:
            raise NativeExperimentError(
                f"paper manifest contains unsupported model id {model_id!r}"
            )
        if model_id in normalized:
            raise NativeExperimentError(
                f"paper manifest contains duplicate model id {model_id!r}"
            )
        phases = entry.get("phases")
        cluster_nodes = entry.get("cluster_nodes")
        if not isinstance(phases, Mapping) or not isinstance(cluster_nodes, Mapping):
            raise NativeExperimentError(
                f"{model_id} must define phases and Table 3 cluster_nodes"
            )
        phase_contract: dict[str, Any] = {}
        for phase in ("prefill", "decode"):
            phase_entry = phases.get(phase)
            parallelism = (
                phase_entry.get("parallelism")
                if isinstance(phase_entry, Mapping)
                else None
            )
            if not isinstance(phase_entry, Mapping) or not isinstance(
                parallelism, Mapping
            ):
                raise NativeExperimentError(
                    f"{model_id}/{phase} must define chips and DP/TP/PP/EP"
                )
            chips = positive_int(phase_entry.get("chips"), f"{model_id}/{phase} chips")
            axes = {
                name: positive_int(
                    parallelism.get(name), f"{model_id}/{phase} {name} parallelism"
                )
                for name in ("data", "tensor", "pipeline", "expert")
            }
            if math.prod(axes.values()) != chips:
                raise NativeExperimentError(
                    f"{model_id}/{phase} DP/TP/PP/EP product does not equal chips"
                )
            service_nodes = positive_int(
                cluster_nodes.get(phase),
                f"{model_id}/{phase} service cluster nodes",
            )
            config = (
                f"dp{axes['data']}-tp{axes['tensor']}-pp{axes['pipeline']}-"
                f"ep{axes['expert']}-chips{chips}-b1"
            )
            phase_contract[phase] = {
                "request_node": {
                    "chips": chips,
                    "parallelism": axes,
                    "config_label": config,
                },
                "service_cluster_nodes": service_nodes,
            }
        normalized[model_id] = {
            "model": MODEL_FILES[model_id],
            "paper_label": str(entry.get("paper_label", "")),
            "phases": phase_contract,
        }
    missing = sorted(set(MODEL_FILES) - set(normalized))
    if missing:
        raise NativeExperimentError(
            "paper manifest lacks Table 3 models: " + ", ".join(missing)
        )
    return {
        "source": configuration_source,
        "request_tokens": {"input": input_tokens, "output": output_tokens},
        "models": normalized,
    }


def _validate_standard_table_3_records(
    records: Sequence[Any], paper: Mapping[str, Any]
) -> dict[str, Any]:
    """Confirm every standard-sweep row uses its Table 3 request-node config."""
    contract = _table_3_configuration_contract(paper)
    if not records:
        raise NativeExperimentError("standard sweep produced no records")
    expected_tokens = contract["request_tokens"]
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise NativeExperimentError(
                f"standard sweep record {index} is not a JSON object"
            )
        model_id = str(row.get("model_id", ""))
        model_contract = contract["models"].get(model_id)
        phase = str(row.get("phase", ""))
        phase_contract = (
            model_contract["phases"].get(phase) if model_contract is not None else None
        )
        if phase_contract is None:
            raise NativeExperimentError(
                f"standard sweep record {index} has unknown Table 3 cell "
                f"{model_id!r}/{phase!r}"
            )
        expected = {
            "model": model_contract["model"],
            "config": phase_contract["request_node"]["config_label"],
            "input_tokens": expected_tokens["input"],
            "output_tokens": expected_tokens["output"],
        }
        mismatches = [
            f"{field}={row.get(field)!r} (expected {value!r})"
            for field, value in expected.items()
            if row.get(field) != value
        ]
        if mismatches:
            raise NativeExperimentError(
                f"standard sweep record {index} violates Table 3: "
                + "; ".join(mismatches)
            )
    return {
        "status": "passed",
        "records_checked": len(records),
        "validated_fields": [
            "model_id",
            "model",
            "phase",
            "config",
            "input_tokens",
            "output_tokens",
        ],
        "contract": contract,
    }


def _annotate_standard_table_3_check(
    output_dir: Path, paper: Mapping[str, Any]
) -> None:
    path = output_dir / "energy_records.json"
    payload = load_json(path)
    records = payload.get("records")
    if not isinstance(records, list):
        raise NativeExperimentError(f"{path} lacks a records list")
    payload["table_3_configuration_check"] = _validate_standard_table_3_records(
        records, paper
    )
    atomic_json(path, payload)


def _trace(
    args: argparse.Namespace,
    paper: Mapping[str, Any],
    model_id: str,
    phase: str,
    input_tokens: int,
    output_tokens: int,
    overrides: Mapping[str, Any] | None = None,
) -> PhaseTrace:
    print(
        f"[native] generating {model_id}/{phase} input={input_tokens} output={output_tokens}"
    )
    hint = (
        args.output_dir
        / "generated"
        / model_id
        / phase
        / f"{model_id}_{input_tokens}_{output_tokens}_{phase}.csv"
    )
    return build_phase_trace(
        repo_root=args.repo_root,
        paper=paper,
        model_id=model_id,
        phase=phase,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        output_hint=hint,
        config_overrides=overrides,
        verbose_simulator=args.verbose_simulator,
    )


def _token_trace_args(
    args: argparse.Namespace,
    model_id: str,
    phase: str,
    input_tokens: int,
    output_tokens: int,
) -> argparse.Namespace:
    """Clone args with a task-private root for repeated model/phase token axes."""
    worker_args = copy(args)
    worker_args.output_dir = (
        args.output_dir
        / "parallel_trace_workspaces"
        / model_id
        / phase
        / f"in{input_tokens}_out{output_tokens}"
    )
    return worker_args


def _analyze(
    args: argparse.Namespace,
    trace: PhaseTrace,
    policy: str,
    threshold: float,
    **kwargs: Any,
) -> Analysis:
    return analyze_trace(
        trace,
        policy=policy,
        threshold_pct=threshold,
        allow_operator_local_fallback=args.quick,
        allow_current_ideal=args.allow_current_ideal,
        **kwargs,
    )


def _attach(
    row: dict[str, Any], run_provenance: Mapping[str, Any], quick: bool
) -> dict[str, Any]:
    row["quick_smoke"] = quick
    row["provenance"] = dict(run_provenance)
    return row


PLANNING_PROVENANCE_FIELDS = (
    "dvfs_execution_mode",
    "ga_execution_mode",
    "ga_exact_batch_size",
    "ga_exact_batch_size_env",
    "intra_op_algorithm",
    "batch_budgets_pct",
    "candidate_evaluation_mode",
    "candidate_batch_size",
    "candidate_batch_size_env",
    "candidate_count",
    "submitted_candidate_tasks",
    "candidate_result_order",
    "candidate_envelope_fraction",
    "candidate_envelope_semantics",
    "candidate_set_shared_across_budgets",
    "independent_per_budget_candidate_semantics_preserved",
    "budget_dependent_candidate_reduction_preserved",
    "shared_envelope_caveat",
    "ideal_shared_envelope_caveat",
    "ideal_raw_candidate_reduction_threshold",
    "ideal_theoretical_lattice_states",
    "ideal_full_theoretical_lattice_enumerated",
    "ga_policy",
    "candidate_generation",
    "baseline_semantics",
    "raw_baseline_time_ns",
    "zero_degradation_baseline_injection",
)


def _planning_provenance(analysis: Analysis) -> dict[str, Any]:
    return {
        field: analysis.record[field]
        for field in PLANNING_PROVENANCE_FIELDS
        if field in analysis.record
    }


def _standard_checkpoint_signature(
    args: argparse.Namespace,
    paper: Mapping[str, Any],
    prov: Mapping[str, Any],
) -> dict[str, Any]:
    selected_models = _selected_models(args, paper)
    selected_phases = _selected_phases(args)
    contract = _table_3_configuration_contract(paper)
    model_axes = {
        model_id: {
            "model": contract["models"][model_id]["model"],
            "phases": {
                phase: contract["models"][model_id]["phases"][phase]
                for phase in selected_phases
            },
        }
        for model_id in selected_models
    }
    return {
        "schema_version": STANDARD_CHECKPOINT_SCHEMA_VERSION,
        "simulator_revision": prov.get("simulator_revision"),
        "workspace_diff_sha256": prov.get("workspace_diff_sha256"),
        "paper_manifest_sha256": _sha256(args.paper_manifest),
        "pipeline_manifest_sha256": _sha256(args.pipeline_manifest),
        "quick": args.quick,
        "allow_current_ideal": args.allow_current_ideal,
        "models": selected_models,
        "jobs": int(getattr(args, "jobs", 1)),
        "trace_workers": int(getattr(args, "trace_workers", 1)),
        "dvfs_ga_vectorized": os.environ.get("DVFS_GA_VECTORIZED"),
        "dvfs_ga_exact_batch_size": os.environ.get("DVFS_GA_EXACT_BATCH_SIZE"),
        "model_configuration_axes": model_axes,
        "phases": list(selected_phases),
        "default_request_tokens": list(_tokens(paper)),
        "thresholds_pct": _thresholds(paper, "paper_sweep", args.quick),
        "policies": ["NoDVFS", *POLICIES],
    }


def _load_standard_checkpoint(
    path: Path, signature: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, NativeExperimentError):
        return {}
    entries = payload.get("completed_traces")
    if (
        payload.get("schema_version") != STANDARD_CHECKPOINT_SCHEMA_VERSION
        or payload.get("signature") != signature
        or not isinstance(entries, Mapping)
    ):
        return {}
    return {
        str(key): dict(value)
        for key, value in entries.items()
        if isinstance(key, str) and isinstance(value, Mapping)
    }


def _checkpoint_cell_key(row: Mapping[str, Any]) -> tuple[str, str, str, float] | None:
    try:
        threshold = float(row["threshold_pct"])
        if not math.isfinite(threshold):
            return None
        return (
            str(row["model_id"]),
            str(row["phase"]),
            str(row["policy"]),
            threshold,
        )
    except (KeyError, TypeError, ValueError):
        return None


def _checkpoint_rows_match(
    rows: Any,
    expected: Sequence[Mapping[str, Any]],
    *,
    trace: PhaseTrace,
    require_trace_fields: bool,
    prov: Mapping[str, Any],
) -> bool:
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        return False
    expected_keys = {
        (
            str(row["model_id"]),
            str(row["phase"]),
            str(row["policy"]),
            float(row["threshold_pct"]),
        )
        for row in expected
    }
    actual_keys = [_checkpoint_cell_key(row) for row in rows]
    if (
        any(key is None for key in actual_keys)
        or len(actual_keys) != len(expected_keys)
        or set(actual_keys) != expected_keys
    ):
        return False
    for row in rows:
        if (
            row.get("model_id") != trace.model_id
            or row.get("model") != trace.model
            or row.get("phase") != trace.phase
            or row.get("pg_strategy") != "NoPG"
            or row.get("quick_smoke") is not False
        ):
            return False
        source_prov = row.get("provenance")
        if not isinstance(source_prov, Mapping) or any(
            source_prov.get(field) != prov.get(field)
            for field in ("simulator_revision", "workspace_diff_sha256")
        ):
            return False
        if require_trace_fields and (
            row.get("input_tokens") != trace.input_tokens
            or row.get("output_tokens") != trace.output_tokens
            or row.get("config") != trace.config_label
        ):
            return False
    return True


def _operator_csv_matches_checkpoint(
    path: Path,
    expected_sha256: Any,
    *,
    trace: PhaseTrace,
    policy: str,
    threshold: float,
) -> bool:
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or not path.is_file()
        or path.stat().st_size == 0
    ):
        return False
    try:
        if _sha256(path) != expected_sha256:
            return False
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {
                "model",
                "model_id",
                "phase",
                "policy",
                "threshold_pct",
                "pg_strategy",
                "quick_smoke",
            }
            if not required.issubset(reader.fieldnames or ()):
                return False
            found = False
            for row in reader:
                found = True
                if (
                    row["model"] != trace.model
                    or row["model_id"] != trace.model_id
                    or row["phase"] != trace.phase
                    or row["policy"] != policy
                    or row["pg_strategy"] != "NoPG"
                    or row["quick_smoke"] != "False"
                    or not math.isclose(
                        float(row["threshold_pct"]), threshold, abs_tol=1e-12
                    )
                ):
                    return False
            return found
    except (OSError, UnicodeError, csv.Error, TypeError, ValueError):
        return False


def _restore_standard_trace(
    entry: Mapping[str, Any],
    *,
    trace: PhaseTrace,
    cells: Sequence[tuple[str, float]],
    operator_root: Path,
    output_dir: Path,
    prov: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    expected = [
        {
            "model_id": trace.model_id,
            "phase": trace.phase,
            "policy": policy,
            "threshold_pct": threshold,
        }
        for policy, threshold in cells
    ]
    records = entry.get("records")
    ivr_records = entry.get("ivr_records")
    if (
        entry.get("model_id") != trace.model_id
        or entry.get("phase") != trace.phase
        or not _checkpoint_rows_match(
            records,
            expected,
            trace=trace,
            require_trace_fields=True,
            prov=prov,
        )
        or not _checkpoint_rows_match(
            ivr_records,
            expected,
            trace=trace,
            require_trace_fields=False,
            prov=prov,
        )
    ):
        return None
    hashes = entry.get("operator_csv_sha256")
    if not isinstance(hashes, Mapping):
        return None
    expected_paths = {
        legacy_csv_path(operator_root, trace, policy, threshold): (policy, threshold)
        for policy, threshold in cells
    }
    relative_paths = {
        path.relative_to(output_dir).as_posix() for path in expected_paths
    }
    if set(hashes) != relative_paths:
        return None
    if any(
        not _operator_csv_matches_checkpoint(
            path,
            hashes[path.relative_to(output_dir).as_posix()],
            trace=trace,
            policy=policy,
            threshold=threshold,
        )
        for path, (policy, threshold) in expected_paths.items()
    ):
        return None
    return ([dict(row) for row in records], [dict(row) for row in ivr_records])


def _standard_trace_checkpoint_entry(
    *,
    trace: PhaseTrace,
    cells: Sequence[tuple[str, float]],
    operator_root: Path,
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    ivr_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    csv_hashes = {}
    for policy, threshold in cells:
        path = legacy_csv_path(operator_root, trace, policy, threshold)
        if not path.is_file() or path.stat().st_size == 0:
            raise NativeExperimentError(
                f"cannot checkpoint {trace.model_id}/{trace.phase}: "
                f"missing operator CSV {path}"
            )
        csv_hashes[path.relative_to(output_dir).as_posix()] = _sha256(path)
    return {
        "model_id": trace.model_id,
        "phase": trace.phase,
        "records": [dict(row) for row in records],
        "ivr_records": [dict(row) for row in ivr_records],
        "operator_csv_sha256": csv_hashes,
    }


def _write_standard_checkpoint(
    path: Path,
    signature: Mapping[str, Any],
    entries: Mapping[str, Mapping[str, Any]],
) -> None:
    atomic_json(
        path,
        {
            "schema_version": STANDARD_CHECKPOINT_SCHEMA_VERSION,
            "signature": dict(signature),
            "completed_traces": dict(entries),
        },
    )


def _trace_worker_count(args: argparse.Namespace, task_count: int) -> int:
    """Return the bounded count of isolated outer trace workers."""
    if args.quick or task_count < 2:
        return 1
    requested = int(getattr(args, "trace_workers", 1))
    jobs = int(getattr(args, "jobs", 1))
    if requested <= 1 or jobs <= 1:
        return 1
    return min(requested, task_count, jobs)


def _trace_task_count(args: argparse.Namespace, paper: Mapping[str, Any]) -> int:
    """Return the deterministic logical outer-task count for one group."""
    if args.group in {
        "standard_sweep",
        "domain_count",
        "temporal_granularity",
    }:
        return len(_selected_models(args, paper)) * len(_selected_phases(args))
    if args.group == "fixed_sequence_sweep":
        sequence = paper["sequence_lengths_tokens"]
        lengths = [int(value) for value in sequence["input_sweep"]]
        if args.quick:
            lengths = [lengths[0], int(sequence["default_input"])]
        return 4 * len(lengths)
    if args.group == "expert_imbalance":
        factors = list(paper["expert_capacity_factors"])
        return 2 if args.quick else len(factors)
    if args.group == "power_gating":
        return 2
    raise NativeExperimentError(f"unknown experiment group: {args.group}")


def _trace_parallelism_provenance(
    args: argparse.Namespace, paper: Mapping[str, Any]
) -> dict[str, Any]:
    task_count = _trace_task_count(args, paper)
    configured_cap = int(
        getattr(
            args,
            "configured_trace_worker_cap",
            getattr(args, "trace_workers", 1),
        )
    )
    effective_workers = (
        _trace_worker_count(args, task_count)
        if args.group in PARALLEL_TRACE_GROUPS
        else 1
    )
    scheduler = (
        "isolated_ray_workers_with_canonical_parent_collation"
        if effective_workers > 1
        else "serial_shared_output_paths"
        if args.group == "expert_imbalance"
        else "serial_in_process"
    )
    payload = {
        "enabled": effective_workers > 1,
        "configured_worker_cap": configured_cap,
        "effective_workers": effective_workers,
        "task_count": task_count,
        "scheduler": scheduler,
        "nested_optimizer_ray_slots": int(getattr(args, "jobs", 1)),
    }
    if args.group == "expert_imbalance":
        payload["isolation_reason"] = (
            "factor and hot/cold traces currently share generated/search output paths; "
            "serial execution prevents file races"
        )
    return payload


def _run_isolated_trace_tasks(
    args: argparse.Namespace,
    worker: Any,
    payloads: Sequence[tuple[int, tuple[Any, ...]]],
    *,
    on_complete: Any | None = None,
) -> dict[int, Any]:
    """Run process-isolated trace jobs and return an index-keyed result map.

    Outer tasks reserve no Ray CPU while waiting for nested Pareto/Ideal work.
    Submission is capped explicitly, and max_calls=1 prevents module-level
    optimizer state from leaking between traces.
    """
    worker_count = _trace_worker_count(args, len(payloads))
    if worker_count == 1:
        completed: dict[int, Any] = {}
        for index, payload in payloads:
            result = worker(*payload)
            completed[index] = result
            if on_complete is not None:
                on_complete(index, result)
        return completed

    import ray

    if not ray.is_initialized():
        raise NativeExperimentError(
            "parallel trace execution requires the managed Ray runtime"
        )
    remote_worker = ray.remote(num_cpus=0, max_calls=1)(worker)
    pending: dict[Any, int] = {}
    completed = {}
    payload_iter = iter(payloads)

    def submit_one() -> bool:
        try:
            index, payload = next(payload_iter)
        except StopIteration:
            return False
        reference = remote_worker.remote(*payload)
        pending[reference] = index
        print(f"[native] started isolated trace {index}")
        return True

    for _ in range(worker_count):
        if not submit_one():
            break
    print(
        f"[native] running {len(payloads)} traces with "
        f"{worker_count} isolated workers ({args.jobs} Ray CPU slots)"
    )
    try:
        while pending:
            ready, _ = ray.wait(list(pending), num_returns=1)
            reference = ready[0]
            index = pending.pop(reference)
            result = ray.get(reference)
            completed[index] = result
            print(f"[native] completed isolated trace {index}")
            if on_complete is not None:
                on_complete(index, result)
            submit_one()
    except BaseException:
        for reference in pending:
            ray.cancel(reference, force=True, recursive=True)
        raise
    return completed


def _standard_cells(thresholds: Sequence[float]) -> list[tuple[str, float]]:
    return [("NoDVFS", 0.0)] + [
        (policy, threshold) for policy in POLICIES for threshold in thresholds
    ]


def _run_standard_trace_worker(
    args: argparse.Namespace,
    trace: PhaseTrace,
    thresholds: Sequence[float],
    prov: Mapping[str, Any],
    operator_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute one model/phase trace in an isolated process."""
    cells = _standard_cells(thresholds)
    batches: dict[str, dict[float, Analysis]] = {}
    if not args.quick:
        batches["DVFS-C"] = analyze_dvfsc_all_budgets(trace, thresholds)
        for policy in ("eNPU-C", "eNPU-All", "Ideal"):
            batches[policy] = analyze_policy_all_budgets(
                trace,
                policy,
                thresholds,
                allow_current_ideal=args.allow_current_ideal,
            )

    trace_records: list[dict[str, Any]] = []
    trace_ivr_records: list[dict[str, Any]] = []
    prefixes = (
        ("dvfs_sa", "sa"),
        ("dvfs_vu", "vu"),
        ("dvfs_sram", "sram"),
        ("dvfs_hbm_mc", "hbm_mc"),
        ("dvfs_hbm_die", "hbm_die"),
        ("dvfs_hbm_io", "hbm_io"),
        ("dvfs_ici_mc", "ici_mc"),
        ("dvfs_ici_phy", "ici_phy"),
    )
    for policy, threshold in cells:
        analysis = (
            batches[policy][threshold]
            if policy in batches
            else _analyze(args, trace, policy, threshold)
        )
        trace_records.append(_attach(analysis.record, prov, args.quick))
        metadata = {
            "model": trace.model,
            "model_id": trace.model_id,
            "phase": trace.phase,
            "policy": policy,
            "threshold_pct": threshold,
            "pg_strategy": "NoPG",
            "quick_smoke": args.quick,
        }
        path = legacy_csv_path(operator_root, trace, policy, threshold)
        write_operator_csv(path, analysis.ops, metadata=metadata)
        loss = 0.0
        total = 0.0
        for op in analysis.ops:
            count = op.stats.count
            total += op.stats.total_energy_J * count
            for dvfs_field, energy_field in prefixes:
                efficiency = (
                    getattr(op, dvfs_field).voltage_conversion_power_efficiency_percent
                    / 100.0
                )
                energy = getattr(op.stats, f"static_energy_{energy_field}_J")
                energy += getattr(op.stats, f"dynamic_energy_{energy_field}_J")
                loss += energy * (1.0 - efficiency) * count
        trace_ivr_records.append(
            _attach(
                {
                    **metadata,
                    **_planning_provenance(analysis),
                    "ivr_overhead_pct": 100.0 * loss / total if total else 0.0,
                },
                prov,
                args.quick,
            )
        )
    return trace_records, trace_ivr_records


def _run_standard_parallel(
    args: argparse.Namespace,
    paper: Mapping[str, Any],
    prov: Mapping[str, Any],
) -> None:
    """Run independent model/phase traces concurrently with parent-only commits."""
    input_tokens, output_tokens = _tokens(paper)
    thresholds = _thresholds(paper, "paper_sweep", args.quick)
    cells = _standard_cells(thresholds)
    expected: list[dict[str, Any]] = []
    operator_root = args.output_dir / "operator_records"
    checkpoint_path = args.output_dir / STANDARD_CHECKPOINT_NAME
    checkpoint_enabled = bool(getattr(args, "resume", False) and not args.quick)
    checkpoint_signature = (
        _standard_checkpoint_signature(args, paper, prov) if checkpoint_enabled else {}
    )
    checkpoint_entries = (
        _load_standard_checkpoint(checkpoint_path, checkpoint_signature)
        if checkpoint_enabled
        else {}
    )
    if checkpoint_enabled and checkpoint_path.is_file() and not checkpoint_entries:
        print(f"[native] ignoring stale or malformed checkpoint: {checkpoint_path}")

    trace_results: dict[int, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    trace_metadata: dict[int, tuple[PhaseTrace, list[tuple[str, float]], str]] = {}
    payloads: list[tuple[int, tuple[Any, ...]]] = []
    trace_index = 0
    for model_id in _selected_models(args, paper):
        for phase in _selected_phases(args):
            trace = _trace(args, paper, model_id, phase, input_tokens, output_tokens)
            expected.extend(
                {
                    "model_id": model_id,
                    "phase": phase,
                    "policy": policy,
                    "threshold_pct": threshold,
                }
                for policy, threshold in cells
            )
            checkpoint_key = f"{model_id}:{phase}"
            trace_metadata[trace_index] = (trace, cells, checkpoint_key)
            restored = (
                _restore_standard_trace(
                    checkpoint_entries[checkpoint_key],
                    trace=trace,
                    cells=cells,
                    operator_root=operator_root,
                    output_dir=args.output_dir,
                    prov=prov,
                )
                if checkpoint_enabled and checkpoint_key in checkpoint_entries
                else None
            )
            if restored is not None:
                trace_results[trace_index] = restored
                print(f"[native] resume: restored {model_id}/{phase} checkpoint")
            else:
                payloads.append(
                    (
                        trace_index,
                        (args, trace, thresholds, prov, operator_root),
                    )
                )
            trace_index += 1

    def complete(
        index: int,
        result: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    ) -> None:
        trace_results[index] = result
        if not checkpoint_enabled:
            return
        trace, trace_cells, checkpoint_key = trace_metadata[index]
        trace_records, trace_ivr_records = result
        checkpoint_entries[checkpoint_key] = _standard_trace_checkpoint_entry(
            trace=trace,
            cells=trace_cells,
            operator_root=operator_root,
            output_dir=args.output_dir,
            records=trace_records,
            ivr_records=trace_ivr_records,
        )
        _write_standard_checkpoint(
            checkpoint_path, checkpoint_signature, checkpoint_entries
        )
        print(
            f"[native] checkpointed {trace.model_id}/{trace.phase}: "
            f"{checkpoint_path}"
        )

    _run_isolated_trace_tasks(
        args,
        _run_standard_trace_worker,
        payloads,
        on_complete=complete,
    )

    records: list[dict[str, Any]] = []
    ivr_records: list[dict[str, Any]] = []
    for index in range(trace_index):
        if index not in trace_results:
            raise NativeExperimentError(
                f"parallel standard trace {index} did not finish"
            )
        trace_records, trace_ivr_records = trace_results[index]
        records.extend(trace_records)
        ivr_records.extend(trace_ivr_records)
    keys = ("model_id", "phase", "policy", "threshold_pct")
    _validate_cells(records, expected, keys)
    _validate_cells(ivr_records, expected, keys)
    atomic_json(args.output_dir / "energy_records.json", {"records": records})
    atomic_json(args.output_dir / "ivr_records.json", {"records": ivr_records})


def run_standard(
    args: argparse.Namespace, paper: Mapping[str, Any], prov: Mapping[str, Any]
) -> None:
    task_count = len(_selected_models(args, paper)) * len(_selected_phases(args))
    if _trace_worker_count(args, task_count) > 1:
        _run_standard_parallel(args, paper, prov)
        _annotate_standard_table_3_check(args.output_dir, paper)
        return

    input_tokens, output_tokens = _tokens(paper)
    thresholds = _thresholds(paper, "paper_sweep", args.quick)
    records: list[dict[str, Any]] = []
    ivr_records: list[dict[str, Any]] = []
    expected: list[dict[str, Any]] = []
    operator_root = args.output_dir / "operator_records"
    checkpoint_path = args.output_dir / STANDARD_CHECKPOINT_NAME
    checkpoint_enabled = bool(getattr(args, "resume", False) and not args.quick)
    checkpoint_signature = (
        _standard_checkpoint_signature(args, paper, prov) if checkpoint_enabled else {}
    )
    checkpoint_entries = (
        _load_standard_checkpoint(checkpoint_path, checkpoint_signature)
        if checkpoint_enabled
        else {}
    )
    if checkpoint_enabled and checkpoint_path.is_file() and not checkpoint_entries:
        print(f"[native] ignoring stale or malformed checkpoint: {checkpoint_path}")
    for model_id in _selected_models(args, paper):
        for phase in _selected_phases(args):
            trace = _trace(args, paper, model_id, phase, input_tokens, output_tokens)
            cells = [("NoDVFS", 0.0)] + [
                (policy, threshold) for policy in POLICIES for threshold in thresholds
            ]
            trace_expected = [
                {
                    "model_id": model_id,
                    "phase": phase,
                    "policy": policy,
                    "threshold_pct": threshold,
                }
                for policy, threshold in cells
            ]
            expected.extend(trace_expected)
            checkpoint_key = f"{model_id}:{phase}"
            restored = (
                _restore_standard_trace(
                    checkpoint_entries[checkpoint_key],
                    trace=trace,
                    cells=cells,
                    operator_root=operator_root,
                    output_dir=args.output_dir,
                    prov=prov,
                )
                if checkpoint_enabled and checkpoint_key in checkpoint_entries
                else None
            )
            if restored is not None:
                restored_records, restored_ivr = restored
                records.extend(restored_records)
                ivr_records.extend(restored_ivr)
                print(f"[native] resume: restored {model_id}/{phase} checkpoint")
                continue
            batches: dict[str, dict[float, Analysis]] = {}
            if not args.quick:
                batches["DVFS-C"] = analyze_dvfsc_all_budgets(trace, thresholds)
                for policy in ("eNPU-C", "eNPU-All", "Ideal"):
                    batches[policy] = analyze_policy_all_budgets(
                        trace,
                        policy,
                        thresholds,
                        allow_current_ideal=args.allow_current_ideal,
                    )
            trace_records: list[dict[str, Any]] = []
            trace_ivr_records: list[dict[str, Any]] = []
            for policy, threshold in cells:
                analysis = (
                    batches[policy][threshold]
                    if policy in batches
                    else _analyze(args, trace, policy, threshold)
                )
                trace_records.append(_attach(analysis.record, prov, args.quick))
                metadata = {
                    "model": trace.model,
                    "model_id": trace.model_id,
                    "phase": phase,
                    "policy": policy,
                    "threshold_pct": threshold,
                    "pg_strategy": "NoPG",
                    "quick_smoke": args.quick,
                }
                path = legacy_csv_path(operator_root, trace, policy, threshold)
                write_operator_csv(path, analysis.ops, metadata=metadata)
                loss = 0.0
                total = 0.0
                prefixes = (
                    ("dvfs_sa", "sa"),
                    ("dvfs_vu", "vu"),
                    ("dvfs_sram", "sram"),
                    ("dvfs_hbm_mc", "hbm_mc"),
                    ("dvfs_hbm_die", "hbm_die"),
                    ("dvfs_hbm_io", "hbm_io"),
                    ("dvfs_ici_mc", "ici_mc"),
                    ("dvfs_ici_phy", "ici_phy"),
                )
                for op in analysis.ops:
                    count = op.stats.count
                    total += op.stats.total_energy_J * count
                    for dvfs_field, energy_field in prefixes:
                        efficiency = (
                            getattr(
                                op, dvfs_field
                            ).voltage_conversion_power_efficiency_percent
                            / 100.0
                        )
                        energy = getattr(op.stats, f"static_energy_{energy_field}_J")
                        energy += getattr(op.stats, f"dynamic_energy_{energy_field}_J")
                        loss += energy * (1.0 - efficiency) * count
                trace_ivr_records.append(
                    _attach(
                        {
                            **metadata,
                            **_planning_provenance(analysis),
                            "ivr_overhead_pct": 100.0 * loss / total if total else 0.0,
                        },
                        prov,
                        args.quick,
                    )
                )
            records.extend(trace_records)
            ivr_records.extend(trace_ivr_records)
            if checkpoint_enabled:
                checkpoint_entries[checkpoint_key] = _standard_trace_checkpoint_entry(
                    trace=trace,
                    cells=cells,
                    operator_root=operator_root,
                    output_dir=args.output_dir,
                    records=trace_records,
                    ivr_records=trace_ivr_records,
                )
                _write_standard_checkpoint(
                    checkpoint_path, checkpoint_signature, checkpoint_entries
                )
                print(f"[native] checkpointed {model_id}/{phase}: {checkpoint_path}")
    keys = ("model_id", "phase", "policy", "threshold_pct")
    _validate_cells(records, expected, keys)
    _validate_cells(ivr_records, expected, keys)
    atomic_json(args.output_dir / "energy_records.json", {"records": records})
    atomic_json(args.output_dir / "ivr_records.json", {"records": ivr_records})
    _annotate_standard_table_3_check(args.output_dir, paper)


def _run_domain_trace_worker(
    args: argparse.Namespace,
    paper: Mapping[str, Any],
    prov: Mapping[str, Any],
    model_id: str,
    phase: str,
    input_tokens: int,
    output_tokens: int,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    modes = ("dom5", "dom4_savu", "dom3")
    trace = _trace(args, paper, model_id, phase, input_tokens, output_tokens)
    baseline = _analyze(args, trace, "NoDVFS", 0.0).record["total_energy_J"]
    batches = (
        {}
        if args.quick
        else {
            mode: analyze_policy_all_budgets(
                trace, "eNPU-All", thresholds, domain_mode=mode
            )
            for mode in modes
        }
    )
    rows = []
    for threshold in thresholds:
        for mode in modes:
            result = (
                batches[mode][threshold]
                if mode in batches
                else _analyze(args, trace, "eNPU-All", threshold, domain_mode=mode)
            )
            rows.append(
                _attach(
                    {
                        **result.record,
                        "mode": mode,
                        "energy_saving_pct": 100.0
                        * (1.0 - result.record["total_energy_J"] / baseline),
                    },
                    prov,
                    args.quick,
                )
            )
    return rows


def run_domain(
    args: argparse.Namespace, paper: Mapping[str, Any], prov: Mapping[str, Any]
) -> None:
    input_tokens, output_tokens = _tokens(paper)
    thresholds = _thresholds(paper, "domain_grid", args.quick)
    modes = ("dom5", "dom4_savu", "dom3")
    axes = [
        (model_id, phase)
        for model_id in _selected_models(args, paper)
        for phase in _selected_phases(args)
    ]
    payloads = [
        (
            index,
            (
                args,
                paper,
                prov,
                model_id,
                phase,
                input_tokens,
                output_tokens,
                thresholds,
            ),
        )
        for index, (model_id, phase) in enumerate(axes)
    ]
    completed = _run_isolated_trace_tasks(args, _run_domain_trace_worker, payloads)
    rows = [row for index in range(len(axes)) for row in completed[index]]
    expected = [
        {
            "model_id": model_id,
            "phase": phase,
            "threshold_pct": threshold,
            "mode": mode,
        }
        for model_id, phase in axes
        for threshold in thresholds
        for mode in modes
    ]
    _validate_cells(rows, expected, ("model_id", "phase", "threshold_pct", "mode"))
    atomic_json(args.output_dir / "domain_count_records.json", {"records": rows})


def _load_temporal_standard_reuse(
    args: argparse.Namespace,
    paper: Mapping[str, Any],
    prov: Mapping[str, Any],
) -> TemporalStandardReuse | None:
    source = getattr(args, "standard_sweep_energy_records", None)
    if source is None:
        return None
    if not source.is_file():
        raise NativeExperimentError(
            f"standard-sweep reuse input does not exist: {source}"
        )
    try:
        payload = load_json(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeExperimentError(
            f"cannot read standard-sweep reuse input {source}: {exc}"
        ) from exc
    source_rows = payload.get("records")
    if not isinstance(source_rows, list) or not all(
        isinstance(row, Mapping) for row in source_rows
    ):
        raise NativeExperimentError(
            f"standard-sweep reuse input has no valid records list: {source}"
        )

    models = _selected_models(args, paper)
    phases = _selected_phases(args)
    thresholds = _thresholds(paper, "paper_sweep", args.quick)
    input_tokens, output_tokens = _tokens(paper)
    model_labels = {model_id: MODEL_FILES[model_id] for model_id in models}
    expected = {
        (model_id, phase, policy, threshold)
        for model_id in models
        for phase in phases
        for policy in TEMPORAL_REUSED_POLICIES
        for threshold in ([0.0] if policy == "NoDVFS" else thresholds)
    }
    indexed: dict[tuple[str, str, str, float], dict[str, Any]] = {}
    for raw_row in source_rows:
        model_id = raw_row.get("model_id")
        phase = raw_row.get("phase")
        policy = raw_row.get("policy")
        if (
            model_id not in models
            or phase not in phases
            or policy not in TEMPORAL_REUSED_POLICIES
        ):
            continue
        key = _checkpoint_cell_key(raw_row)
        if key is None or key not in expected:
            raise NativeExperimentError(
                f"standard-sweep reuse input has unexpected cell: "
                f"{model_id}/{phase}/{policy}/{raw_row.get('threshold_pct')}"
            )
        if key in indexed:
            raise NativeExperimentError(
                f"standard-sweep reuse input has duplicate cell: {key}"
            )
        source_prov = raw_row.get("provenance")
        if (
            raw_row.get("model") != model_labels[model_id]
            or raw_row.get("input_tokens") != input_tokens
            or raw_row.get("output_tokens") != output_tokens
            or raw_row.get("pg_strategy") != "NoPG"
            or raw_row.get("quick_smoke") is not args.quick
            or not isinstance(raw_row.get("config"), str)
            or not isinstance(source_prov, Mapping)
            or source_prov.get("measurement_mode") != "standard_sweep"
            or source_prov.get("simulator_revision") != prov.get("simulator_revision")
            or source_prov.get("workspace_diff_sha256")
            != prov.get("workspace_diff_sha256")
            or source_prov.get("allow_current_ideal") != args.allow_current_ideal
        ):
            raise NativeExperimentError(
                f"standard-sweep reuse cell does not match this run: {key}"
            )
        indexed[key] = dict(raw_row)
    missing = sorted(expected - set(indexed), key=str)
    if missing:
        raise NativeExperimentError(
            f"standard-sweep reuse input is missing {len(missing)} required cells: "
            + ", ".join(str(key) for key in missing[:8])
        )
    return TemporalStandardReuse(
        source=source,
        source_sha256=_sha256(source),
        rows=indexed,
    )


def _temporal_reused_record(
    reuse: TemporalStandardReuse,
    *,
    trace: PhaseTrace,
    policy: str,
    threshold: float,
    prov: Mapping[str, Any],
    quick: bool,
) -> dict[str, Any]:
    key = (trace.model_id, trace.phase, policy, threshold)
    source_row = reuse.rows[key]
    if source_row.get("config") != trace.config_label:
        raise NativeExperimentError(
            f"standard-sweep reuse config mismatch for {key}: "
            f"{source_row.get('config')!r} != {trace.config_label!r}"
        )
    row = dict(source_row)
    source_measurement_provenance = row.pop("provenance")
    row.update(
        {
            "reused_from_standard_sweep": True,
            "computed_in_group": "standard_sweep",
            "reuse_provenance": {
                "source_group": "standard_sweep",
                "source_path": str(reuse.source),
                "source_sha256": reuse.source_sha256,
                "source_measurement_provenance": source_measurement_provenance,
                "validation": (
                    "exact model/phase/policy/threshold/token/config/quick/"
                    "revision/workspace match"
                ),
            },
        }
    )
    return _attach(row, prov, quick)


def _run_temporal_trace_worker(
    args: argparse.Namespace,
    paper: Mapping[str, Any],
    prov: Mapping[str, Any],
    reuse: TemporalStandardReuse | None,
    model_id: str,
    phase: str,
    input_tokens: int,
    output_tokens: int,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    policies = ("NoDVFS", "DVFS-C", "DVFS-C-ms", "eNPU-All", "eNPU-ms", "Ideal")
    trace = _trace(args, paper, model_id, phase, input_tokens, output_tokens)
    batches: dict[str, dict[float, Analysis]] = {}
    if not args.quick:
        batches["DVFS-C-ms"] = analyze_dvfsc_all_budgets(
            trace, thresholds, millisecond_regions=True, epoch_ns=5_000_000.0
        )
        batches["eNPU-ms"] = analyze_enpu_ms_all_budgets(
            trace, thresholds, epoch_ns=5_000_000.0
        )
        if reuse is None:
            batches["DVFS-C"] = analyze_dvfsc_all_budgets(trace, thresholds)
            batches["eNPU-All"] = analyze_policy_all_budgets(
                trace,
                "eNPU-All",
                thresholds,
            )
            batches["Ideal"] = analyze_policy_all_budgets(
                trace,
                "Ideal",
                thresholds,
                allow_current_ideal=args.allow_current_ideal,
            )
    rows = []
    for policy in policies:
        policy_thresholds = [0.0] if policy == "NoDVFS" else thresholds
        for threshold in policy_thresholds:
            if reuse is not None and policy in TEMPORAL_REUSED_POLICIES:
                row = _temporal_reused_record(
                    reuse,
                    trace=trace,
                    policy=policy,
                    threshold=threshold,
                    prov=prov,
                    quick=args.quick,
                )
            else:
                result = (
                    batches[policy][threshold]
                    if policy in batches
                    else _analyze(
                        args,
                        trace,
                        policy,
                        threshold,
                        epoch_ns=5_000_000.0,
                    )
                )
                row = _attach(
                    {
                        **result.record,
                        "reused_from_standard_sweep": False,
                        "computed_in_group": "temporal_granularity",
                    },
                    prov,
                    args.quick,
                )
            rows.append(row)
    return rows


def run_temporal(
    args: argparse.Namespace, paper: Mapping[str, Any], prov: Mapping[str, Any]
) -> None:
    input_tokens, output_tokens = _tokens(paper)
    thresholds = _thresholds(paper, "paper_sweep", args.quick)
    reuse = _load_temporal_standard_reuse(args, paper, prov)
    args.temporal_reuse_provenance = (
        {
            "enabled": True,
            "source_group": "standard_sweep",
            "source_path": str(reuse.source),
            "source_sha256": reuse.source_sha256,
            "reused_policies": list(TEMPORAL_REUSED_POLICIES),
            "computed_policies": list(TEMPORAL_COMPUTED_POLICIES),
            "reused_cell_count": len(reuse.rows),
        }
        if reuse is not None
        else {
            "enabled": False,
            "standalone_recomputation": True,
            "reused_policies": [],
            "computed_policies": [
                "NoDVFS",
                "DVFS-C",
                *TEMPORAL_COMPUTED_POLICIES,
                "eNPU-All",
                "Ideal",
            ],
        }
    )
    policies = ("NoDVFS", "DVFS-C", "DVFS-C-ms", "eNPU-All", "eNPU-ms", "Ideal")
    axes = [
        (model_id, phase)
        for model_id in _selected_models(args, paper)
        for phase in _selected_phases(args)
    ]
    payloads = [
        (
            index,
            (
                args,
                paper,
                prov,
                reuse,
                model_id,
                phase,
                input_tokens,
                output_tokens,
                thresholds,
            ),
        )
        for index, (model_id, phase) in enumerate(axes)
    ]
    completed = _run_isolated_trace_tasks(args, _run_temporal_trace_worker, payloads)
    rows = [row for index in range(len(axes)) for row in completed[index]]
    expected = [
        {
            "model_id": model_id,
            "phase": phase,
            "policy": policy,
            "threshold_pct": threshold,
        }
        for model_id, phase in axes
        for policy in policies
        for threshold in ([0.0] if policy == "NoDVFS" else thresholds)
    ]
    _validate_cells(rows, expected, ("model_id", "phase", "policy", "threshold_pct"))
    atomic_json(args.output_dir / "energy_records.json", {"records": rows})


def _run_sequence_trace_worker(
    args: argparse.Namespace,
    paper: Mapping[str, Any],
    prov: Mapping[str, Any],
    model_id: str,
    input_tokens: int,
    phase: str,
    output_tokens: int,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    worker_args = _token_trace_args(args, model_id, phase, input_tokens, output_tokens)
    trace = _trace(worker_args, paper, model_id, phase, input_tokens, output_tokens)
    batch = (
        {} if args.quick else analyze_policy_all_budgets(trace, "eNPU-All", thresholds)
    )
    rows = []
    rows.append(
        _attach(_analyze(worker_args, trace, "NoDVFS", 0).record, prov, args.quick)
    )
    for threshold in thresholds:
        result = (
            batch[threshold]
            if threshold in batch
            else _analyze(worker_args, trace, "eNPU-All", threshold)
        )
        rows.append(_attach(result.record, prov, args.quick))
    return rows


def run_sequences(
    args: argparse.Namespace, paper: Mapping[str, Any], prov: Mapping[str, Any]
) -> None:
    sequence = paper["sequence_lengths_tokens"]
    lengths = [int(value) for value in sequence["input_sweep"]]
    if args.quick:
        lengths = [lengths[0], int(sequence["default_input"])]
    output_tokens = int(sequence["fixed_output_for_sweep"])
    thresholds = _thresholds(paper, "figure_18", args.quick)
    axes = [
        (model_id, input_tokens, phase)
        for model_id in ("llama3_70b", "deepseekv3_671b")
        for input_tokens in lengths
        for phase in ("prefill", "decode")
    ]
    payloads = [
        (
            index,
            (
                args,
                paper,
                prov,
                model_id,
                input_tokens,
                phase,
                output_tokens,
                thresholds,
            ),
        )
        for index, (model_id, input_tokens, phase) in enumerate(axes)
    ]
    completed = _run_isolated_trace_tasks(args, _run_sequence_trace_worker, payloads)
    rows = [row for index in range(len(axes)) for row in completed[index]]
    expected = [
        {
            "model_id": model_id,
            "phase": phase,
            "input_tokens": input_tokens,
            "policy": policy,
            "threshold_pct": threshold,
        }
        for model_id, input_tokens, phase in axes
        for policy, threshold in [
            ("NoDVFS", 0.0),
            *(("eNPU-All", threshold) for threshold in thresholds),
        ]
    ]
    _validate_cells(
        rows, expected, ("model_id", "phase", "input_tokens", "policy", "threshold_pct")
    )
    atomic_json(args.output_dir / "energy_records.json", {"records": rows})


FIGURE20_DVFSC_GA_GENERATIONS = 20


def _cold_chip_trace(trace: PhaseTrace) -> PhaseTrace:
    ops = [
        op
        for op in trace.ops
        if "most_loaded" not in str(getattr(op, "description", "")).lower()
    ]
    if len(ops) == len(trace.ops):
        raise NativeExperimentError(
            "Figure 20 per-chip-load model found no most_loaded expert operators"
        )
    return PhaseTrace(
        model_id=trace.model_id,
        model=trace.model,
        phase=trace.phase,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        config=trace.config,
        ops=ops,
        config_label=trace.config_label,
    )


def _imbalance_search_batch(
    args: argparse.Namespace,
    trace: PhaseTrace,
    policy: str,
    thresholds: Sequence[float],
) -> dict[float, Analysis]:
    """Compile all Figure 20 thresholds once for one trace/chip class."""
    if args.quick:
        kwargs = (
            {"dvfsc_max_generations": FIGURE20_DVFSC_GA_GENERATIONS}
            if policy == "DVFS-C"
            else {}
        )
        return {
            threshold: _analyze(args, trace, policy, threshold, **kwargs)
            for threshold in thresholds
        }
    if policy == "DVFS-C":
        return analyze_dvfsc_all_budgets(
            trace,
            thresholds,
            population_size=200,
            max_generations=FIGURE20_DVFSC_GA_GENERATIONS,
            crossover_prob=0.8,
            mutation_prob=0.03,
            elitism_count=5,
            use_pareto_policy=True,
        )
    return analyze_policy_all_budgets(trace, policy, thresholds)


def _transplanted_energy(
    real: PhaseTrace,
    planned: Analysis,
    policy: str,
    threshold: float,
) -> float:
    schedule = transplant_modal_schedule(planned.ops, real.ops)
    measured = evaluate_configured_trace(
        real, schedule, policy=policy, threshold_pct=threshold
    )
    return float(measured.record["total_energy_J"])


def _cached_imbalance_row(
    args: argparse.Namespace,
    prov: Mapping[str, Any],
    real: PhaseTrace,
    factor: float,
    threshold: float,
    policy: str,
    baseline_hot: float,
    baseline_cold: float,
    real_batches: Mapping[str, Mapping[str, Mapping[float, Analysis]]],
    worst_batches: Mapping[str, Mapping[str, Mapping[float, Analysis]]],
) -> dict[str, Any]:
    planning = real_batches[policy]["hot"][threshold]
    hot = (
        baseline_hot,
        _transplanted_energy(real, planning, policy, threshold),
        _transplanted_energy(
            real, worst_batches[policy]["hot"][threshold], policy, threshold
        ),
    )
    cold_trace = _cold_chip_trace(real)
    cold = (
        baseline_cold,
        _transplanted_energy(
            cold_trace,
            real_batches[policy]["cold"][threshold],
            policy,
            threshold,
        ),
        _transplanted_energy(
            cold_trace,
            worst_batches[policy]["cold"][threshold],
            policy,
            threshold,
        ),
    )
    expert_parallelism = int(real.config.expert_parallelism_degree)
    num_chips = int(real.config.num_chips)
    if expert_parallelism < 2 or num_chips % expert_parallelism != 0:
        raise NativeExperimentError(
            "Figure 20 requires a chip count divisible across at least two expert groups"
        )
    n_hot = num_chips // expert_parallelism
    n_cold = num_chips - n_hot
    if n_hot <= 0 or n_cold <= 0:
        raise NativeExperimentError(
            "Figure 20 per-chip-load model requires both hot and cold expert groups"
        )
    baseline = n_hot * hot[0] + n_cold * cold[0]
    oracle = n_hot * hot[1] + n_cold * cold[1]
    worst_case = n_hot * hot[2] + n_cold * cold[2]
    saving_oracle = 100.0 * (1.0 - oracle / baseline)
    saving_wc = 100.0 * (1.0 - worst_case / baseline)
    achievable = baseline - oracle
    return _attach(
        {
            **_planning_provenance(planning),
            "model": real.model,
            "model_id": real.model_id,
            "phase": "prefill",
            "policy": policy,
            "real_f": factor,
            "wc_f": 32.0,
            "nwc": 8,
            "pd": threshold / 100.0,
            "pd_pct": threshold,
            "E_none_J": baseline,
            "E_oracle_J": oracle,
            "E_wc_J": worst_case,
            "saving_oracle_pct": saving_oracle,
            "saving_wc_pct": saving_wc,
            "gap_pp": saving_oracle - saving_wc,
            "lost_fraction": (
                (worst_case - oracle) / achievable if achievable > 0 else 0.0
            ),
            "per_chip_load": True,
            "num_chips": num_chips,
            "n_hot": n_hot,
            "n_cold": n_cold,
            "E_none_hot_J": hot[0],
            "E_none_cold_J": cold[0],
            "E_oracle_hot_J": hot[1],
            "E_oracle_cold_J": cold[1],
            "E_wc_hot_J": hot[2],
            "E_wc_cold_J": cold[2],
            "hot_expert_share_pct": (
                100.0 * (hot[0] - cold[0]) / hot[0] if hot[0] else 0.0
            ),
            "all_to_all_imbalance_aware": True,
            "all_to_all_model": "integrated receiver-skew/incast-aware generator",
            "ga_gens": (FIGURE20_DVFSC_GA_GENERATIONS if policy == "DVFS-C" else None),
            "trace_cache_reused": True,
            "all_budget_search": not args.quick,
            "schedule_transplant": (
                "count-weighted modal semantic node key; fusion-index independent"
            ),
        },
        prov,
        args.quick,
    )


def run_imbalance(
    args: argparse.Namespace, paper: Mapping[str, Any], prov: Mapping[str, Any]
) -> None:
    factors = [float(value) for value in paper["expert_capacity_factors"]]
    if args.quick:
        factors = [factors[0], factors[-1]]
    thresholds = [
        float(value)
        for value in paper["performance_degradation_thresholds_percent"][
            "figure_20_sweep"
        ]
    ]
    if args.quick:
        thresholds = [thresholds[0], thresholds[-1]]
    policies = ("DVFS-C", "eNPU-All")
    input_tokens, output_tokens = _tokens(paper)
    common = {"num_worst_case_experts": 8}

    # Trace generation is independent of policy and threshold. Generate each
    # capacity factor once; the factor-32 trace is also the provisioning trace.
    traces = {
        factor: _trace(
            args,
            paper,
            "deepseekv3_671b",
            "prefill",
            input_tokens,
            output_tokens,
            {**common, "expert_load_imbalance_factor": factor},
        )
        for factor in factors
    }
    if 32.0 not in traces:
        traces[32.0] = _trace(
            args,
            paper,
            "deepseekv3_671b",
            "prefill",
            input_tokens,
            output_tokens,
            {**common, "expert_load_imbalance_factor": 32.0},
        )
    cold_traces = {factor: _cold_chip_trace(trace) for factor, trace in traces.items()}
    worst = traces[32.0]
    worst_batches = {
        policy: {
            "hot": _imbalance_search_batch(args, worst, policy, thresholds),
            "cold": _imbalance_search_batch(
                args, cold_traces[32.0], policy, thresholds
            ),
        }
        for policy in policies
    }

    rows: list[dict[str, Any]] = []
    for factor in factors:
        real = traces[factor]
        real_cold = cold_traces[factor]
        baseline_hot = float(
            _analyze(args, real, "NoDVFS", 0.0).record["total_energy_J"]
        )
        baseline_cold = float(
            _analyze(args, real_cold, "NoDVFS", 0.0).record["total_energy_J"]
        )
        real_batches = (
            worst_batches
            if factor == 32.0
            else {
                policy: {
                    "hot": _imbalance_search_batch(args, real, policy, thresholds),
                    "cold": _imbalance_search_batch(
                        args, real_cold, policy, thresholds
                    ),
                }
                for policy in policies
            }
        )
        rows.extend(
            _cached_imbalance_row(
                args,
                prov,
                real,
                factor,
                threshold,
                policy,
                baseline_hot,
                baseline_cold,
                real_batches,
                worst_batches,
            )
            for threshold in thresholds
            for policy in policies
        )
    expected = [
        {"real_f": factor, "pd_pct": threshold, "policy": policy}
        for factor in factors
        for threshold in thresholds
        for policy in policies
    ]
    _validate_cells(rows, expected, ("real_f", "pd_pct", "policy"))
    atomic_json(args.output_dir / "expert_imbalance_records.json", {"records": rows})


def _run_pg_trace_worker(
    args: argparse.Namespace,
    paper: Mapping[str, Any],
    prov: Mapping[str, Any],
    phase: str,
    input_tokens: int,
    output_tokens: int,
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    rows = []
    # The paper text is ambiguous; the artifact fixes Figure 21 to Llama3-70B.
    model_id = "llama3_70b"
    trace = _trace(args, paper, model_id, phase, input_tokens, output_tokens)

    baseline = _analyze(args, trace, "NoDVFS", 0.0, pg_strategy="NoPG")
    baseline_schedule = _schedule_sha256(baseline.ops)
    baseline.record.update(
        {
            "schedule_sha256": baseline_schedule,
            "schedule_origin": "planned_once_under_NoPG",
        }
    )
    rows.append(_attach(baseline.record, prov, args.quick))

    pg_only = evaluate_configured_trace(
        trace, baseline.ops, policy="NoDVFS", threshold_pct=0.0, pg_strategy="Full"
    )
    if _schedule_sha256(pg_only.ops) != baseline_schedule:
        raise NativeExperimentError("PG-only evaluation changed the NoDVFS schedule")
    pg_only.record.update(
        {
            "schedule_sha256": baseline_schedule,
            "schedule_origin": "NoDVFS_NoPG_schedule_remeasured_with_Full_ReGate",
        }
    )
    rows.append(_attach(pg_only.record, prov, args.quick))

    planned_batch = (
        {}
        if args.quick
        else analyze_policy_all_budgets(
            trace,
            "eNPU-All",
            thresholds,
            pg_strategy="NoPG",
        )
    )
    for threshold in thresholds:
        planned = (
            planned_batch[threshold]
            if threshold in planned_batch
            else _analyze(args, trace, "eNPU-All", threshold, pg_strategy="NoPG")
        )
        schedule = _schedule_sha256(planned.ops)
        planned.record.update(
            {
                "schedule_sha256": schedule,
                "schedule_origin": "eNPU-All_planned_once_under_NoPG",
            }
        )
        rows.append(_attach(planned.record, prov, args.quick))

        combined = evaluate_configured_trace(
            trace,
            planned.ops,
            policy="eNPU-All",
            threshold_pct=threshold,
            pg_strategy="Full",
        )
        if _schedule_sha256(combined.ops) != schedule:
            raise NativeExperimentError(
                "ReGate evaluation changed the eNPU-All schedule"
            )
        combined.record.update(
            {
                "schedule_sha256": schedule,
                "schedule_origin": "eNPU-All_NoPG_schedule_remeasured_with_Full_ReGate",
            }
        )
        rows.append(_attach(combined.record, prov, args.quick))
    return rows


def run_pg(
    args: argparse.Namespace, paper: Mapping[str, Any], prov: Mapping[str, Any]
) -> None:
    input_tokens, output_tokens = _tokens(paper)
    thresholds = _thresholds(paper, "figure_21", args.quick)
    phases = ("prefill", "decode")
    payloads = [
        (
            index,
            (args, paper, prov, phase, input_tokens, output_tokens, thresholds),
        )
        for index, phase in enumerate(phases)
    ]
    completed = _run_isolated_trace_tasks(args, _run_pg_trace_worker, payloads)
    rows = [row for index in range(len(phases)) for row in completed[index]]
    expected = [
        {
            "phase": phase,
            "policy": policy,
            "pg_strategy": pg_strategy,
            "threshold_pct": threshold,
        }
        for phase in phases
        for policy, pg_strategy, threshold in [
            ("NoDVFS", "NoPG", 0.0),
            ("NoDVFS", "Full", 0.0),
            *(
                value
                for threshold in thresholds
                for value in (
                    ("eNPU-All", "NoPG", threshold),
                    ("eNPU-All", "Full", threshold),
                )
            ),
        ]
    ]

    _validate_cells(rows, expected, ("phase", "policy", "pg_strategy", "threshold_pct"))
    atomic_json(args.output_dir / "energy_records.json", {"records": rows})


def _validate_cells(
    actual: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> None:
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row.get(field) for field in keys)

    actual_keys = {key(row) for row in actual}
    expected_keys = {key(row) for row in expected}
    missing = sorted(expected_keys - actual_keys, key=str)
    if missing:
        preview = ", ".join(str(value) for value in missing[:8])
        raise NativeExperimentError(
            f"experiment matrix is missing {len(missing)} cells: {preview}"
        )


def _top_level_provenance(
    args: argparse.Namespace,
    paper: Mapping[str, Any],
    prov: Mapping[str, Any],
) -> None:
    pdf_available = args.paper_pdf is not None and args.paper_pdf.is_file()
    implementation = (
        "mixed_native_and_authoritative_trace_util_ms_ports"
        if args.group == "temporal_granularity"
        else "native_neusim"
    )
    payload = {
        "schema_version": 1,
        **prov,
        "group": args.group,
        "group_implementation": implementation,
        "group_policy_implementations": (
            {
                "NoDVFS": "native_neusim",
                "DVFS-C": "native_neusim",
                "DVFS-C-ms": "authoritative_trace_util_region_port",
                "eNPU-ms": "authoritative_trace_util_port",
                "eNPU-All": "native_neusim",
                "Ideal": "native_neusim_bounded_reduction",
            }
            if args.group == "temporal_granularity"
            else None
        ),
        "quick_smoke": args.quick,
        "allow_current_ideal": args.allow_current_ideal,
        "paper_manifest": str(args.paper_manifest),
        "paper_manifest_sha256": _sha256(args.paper_manifest),
        "paper_pdf": str(args.paper_pdf) if args.paper_pdf else None,
        "paper_pdf_sha256": _sha256(args.paper_pdf) if pdf_available else None,
        "paper_pdf_status": (
            "hashed"
            if pdf_available
            else "not supplied or auto-detected; hash unavailable"
        ),
        "selected_axes": {
            "models": _selected_models(args, paper),
            "phases": list(_selected_phases(args)),
            "default_tokens": list(_tokens(paper)),
        },
        "table_3_configuration_contract": _table_3_configuration_contract(paper),
        "trace_parallelism": _trace_parallelism_provenance(args, paper),
    }
    if args.group == "temporal_granularity":
        payload["authoritative_ms_source"] = dict(AUTHORITATIVE_MS_SOURCE)
        reuse = getattr(args, "temporal_reuse_provenance", None)
        if reuse is not None:
            payload["standard_sweep_reuse"] = reuse
            if reuse.get("enabled"):
                for policy in TEMPORAL_REUSED_POLICIES:
                    payload["group_policy_implementations"][
                        policy
                    ] = "validated_standard_sweep_reuse"
    atomic_json(args.output_dir / "provenance.json", payload)


def build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parents[1]
    repo = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, choices=GROUPS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        choices=tuple(MODEL_FILES),
        help=(
            "repeatable model selector for standard_sweep, domain_count, and "
            "temporal_granularity; omitted selects the normal full/quick matrix"
        ),
    )
    parser.add_argument(
        "--phase",
        dest="phases",
        action="append",
        choices=PHASES,
        help=(
            "repeatable phase selector for standard_sweep, domain_count, and "
            "temporal_granularity; omitted selects prefill and decode"
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument(
        "--paper-manifest",
        type=Path,
        default=here / "config" / "paper_experiments.json",
    )
    parser.add_argument(
        "--pipeline-manifest", type=Path, default=here / "config" / "pipeline.json"
    )
    parser.add_argument(
        "--standard-sweep-energy-records",
        type=Path,
        help=(
            "optional completed standard_sweep energy_records.json; temporal_granularity "
            "validates and reuses matching native cells"
        ),
    )
    parser.add_argument(
        "--paper-pdf", type=Path, help="optional paper PDF to hash in provenance"
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Ray CPU slots used by native optimizer fan-out and recorded in provenance",
    )
    parser.add_argument(
        "--trace-workers",
        "--standard-trace-workers",
        dest="trace_workers",
        type=int,
        default=None,
        help=(
            "isolated outer trace workers for safe groups; defaults to min(4, jobs). "
            "The legacy --standard-trace-workers spelling remains accepted"
        ),
    )
    parser.add_argument(
        "--verbose-simulator",
        action="store_true",
        help="show verbose operator-generator output",
    )
    parser.add_argument(
        "--allow-current-ideal",
        "--allow-large-exact",
        dest="allow_current_ideal",
        action="store_true",
        help=(
            "allow the current bounded/reduced Ideal request search; the legacy "
            "--allow-large-exact spelling is accepted but does not enumerate the "
            "full 48.1M-state theoretical lattice"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="actual NeuSim smoke measurement on a reduced matrix",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "enable signed per-model/phase checkpoints for full native standard_sweep"
        ),
    )
    return parser


def _resolve_trace_workers(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if args.trace_workers is not None and args.trace_workers < 1:
        parser.error("--trace-workers must be at least 1")
    if args.group not in PARALLEL_TRACE_GROUPS:
        if args.trace_workers not in (None, 1):
            parser.error(
                f"{args.group} is intentionally isolated and requires --trace-workers=1"
            )
        args.configured_trace_worker_cap = 1
        args.trace_workers = 1
        return
    configured_cap = min(args.trace_workers or DEFAULT_TRACE_WORKERS, args.jobs)
    args.configured_trace_worker_cap = configured_cap
    args.trace_workers = 1 if args.quick else configured_cap


def _validate_scope_selection(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    """Validate optional model/phase filters before starting Ray or writing output."""
    for option, values in (
        ("--model", tuple(args.models or ())),
        ("--phase", tuple(args.phases or ())),
    ):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            parser.error(f"{option} repeats selection(s): {', '.join(duplicates)}")
    if (args.models or args.phases) and args.group not in SCOPED_GROUPS:
        parser.error(
            "--model/--phase are only supported for "
            + ", ".join(sorted(SCOPED_GROUPS))
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.repo_root = args.repo_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.paper_manifest = args.paper_manifest.expanduser().resolve()
    args.pipeline_manifest = args.pipeline_manifest.expanduser().resolve()
    args.standard_sweep_energy_records = (
        args.standard_sweep_energy_records.expanduser().resolve()
        if args.standard_sweep_energy_records
        else None
    )
    args.paper_pdf = args.paper_pdf.expanduser().resolve() if args.paper_pdf else None
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    _validate_scope_selection(args, parser)
    _resolve_trace_workers(args, parser)
    if not args.quick and os.environ.get("DVFS_GA_VECTORIZED") == "1":
        parser.error("full artifact runs require the exact scalar recovered DVFS-C GA")
    if args.paper_pdf is None:
        candidate = args.repo_root.parent / "eNPU_micro26ae" / "NPU_DVFS_paper.pdf"
        args.paper_pdf = candidate if candidate.is_file() else None
    managed_ray = False
    try:
        if args.group in RAY_GROUPS:
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                os.environ.setdefault(name, "1")
            import ray

            if not ray.is_initialized():
                ray.init(
                    num_cpus=args.jobs,
                    include_dashboard=False,
                    log_to_driver=args.verbose_simulator,
                )
                managed_ray = True
        paper = load_json(args.paper_manifest)
        _table_3_configuration_contract(paper)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        prov = {
            **provenance(args.repo_root, mode=args.group, quick=args.quick),
            "host": _host_provenance(args),
            "allow_current_ideal": args.allow_current_ideal,
            "ideal_search_backend": _ideal_search_provenance(),
        }
        runners = {
            "standard_sweep": run_standard,
            "domain_count": run_domain,
            "temporal_granularity": run_temporal,
            "fixed_sequence_sweep": run_sequences,
            "expert_imbalance": run_imbalance,
            "power_gating": run_pg,
        }
        runners[args.group](args, paper, prov)
        _top_level_provenance(args, paper, prov)
        print(f"[native] completed {args.group}: {args.output_dir}")
        return 0
    except (NativeExperimentError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    finally:
        if managed_ray:
            import ray

            ray.shutdown()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
