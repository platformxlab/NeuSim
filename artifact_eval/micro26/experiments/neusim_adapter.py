"""Small compatibility layer between the artifact matrix and NeuSim APIs.

This module intentionally contains all knowledge of the evolving DVFS optimizer
interface.  Experiment code deals only in generated phase traces and normalized
records.  Measurements are always produced by the checked-out NeuSim; paper
reference values never enter the simulation path.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib
import inspect
import io
import json
import math
import re
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class NativeExperimentError(RuntimeError):
    """A fidelity or configuration problem that the caller can act on."""


POLICY_VALUES = {
    "NoDVFS": "None",
    "DVFS-C": "DVFSC",
    "DVFS-C-ms": "DVFSCms",
    "eNPU-C": "Custom",
    "eNPU-All": "CustomAll",
    "eNPU-ms": "CustomAllms",
    "Ideal": "Ideal",
}

# Current Ideal first keeps each voltage extrema. If that still exceeds this
# limit, a second deterministic balanced sampling stage preserves the true V/f
# endpoints in each domain and enforces the cap before lazy enumeration. Keep
# this separate from ``current_state_space()``, which reports the unreduced,
# theoretical table cardinality.
CURRENT_IDEAL_RAW_CONFIG_LIMIT = 2_000_000

CURRENT_IDEAL_SEARCH_EXECUTION = {
    "candidate_batch_size_default": 128,
    "maximum_default_inflight_batches": 24,
    "candidate_batch_size_environment": "DVFS_PARETO_BATCH_SIZE",
    "inflight_batches_environment": "DVFS_PARETO_MAX_INFLIGHT_BATCHES",
    "batch_tasks_per_operator_at_cap": 15_625,
    "standard_shared_envelope_audit_candidates": 172_107_043,
    "audit_analysis_rate_candidates_per_second": 5_100,
    "audit_estimated_ideal_hours": 9.4,
}


def _compact_ideal_pareto_generation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    searches = value.get("operator_searches")
    if not isinstance(searches, list):
        searches = []
    reductions = [
        search.get("config_reduction", {})
        for search in searches
        if isinstance(search, Mapping)
    ]
    return {
        "outer_scheduler": value.get("outer_scheduler"),
        "operator_count": len(searches),
        "total_analyzed_candidates": int(
            value.get(
                "total_analyzed_candidates",
                sum(
                    int(search.get("num_analyzed_candidates", 0))
                    for search in searches
                    if isinstance(search, Mapping)
                ),
            )
        ),
        "total_batch_tasks": int(
            value.get(
                "total_batch_tasks",
                sum(
                    int(search.get("submitted_batches", 0))
                    for search in searches
                    if isinstance(search, Mapping)
                ),
            )
        ),
        "operators_with_balanced_endpoint_cap": sum(
            bool(reduction.get("balanced_endpoint_cap_applied"))
            for reduction in reductions
            if isinstance(reduction, Mapping)
        ),
        "maximum_final_candidate_product": max(
            (
                int(reduction.get("final_candidate_product", 0))
                for reduction in reductions
                if isinstance(reduction, Mapping)
            ),
            default=0,
        ),
        "maximum_inflight_candidates": max(
            (
                int(search.get("max_inflight_candidates", 0))
                for search in searches
                if isinstance(search, Mapping)
            ),
            default=0,
        ),
    }


# Keep the recovered millisecond-policy source identity next to the compatibility
# adapter so every normalized record is directly auditable against the
# authoritative trace-util files.
AUTHORITATIVE_MS_SOURCE = {
    "repository_commit": "8ad6961b2a266e91ebb1162c7e2c5df61d10b1a4",
    "dvfs_enpu_ms_sha256": (
        "5dac84fa466e1270ec9b2d9909cc5e1221bd9b517f0d2038fc3363251230c112"
    ),
    "dvfs_region_merge_sha256": (
        "4e7433fc726dcf9ae902c9441a353d919a19baaa43a42c07c89b3a916233f86c"
    ),
    "test_dvfs_enpu_ms_sha256": (
        "223a9eae47ac83a8579738c2b21921cb32a58f107854140655823a35e8432992"
    ),
}

MODEL_FILES = {
    "llama3_70b": "llama3-70b",
    "llama3_1_405b": "llama3_1-405b",
    "deepseekv2_236b": "deepseekv2-236b",
    "deepseekv3_671b": "deepseekv3-671b",
}

GA_PROVENANCE_FIELDS = (
    "ga_execution_mode",
    "ga_exact_batch_size",
    "ga_exact_batch_size_env",
)

MS_CANDIDATE_BATCH_PROVENANCE_FIELDS = (
    "candidate_evaluation_mode",
    "candidate_batch_size",
    "candidate_batch_size_env",
    "candidate_count",
    "submitted_candidate_tasks",
    "candidate_result_order",
)


def _ms_candidate_batch_provenance(timing: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: timing[field]
        for field in MS_CANDIDATE_BATCH_PROVENANCE_FIELDS
        if field in timing
    }


@dataclass(frozen=True)
class PhaseTrace:
    model_id: str
    model: str
    phase: str
    input_tokens: int
    output_tokens: int
    config: Any
    ops: list[Any]
    config_label: str


@dataclass(frozen=True)
class Analysis:
    ops: list[Any]
    record: dict[str, Any]
    timing: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise NativeExperimentError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def repo_revision(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def workspace_diff(repo_root: Path) -> tuple[bool | None, str]:
    """Hash tracked diffs plus untracked content for reproducible dirty runs."""
    try:
        status = subprocess.run(
            ("git", "status", "--porcelain=v1", "-z"),
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
        diff = subprocess.run(
            ("git", "diff", "--binary", "HEAD"),
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout
        untracked = subprocess.run(
            ("git", "ls-files", "--others", "--exclude-standard", "-z"),
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError):
        return None, "unknown"

    digest = hashlib.sha256()
    digest.update(b"git-status\0")
    digest.update(status)
    digest.update(b"git-diff-head\0")
    digest.update(diff)
    for encoded in sorted(path for path in untracked if path):
        digest.update(b"untracked-path\0")
        digest.update(encoded)
        candidate = repo_root / encoded.decode("utf-8", errors="surrogateescape")
        if candidate.is_file():
            digest.update(b"\0untracked-content\0")
            try:
                with candidate.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            except OSError:
                digest.update(b"<unreadable>")
    return bool(status), digest.hexdigest()


def provenance(repo_root: Path, *, mode: str, quick: bool) -> dict[str, Any]:
    dirty, diff_sha256 = workspace_diff(repo_root)
    return {
        "producer": "NeuSim MICRO26 native experiment driver",
        "simulator": "NeuSim",
        "simulator_revision": repo_revision(repo_root),
        "workspace_dirty": dirty,
        "workspace_diff_sha256": diff_sha256,
        "measurement_mode": mode,
        "matrix_mode": "quick_smoke" if quick else "paper_matrix",
        "result_interpretation": "current_neusim_measurement",
        "paper_values_used_as_simulator_output": False,
    }


def _model_entry(paper: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    for entry in paper["models"]:
        if entry["id"] == model_id:
            return entry
    raise NativeExperimentError(f"model {model_id!r} is not in the paper manifest")


def build_phase_trace(
    *,
    repo_root: Path,
    paper: Mapping[str, Any],
    model_id: str,
    phase: str,
    input_tokens: int,
    output_tokens: int,
    output_hint: Path,
    config_overrides: Mapping[str, Any] | None = None,
    verbose_simulator: bool = False,
) -> PhaseTrace:
    """Generate one prefill or decode trace with the manifest parallelism."""
    from neusim.npusim.frontend.llm_ops_generator import (
        DeepSeekOpsGenerator,
        LLMOpsGeneratorInference,
    )
    from neusim.npusim.frontend.run_sim_lib import map_parallelism_to_ici_axes

    entry = _model_entry(paper, model_id)
    model = MODEL_FILES[model_id]
    parallelism = entry["phases"][phase]["parallelism"]
    dp = int(parallelism["data"])
    tp = int(parallelism["tensor"])
    pp = int(parallelism["pipeline"])
    ep = int(parallelism.get("expert", 1))
    chips = int(entry["phases"][phase]["chips"])

    configs = repo_root / "configs"
    merged: dict[str, Any] = {}
    for path in (
        configs / "models" / f"{model}.json",
        configs / "chips" / "tpuv5p.json",
        configs / "systems" / "system_config.json",
    ):
        merged.update(load_json(path))
    merged.update(
        {
            "model_name": model,
            "input_seqlen": int(input_tokens),
            "output_seqlen": int(output_tokens),
            "global_batch_size": 1,
            "microbatch_size_ici": 1,
            "microbatch_size_dcn": 1,
            "num_chips": chips,
            "data_parallelism_degree": dp,
            "tensor_parallelism_degree": tp,
            "pipeline_parallelism_degree": pp,
            "data_parallel_degree_dcn": 1,
            "tensor_parallel_degree_dcn": 1,
            "pipeline_parallel_degree_dcn": 1,
            "output_file_path": str(output_hint),
        }
    )
    if "deepseek" in model:
        merged.update(
            {
                "expert_parallelism_degree": ep,
                "expert_parallel_degree_dcn": 1,
            }
        )
    if config_overrides:
        merged.update(dict(config_overrides))

    parallelism_config = {
        "data_parallelism_degree": dp,
        "tensor_parallelism_degree": tp,
        "pipeline_parallelism_degree": pp,
    }
    if "deepseek" in model:
        parallelism_config["expert_parallelism_degree"] = ep
    axes = map_parallelism_to_ici_axes(model, "5p", parallelism_config)
    merged["num_data_parallel_axes"] = axes[0]
    merged["num_tensor_parallel_axes"] = axes[1]
    merged["num_pipeline_parallel_axes"] = axes[2]
    if "deepseek" in model:
        merged["num_expert_parallel_axes"] = axes[3]

    generator = (
        DeepSeekOpsGenerator(merged)
        if "deepseek" in model
        else LLMOpsGeneratorInference(merged)
    )
    output_context = (
        contextlib.nullcontext()
        if verbose_simulator
        else contextlib.redirect_stdout(io.StringIO())
    )
    with output_context:
        _all, prefill, decode = generator.generate(
            dump_to_file=False, separate_prefill_decode=True, analyze_energy=False
        )
    selected = prefill if phase == "prefill" else decode
    label = f"dp{dp}-tp{tp}-pp{pp}-ep{ep}-chips{chips}-b1"
    return PhaseTrace(
        model_id=model_id,
        model=model,
        phase=phase,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        config=generator.config,
        ops=selected,
        config_label=label,
    )


def _optimizer() -> Any | None:
    """Return the integrated request-level optimizer when present."""
    for module_name in (
        "neusim.npusim.frontend.dvfs_optimizer",
        "neusim.npusim.frontend.power_management_config_lib",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        function = getattr(module, "configure_dvfs_for_ops", None)
        if callable(function):
            return function
    return None


def has_ms_region_support() -> bool:
    return all(
        importlib.util.find_spec(name) is not None
        for name in (
            "neusim.npusim.frontend.dvfs_region_merge",
            "neusim.npusim.frontend.dvfs_enpu_ms",
        )
    )


def _call_supported(function: Any, *args: Any, **kwargs: Any) -> Any:
    parameters = inspect.signature(function).parameters
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return function(*args, **accepted)


def analyze_trace(
    trace: PhaseTrace,
    *,
    policy: str,
    threshold_pct: float,
    pg_strategy: str = "NoPG",
    allow_operator_local_fallback: bool,
    allow_current_ideal: bool = False,
    allow_large_exact: bool | None = None,
    domain_mode: str = "dom5",
    epoch_ns: float = 5_000_000.0,
    dvfsc_max_generations: int | None = None,
) -> Analysis:
    """Configure and evaluate a phase trace, returning normalized measurements.

    Full paper-matrix runs require the request-level optimizer for nonzero
    degradation budgets.  Quick smoke runs may exercise the component policy
    locally; those records are prominently tagged and are never described as a
    paper reproduction.
    """
    from neusim.npusim.frontend import power_analysis_lib as power_lib
    from neusim.npusim.frontend.Operator import DVFSConfig, DVFSPolicy

    # Compatibility for callers written before the opt-in was named according
    # to what the backend actually does. This alias does not disable or raise
    # the backend's 2M raw-candidate reduction threshold.
    if allow_large_exact is not None:
        allow_current_ideal = allow_current_ideal or allow_large_exact

    if policy not in POLICY_VALUES:
        raise NativeExperimentError(f"unknown normalized policy {policy!r}")
    if policy in {"DVFS-C-ms", "eNPU-ms"} and not has_ms_region_support():
        raise NativeExperimentError(
            f"{policy} requires the integrated dvfs_region_merge and dvfs_enpu_ms modules"
        )
    dvfs = DVFSConfig(
        policy=DVFSPolicy.from_str(POLICY_VALUES[policy]),
        performance_degradation_percentage=float(threshold_pct) / 100.0,
        frequency_adjustment_interval_ns=float(epoch_ns),
        custom_compute_domain_mode=domain_mode,
    )
    ops = deepcopy(trace.ops)
    analysis_config = deepcopy(trace.config)
    analysis_config.enable_dvfs = True
    # Quick runs deliberately exercise the cheap operator-local path even when
    # the request optimizer is installed. NoDVFS never needs request search.
    optimizer = (
        None if allow_operator_local_fallback or policy == "NoDVFS" else _optimizer()
    )
    timing: dict[str, Any] = {}
    start = time.perf_counter()

    requires_request_search = policy != "NoDVFS" and float(threshold_pct) > 0
    search_algorithm = "auto"
    if optimizer is not None and policy == "Ideal" and not allow_current_ideal:
        theoretical_product = int(current_state_space()["exact_product"])
        if theoretical_product > 10_000_000:
            raise NativeExperimentError(
                "the current Ideal request search is intentionally opt-in. The "
                f"unreduced NeuSim frequency-table lattice has {theoretical_product:,} "
                "theoretical combinations, but the current backend does not enumerate "
                "that complete lattice: after per-operator budget filtering, raw "
                f"candidate products above {CURRENT_IDEAL_RAW_CONFIG_LIMIT:,} are "
                "reduced to per-voltage frequency extrema and, if still above the "
                "limit, deterministically balanced while preserving true V/f endpoints "
                "before lazy enumeration. Rerun "
                "with --allow-current-ideal to authorize this potentially expensive "
                "bounded/reduced implementation, or use --quick for a visibly labeled "
                "operator-local smoke test."
            )
    if optimizer is not None:
        selected_optimizer = optimizer
        optimizer_kwargs: dict[str, Any] = {}
        if (
            policy == "DVFS-C"
            and threshold_pct > 0
            and dvfsc_max_generations is not None
        ):
            from neusim.npusim.frontend.dvfs_optimizer import (
                configure_dvfs_c_with_degradation,
            )

            selected_optimizer = configure_dvfs_c_with_degradation
            optimizer_kwargs["max_generations"] = int(dvfsc_max_generations)
            search_algorithm = f"DVFSC_GA_{int(dvfsc_max_generations)}_generations"
        configured = _call_supported(
            selected_optimizer,
            ops,
            analysis_config,
            dvfs,
            dump_pareto_points_to_file=False,
            algorithm=search_algorithm,
            pg_config=pg_strategy,
            timing_result=timing,
            **optimizer_kwargs,
        )
        if configured is not None:
            ops = configured
        execution_mode = "request_optimizer"
    elif requires_request_search and not allow_operator_local_fallback:
        raise NativeExperimentError(
            "the integrated request-level DVFS optimizer is unavailable; refusing "
            f"to ignore the {threshold_pct:g}% budget for {policy}"
        )
    else:
        for op in ops:
            power_lib.configure_dvfs_for_op(op, analysis_config, dvfs)
        execution_mode = "operator_local_fallback"

    configuration_seconds = time.perf_counter() - start
    timing.setdefault("configuration_wall_seconds", configuration_seconds)
    timing.setdefault("pareto_generation_seconds", 0.0)
    timing.setdefault("inter_op_search_seconds", configuration_seconds)
    timing["execution_mode"] = execution_mode
    timing["intra_op_algorithm"] = (
        "current_ideal_bounded_reduction"
        if execution_mode == "request_optimizer" and policy == "Ideal"
        else search_algorithm
        if execution_mode == "request_optimizer"
        else "operator_local"
    )
    if execution_mode == "request_optimizer" and policy == "Ideal":
        timing.update(
            {
                "ideal_raw_candidate_reduction_threshold": CURRENT_IDEAL_RAW_CONFIG_LIMIT,
                "ideal_theoretical_lattice_states": int(
                    current_state_space()["exact_product"]
                ),
                "ideal_full_theoretical_lattice_enumerated": False,
                "ideal_reduction_semantics": (
                    "per-voltage extrema followed when necessary by a deterministic "
                    "balanced endpoint-preserving cap"
                ),
                "ideal_search_execution": dict(CURRENT_IDEAL_SEARCH_EXECUTION),
                "ideal_pareto_generation_summary": _compact_ideal_pareto_generation(
                    timing.get("pareto_generation")
                ),
            }
        )

    # The optimizer only selects V/f.  Evaluate every returned operator once.
    for op in ops:
        power_lib.analyze_operator_energy(
            op,
            analysis_config,
            pg_config=pg_strategy,
            dvfs_config=dvfs,
            set_dvfs_config_for_op=False,
        )
    record = summarize(
        ops,
        model=trace.model,
        model_id=trace.model_id,
        phase=trace.phase,
        config=trace.config_label,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        policy=policy,
        threshold_pct=threshold_pct,
        pg_strategy=pg_strategy,
    )
    record["dvfs_execution_mode"] = execution_mode
    record["intra_op_algorithm"] = timing["intra_op_algorithm"]
    for field in GA_PROVENANCE_FIELDS:
        if field in timing:
            record[field] = timing[field]
    if execution_mode == "request_optimizer" and policy == "Ideal":
        record.update(
            {
                "ideal_raw_candidate_reduction_threshold": CURRENT_IDEAL_RAW_CONFIG_LIMIT,
                "ideal_theoretical_lattice_states": int(
                    current_state_space()["exact_product"]
                ),
                "ideal_full_theoretical_lattice_enumerated": False,
                "ideal_reduction_semantics": timing["ideal_reduction_semantics"],
                "ideal_search_execution": dict(CURRENT_IDEAL_SEARCH_EXECUTION),
                "ideal_pareto_generation_summary": timing[
                    "ideal_pareto_generation_summary"
                ],
            }
        )
    if policy == "DVFS-C" and dvfsc_max_generations is not None:
        record["ga_gens"] = int(dvfsc_max_generations)
    record["domain_mode"] = domain_mode
    if policy.endswith("-ms"):
        record["epoch_ns"] = epoch_ns
    return Analysis(ops=ops, record=record, timing=timing)


def evaluate_configured_trace(
    trace: PhaseTrace,
    ops: Sequence[Any],
    *,
    policy: str,
    threshold_pct: float,
    pg_strategy: str = "NoPG",
    epoch_ns: float = 5_000_000.0,
) -> Analysis:
    """Evaluate an already-configured schedule without rerunning selection."""
    from neusim.npusim.frontend import power_analysis_lib as power_lib
    from neusim.npusim.frontend.Operator import DVFSConfig, DVFSPolicy

    dvfs = DVFSConfig(
        policy=DVFSPolicy.from_str(POLICY_VALUES[policy]),
        performance_degradation_percentage=float(threshold_pct) / 100.0,
        frequency_adjustment_interval_ns=float(epoch_ns),
    )

    def identity(op: Any) -> tuple[str, str, str]:
        return str(op.name), str(op.description), str(op.opcode)

    raw_groups: dict[tuple[str, str, str], list[Any]] = defaultdict(list)
    for raw in trace.ops:
        raw_groups[identity(raw)].append(raw)
    progress: dict[tuple[str, str, str], tuple[int, int]] = {
        key: (0, int(group[0].stats.count)) for key, group in raw_groups.items()
    }
    evaluated = deepcopy(list(ops))
    for op in evaluated:
        key = identity(op)
        group = raw_groups.get(key)
        if not group:
            raise NativeExperimentError(
                f"configured schedule contains unknown operator identity {key!r}"
            )
        index, remaining = progress[key]
        selected_count = int(op.stats.count)
        if index >= len(group) or selected_count > remaining:
            raise NativeExperimentError(
                f"configured schedule count does not align with raw occurrence {key!r}"
            )
        raw = group[index]
        op.stats = deepcopy(raw.stats)
        op.stats.count = selected_count
        remaining -= selected_count
        if remaining == 0:
            index += 1
            if index < len(group):
                remaining = int(group[index].stats.count)
        progress[key] = index, remaining
    incomplete = [
        key for key, group in raw_groups.items() if progress[key][0] != len(group)
    ]
    if incomplete:
        raise NativeExperimentError(
            f"configured schedule omitted raw operator occurrence(s): {incomplete[:4]}"
        )
    analysis_config = deepcopy(trace.config)
    analysis_config.enable_dvfs = True
    for op in evaluated:
        power_lib.analyze_operator_energy(
            op,
            analysis_config,
            pg_config=pg_strategy,
            dvfs_config=dvfs,
            set_dvfs_config_for_op=False,
        )
    record = summarize(
        evaluated,
        model=trace.model,
        model_id=trace.model_id,
        phase=trace.phase,
        config=trace.config_label,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        policy=policy,
        threshold_pct=threshold_pct,
        pg_strategy=pg_strategy,
    )
    record["dvfs_execution_mode"] = "transplanted_measured_schedule"
    return Analysis(ops=evaluated, record=record, timing={})


def analyze_policy_all_budgets(
    trace: PhaseTrace,
    policy: str,
    thresholds_pct: Sequence[float],
    *,
    allow_current_ideal: bool = False,
    domain_mode: str = "dom5",
    pg_strategy: str = "NoPG",
) -> dict[float, Analysis]:
    """Run one generic request policy across budgets with shared Pareto points.

    Returned schedules are remeasured with the normal NeuSim energy path. If a
    larger budget evaluates to more energy than the best smaller-budget
    schedule, that already-feasible schedule is carried forward and remeasured.
    """
    from neusim.npusim.frontend.dvfs_optimizer import (
        configure_dvfs_for_ops_all_budgets,
    )
    from neusim.npusim.frontend.Operator import DVFSConfig, DVFSPolicy

    if policy not in {"eNPU-C", "eNPU-All", "Ideal"}:
        raise NativeExperimentError(
            "generic all-budget analysis supports eNPU-C, eNPU-All, and Ideal; "
            f"got {policy!r}"
        )
    if not thresholds_pct:
        raise NativeExperimentError(
            f"all-budget {policy} analysis needs at least one budget"
        )
    normalized = sorted({float(value) for value in thresholds_pct})
    if any(not math.isfinite(value) or value < 0 for value in normalized):
        raise NativeExperimentError(
            "thresholds must be finite, non-negative percentages"
        )
    if policy == "Ideal" and not allow_current_ideal:
        theoretical_product = int(current_state_space()["exact_product"])
        raise NativeExperimentError(
            "the current Ideal request search is intentionally opt-in. The "
            f"unreduced NeuSim frequency-table lattice has {theoretical_product:,} "
            "theoretical combinations, while the backend applies its documented "
            f"{CURRENT_IDEAL_RAW_CONFIG_LIMIT:,}-candidate bounded reduction. "
            "Rerun with --allow-current-ideal or use --quick for a labeled smoke test."
        )

    dvfs = DVFSConfig(
        policy=DVFSPolicy.from_str(POLICY_VALUES[policy]),
        performance_degradation_percentage=max(normalized) / 100.0,
        custom_compute_domain_mode=domain_mode,
    )
    ops = deepcopy(trace.ops)
    analysis_config = deepcopy(trace.config)
    analysis_config.enable_dvfs = True
    budgets = [value / 100.0 for value in normalized]
    start = time.perf_counter()
    configured = configure_dvfs_for_ops_all_budgets(
        ops,
        analysis_config,
        dvfs,
        budgets,
        pg_config=pg_strategy,
    )
    batch_wall = time.perf_counter() - start
    helper_timings: Mapping[str, Any] = getattr(
        configure_dvfs_for_ops_all_budgets, "last_timings", {}
    )
    point_generation = float(helper_timings.get("point_gen_s", 0.0))
    search_timings = helper_timings.get("search_s", {})
    algorithm = str(
        helper_timings.get("algorithm", "shared_100pct_candidate_envelope_heuristic")
    )

    output: dict[float, Analysis] = {}
    best: Analysis | None = None
    best_energy = math.inf
    for threshold in normalized:
        budget = threshold / 100.0
        configured_ops = configured.get(budget)
        if configured_ops is None:
            raise NativeExperimentError(
                f"all-budget helper omitted the {threshold:g}% {policy} cell"
            )
        result = evaluate_configured_trace(
            trace,
            configured_ops,
            policy=policy,
            threshold_pct=threshold,
            pg_strategy=pg_strategy,
        )
        carried_forward = False
        energy = float(result.record["total_energy_J"])
        if best is not None and energy > best_energy * (1.0 + 1e-12):
            result = evaluate_configured_trace(
                trace,
                best.ops,
                policy=policy,
                threshold_pct=threshold,
                pg_strategy=pg_strategy,
            )
            carried_forward = True
        else:
            best = result
            best_energy = energy

        search_seconds = (
            float(search_timings.get(budget, 0.0))
            if isinstance(search_timings, Mapping)
            else 0.0
        )
        timing = {
            "pareto_generation_seconds": point_generation,
            "inter_op_search_seconds": search_seconds,
            "configuration_wall_seconds": point_generation + search_seconds,
            "batch_configuration_wall_seconds": batch_wall,
            "execution_mode": "request_optimizer_all_budgets",
            "intra_op_algorithm": algorithm,
            "batch_budgets_pct": normalized,
            "candidate_envelope_fraction": float(
                helper_timings.get("candidate_envelope_fraction", 1.0)
            ),
            "candidate_envelope_semantics": helper_timings.get(
                "candidate_envelope_semantics"
            ),
            "budget_dependent_candidate_reduction_preserved": helper_timings.get(
                "budget_dependent_candidate_reduction_preserved"
            ),
            "candidate_set_shared_across_budgets": helper_timings.get(
                "candidate_set_shared_across_budgets"
            ),
            "independent_per_budget_candidate_semantics_preserved": helper_timings.get(
                "independent_per_budget_candidate_semantics_preserved"
            ),
            "shared_envelope_caveat": helper_timings.get("shared_envelope_caveat"),
            "pareto_generation": helper_timings.get("pareto_generation"),
            "monotonic_schedule_carried_forward": carried_forward,
        }
        if policy == "Ideal":
            timing.update(
                {
                    "ideal_raw_candidate_reduction_threshold": CURRENT_IDEAL_RAW_CONFIG_LIMIT,
                    "ideal_theoretical_lattice_states": int(
                        current_state_space()["exact_product"]
                    ),
                    "ideal_full_theoretical_lattice_enumerated": False,
                    "ideal_reduction_semantics": (
                        "per-voltage extrema followed when necessary by a deterministic "
                        "balanced endpoint-preserving cap"
                    ),
                    "ideal_search_execution": dict(CURRENT_IDEAL_SEARCH_EXECUTION),
                    "ideal_pareto_generation_summary": (
                        _compact_ideal_pareto_generation(
                            helper_timings.get("pareto_generation")
                        )
                    ),
                    "ideal_shared_envelope_caveat": helper_timings.get(
                        "ideal_shared_envelope_caveat"
                    ),
                }
            )
        result.record.update(
            {
                "dvfs_execution_mode": timing["execution_mode"],
                "intra_op_algorithm": algorithm,
                "batch_budgets_pct": normalized,
                "candidate_envelope_fraction": timing["candidate_envelope_fraction"],
                "candidate_set_shared_across_budgets": timing[
                    "candidate_set_shared_across_budgets"
                ],
                "independent_per_budget_candidate_semantics_preserved": timing[
                    "independent_per_budget_candidate_semantics_preserved"
                ],
                "budget_dependent_candidate_reduction_preserved": timing[
                    "budget_dependent_candidate_reduction_preserved"
                ],
                "shared_envelope_caveat": timing["shared_envelope_caveat"],
                "monotonic_schedule_carried_forward": carried_forward,
                "domain_mode": domain_mode,
            }
        )
        if policy == "Ideal":
            result.record.update(
                {
                    "ideal_raw_candidate_reduction_threshold": CURRENT_IDEAL_RAW_CONFIG_LIMIT,
                    "ideal_theoretical_lattice_states": int(
                        current_state_space()["exact_product"]
                    ),
                    "ideal_full_theoretical_lattice_enumerated": False,
                    "ideal_reduction_semantics": timing["ideal_reduction_semantics"],
                    "ideal_search_execution": dict(CURRENT_IDEAL_SEARCH_EXECUTION),
                    "ideal_pareto_generation_summary": timing[
                        "ideal_pareto_generation_summary"
                    ],
                    "ideal_shared_envelope_caveat": timing[
                        "ideal_shared_envelope_caveat"
                    ],
                }
            )
        output[threshold] = Analysis(
            ops=result.ops, record=result.record, timing=timing
        )
    return output


def analyze_dvfsc_all_budgets(
    trace: PhaseTrace,
    thresholds_pct: Sequence[float],
    *,
    millisecond_regions: bool = False,
    epoch_ns: float = 5_000_000.0,
    pg_strategy: str = "NoPG",
    population_size: int = 1000,
    max_generations: int = 500,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.15,
    elitism_count: int = 50,
    seed: int = 42,
    use_pareto_policy: bool = False,
) -> dict[float, Analysis]:
    """Run the authorized shared-precompute DVFS-C threshold sweep."""
    if not thresholds_pct:
        raise NativeExperimentError(
            "all-budget DVFS-C analysis needs at least one budget"
        )
    if millisecond_regions and not has_ms_region_support():
        raise NativeExperimentError(
            "DVFS-C-ms all-budget analysis requires the authoritative region modules"
        )
    from neusim.npusim.frontend.dvfs_optimizer import (
        configure_dvfs_c_ms_all_budgets,
        configure_dvfs_c_no_pareto_all_budgets,
    )
    from neusim.npusim.frontend.Operator import DVFSPolicy

    normalized = [float(value) for value in thresholds_pct]
    budgets = sorted({value / 100.0 for value in normalized})
    ops = deepcopy(trace.ops)
    analysis_config = deepcopy(trace.config)
    analysis_config.enable_dvfs = True
    start = time.perf_counter()
    if millisecond_regions:
        helper = configure_dvfs_c_ms_all_budgets
        configured = helper(
            ops, analysis_config, budgets, float(epoch_ns), pg_config=pg_strategy
        )
        helper_timings: Mapping[str, Any] = getattr(helper, "last_timings", {})
        algorithm = "checkpointed_region_GA_all_budgets"
    else:
        helper = configure_dvfs_c_no_pareto_all_budgets
        configured = helper(
            ops,
            analysis_config,
            budgets,
            pg_config=pg_strategy,
            population_size=population_size,
            max_generations=max_generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            elitism_count=elitism_count,
            seed=seed,
            ga_policy=(
                DVFSPolicy.DVFS_C if use_pareto_policy else DVFSPolicy.DVFS_C_NO_PARETO
            ),
        )
        helper_timings = getattr(helper, "last_timings", {})
        family = "DVFSC" if use_pareto_policy else "DVFSCNoPareto"
        algorithm = (
            f"{family}_shared_points_GA_all_budgets_"
            f"{population_size}x{max_generations}"
        )
    batch_wall = time.perf_counter() - start

    point_generation = float(helper_timings.get("point_gen_s", batch_wall))
    ga_timings = helper_timings.get("ga_s", {})
    ga_details_by_budget = helper_timings.get("ga_details", {})
    ms_candidate_batching = (
        _ms_candidate_batch_provenance(helper_timings) if millisecond_regions else {}
    )
    output: dict[float, Analysis] = {}
    for threshold in normalized:
        budget = threshold / 100.0
        budget_ga_details: dict[str, Any] = {}
        if isinstance(ga_details_by_budget, Mapping):
            candidate_details = ga_details_by_budget.get(budget)
            if isinstance(candidate_details, Mapping):
                budget_ga_details = deepcopy(dict(candidate_details))
        configured_ops = configured.get(budget)
        if configured_ops is None:
            raise NativeExperimentError(
                f"all-budget helper omitted the {threshold:g}% DVFS-C cell"
            )
        result = evaluate_configured_trace(
            trace,
            configured_ops,
            policy="DVFS-C-ms" if millisecond_regions else "DVFS-C",
            threshold_pct=threshold,
            pg_strategy=pg_strategy,
            epoch_ns=epoch_ns,
        )
        ga_seconds = (
            float(ga_timings.get(budget, 0.0))
            if isinstance(ga_timings, Mapping)
            else 0.0
        )
        timing = {
            "pareto_generation_seconds": point_generation,
            "inter_op_search_seconds": ga_seconds,
            "configuration_wall_seconds": point_generation + ga_seconds,
            "batch_configuration_wall_seconds": batch_wall,
            "execution_mode": "request_optimizer_all_budgets",
            "intra_op_algorithm": algorithm,
            "batch_budgets_pct": normalized,
            **ms_candidate_batching,
            "ga_policy": helper_timings.get("ga_policy"),
            "candidate_generation": helper_timings.get("candidate_generation"),
            "baseline_semantics": helper_timings.get("baseline_semantics"),
            "raw_baseline_time_ns": helper_timings.get("raw_baseline_time_ns"),
            "zero_degradation_baseline_injection": helper_timings.get(
                "zero_degradation_baseline_injection"
            ),
        }
        if budget_ga_details:
            timing["ga_details"] = budget_ga_details
            for field in GA_PROVENANCE_FIELDS:
                if field in budget_ga_details:
                    timing[field] = budget_ga_details[field]
        result.record.update(
            {
                "dvfs_execution_mode": timing["execution_mode"],
                "intra_op_algorithm": algorithm,
                "batch_budgets_pct": normalized,
                **ms_candidate_batching,
            }
        )
        result.record.update(
            {
                "ga_policy": timing["ga_policy"],
                "candidate_generation": timing["candidate_generation"],
                "baseline_semantics": timing["baseline_semantics"],
                "raw_baseline_time_ns": timing["raw_baseline_time_ns"],
                "zero_degradation_baseline_injection": timing[
                    "zero_degradation_baseline_injection"
                ],
            }
        )
        if budget_ga_details:
            result.record["ga_details"] = deepcopy(budget_ga_details)
            for field in GA_PROVENANCE_FIELDS:
                if field in timing:
                    result.record[field] = timing[field]
        if millisecond_regions:
            result.record["epoch_ns"] = float(epoch_ns)
        else:
            result.record["ga_gens"] = int(max_generations)
        output[threshold] = Analysis(
            ops=result.ops, record=result.record, timing=timing
        )
    return output


def analyze_enpu_ms_all_budgets(
    trace: PhaseTrace,
    thresholds_pct: Sequence[float],
    *,
    epoch_ns: float = 5_000_000.0,
    pg_strategy: str = "NoPG",
) -> dict[float, Analysis]:
    """Run the authoritative request-level eNPU-ms sweep with shared precompute."""
    if not thresholds_pct:
        raise NativeExperimentError(
            "all-budget eNPU-ms analysis needs at least one budget"
        )
    if not has_ms_region_support():
        raise NativeExperimentError(
            "eNPU-ms all-budget analysis requires the authoritative region modules"
        )
    from neusim.npusim.frontend.dvfs_enpu_ms import (
        configure_enpu_ms_all_budgets,
    )

    normalized = [float(value) for value in thresholds_pct]
    budgets = sorted({value / 100.0 for value in normalized})
    ops = deepcopy(trace.ops)
    analysis_config = deepcopy(trace.config)
    analysis_config.enable_dvfs = True
    start = time.perf_counter()
    configured = configure_enpu_ms_all_budgets(
        ops,
        analysis_config,
        budgets,
        float(epoch_ns),
        pg_config=pg_strategy,
    )
    batch_wall = time.perf_counter() - start
    helper_timings: Mapping[str, Any] = getattr(
        configure_enpu_ms_all_budgets, "last_timings", {}
    )
    ms_candidate_batching = _ms_candidate_batch_provenance(helper_timings)
    point_generation = float(helper_timings.get("point_gen_s", 0.0))
    region_evaluation = float(helper_timings.get("region_candidate_evaluation_s", 0.0))
    if "exact_measurement_s" not in helper_timings:
        raise NativeExperimentError(
            "authoritative eNPU-ms helper omitted exact_measurement_s"
        )
    exact_measurement = float(helper_timings["exact_measurement_s"])
    if "total_s" not in helper_timings:
        raise NativeExperimentError("authoritative eNPU-ms helper omitted total_s")
    helper_total = float(helper_timings["total_s"])
    if not math.isfinite(exact_measurement) or exact_measurement < 0.0:
        raise NativeExperimentError(
            "authoritative eNPU-ms helper returned invalid exact_measurement_s"
        )
    if not math.isfinite(helper_total) or helper_total < 0.0:
        raise NativeExperimentError(
            "authoritative eNPU-ms helper returned invalid total_s"
        )
    search_timings = helper_timings.get("ga_s", helper_timings.get("search_s", {}))
    selected_times = helper_timings.get("selected_time_ns", {})
    selected_energies = helper_timings.get("selected_energy_J", {})
    allowed_times = helper_timings.get("allowed_time_ns", {})
    movements = helper_timings.get("movement_from_fastest_pct", {})
    variable_budget_usage = helper_timings.get("variable_budget_used_pct", {})
    raw_time = float(helper_timings.get("raw_baseline_time_ns", 0.0))
    fastest_time = float(helper_timings.get("fastest_plan_time_ns", raw_time))
    if "fixed_peak_plan_overhead_ns" in helper_timings:
        fixed_allowance = float(helper_timings["fixed_peak_plan_overhead_ns"])
    elif "fixed_transition_allowance_ns" in helper_timings:
        fixed_allowance = float(helper_timings["fixed_transition_allowance_ns"])
    else:
        raise NativeExperimentError(
            "authoritative eNPU-ms helper omitted fixed peak-plan overhead"
        )
    fixed_overhead_scope = str(
        helper_timings.get(
            "fixed_overhead_scope",
            "peak CUSTOM_ALL_ms schedule versus original trace time",
        )
    )
    zero_budget_allowed_time = float(
        helper_timings.get("zero_budget_allowed_time_ns", max(raw_time, fastest_time))
    )
    natural_headroom = float(
        helper_timings.get("natural_headroom_ns", max(0.0, raw_time - fastest_time))
    )
    algorithm = str(
        helper_timings.get(
            "algorithm",
            "authoritative_5domain_regional_GA_all_budgets",
        )
    )
    implementation_provenance = str(
        helper_timings.get("implementation_provenance", "authoritative_trace_util_port")
    )

    output: dict[float, Analysis] = {}
    for threshold in normalized:
        budget = threshold / 100.0
        configured_ops = configured.get(budget)
        if configured_ops is None:
            raise NativeExperimentError(
                f"all-budget helper omitted the {threshold:g}% eNPU-ms cell"
            )
        result = evaluate_configured_trace(
            trace,
            configured_ops,
            policy="eNPU-ms",
            threshold_pct=threshold,
            pg_strategy=pg_strategy,
            epoch_ns=epoch_ns,
        )
        if not isinstance(selected_times, Mapping) or budget not in selected_times:
            raise NativeExperimentError(
                "authoritative eNPU-ms helper omitted exact selected_time_ns "
                f"for budget {budget:g}"
            )
        if (
            not isinstance(selected_energies, Mapping)
            or budget not in selected_energies
        ):
            raise NativeExperimentError(
                "authoritative eNPU-ms helper omitted exact selected_energy_J "
                f"for budget {budget:g}"
            )
        selected_time = float(selected_times[budget])
        selected_energy = float(selected_energies[budget])
        if not math.isclose(
            float(result.record["total_exe_time_ns"]),
            selected_time,
            rel_tol=1e-9,
            abs_tol=1e-5,
        ):
            raise NativeExperimentError(
                "eNPU-ms transplanted schedule time differs from candidate evaluation"
            )
        if not math.isclose(
            float(result.record["total_energy_J"]),
            selected_energy,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise NativeExperimentError(
                "eNPU-ms transplanted schedule energy differs from candidate evaluation"
            )
        search_seconds = (
            float(search_timings.get(budget, 0.0))
            if isinstance(search_timings, Mapping)
            else 0.0
        )
        allowed_time = float(allowed_times.get(budget, math.nan))
        movement_from_fastest = float(movements.get(budget, math.nan))
        variable_budget_used = float(variable_budget_usage.get(budget, math.nan))
        timing = {
            "pareto_generation_seconds": point_generation,
            "regional_candidate_evaluation_seconds": region_evaluation,
            "inter_op_search_seconds": search_seconds,
            "batch_exact_measurement_seconds": exact_measurement,
            "configuration_wall_seconds": helper_total,
            "batch_configuration_wall_seconds": batch_wall,
            "helper_total_seconds": helper_total,
            "configuration_accounting_scope": "shared_all_budget_batch",
            "execution_mode": "request_optimizer_all_budgets",
            "intra_op_algorithm": algorithm,
            "batch_budgets_pct": normalized,
            **ms_candidate_batching,
            "implementation_provenance": implementation_provenance,
            "authoritative_ms_source": dict(AUTHORITATIVE_MS_SOURCE),
            "candidate_slowdown_envelope": helper_timings.get(
                "candidate_slowdown_envelope"
            ),
            "power_gating_config": helper_timings.get("power_gating_config"),
            "raw_baseline_time_ns": raw_time,
            "fastest_plan_time_ns": fastest_time,
            "fixed_transition_allowance_ns": fixed_allowance,
            "fixed_peak_plan_overhead_ns": fixed_allowance,
            "fixed_overhead_scope": fixed_overhead_scope,
            "zero_budget_allowed_time_ns": zero_budget_allowed_time,
            "natural_headroom_ns": natural_headroom,
            "allowed_time_ns": allowed_time,
            "selected_time_ns": selected_time,
            "selected_energy_J": selected_energy,
            "movement_from_fastest_pct": movement_from_fastest,
            "variable_budget_used_pct": variable_budget_used,
            "nominal_slowdown_budget_excludes_fixed_overhead": True,
        }
        actual_overhead_pct = (
            100.0 * (selected_time / raw_time - 1.0) if raw_time else math.nan
        )
        result.record.update(
            {
                "dvfs_execution_mode": timing["execution_mode"],
                "intra_op_algorithm": algorithm,
                "batch_budgets_pct": normalized,
                **ms_candidate_batching,
                "epoch_ns": float(epoch_ns),
                "implementation_provenance": implementation_provenance,
                "authoritative_ms_source": dict(AUTHORITATIVE_MS_SOURCE),
                "candidate_slowdown_envelope": timing["candidate_slowdown_envelope"],
                "raw_baseline_time_ns": raw_time,
                "fastest_plan_time_ns": fastest_time,
                "fixed_transition_allowance_ns": fixed_allowance,
                "fixed_peak_plan_overhead_ns": fixed_allowance,
                "fixed_overhead_scope": fixed_overhead_scope,
                "zero_budget_allowed_time_ns": zero_budget_allowed_time,
                "natural_headroom_ns": natural_headroom,
                "allowed_time_ns": allowed_time,
                "nominal_slowdown_budget_excludes_fixed_overhead": True,
                "actual_total_overhead_pct": actual_overhead_pct,
                "movement_from_fastest_pct": movement_from_fastest,
                "variable_budget_used_pct": variable_budget_used,
            }
        )
        output[threshold] = Analysis(
            ops=result.ops, record=result.record, timing=timing
        )
    return output


def _sum(ops: Iterable[Any], expression: Any) -> float:
    return float(sum(expression(op) * op.stats.count for op in ops))


def summarize(
    ops: Sequence[Any],
    *,
    model: str,
    model_id: str,
    phase: str,
    config: str,
    input_tokens: int,
    output_tokens: int,
    policy: str,
    threshold_pct: float,
    pg_strategy: str,
) -> dict[str, Any]:
    if not ops:
        raise NativeExperimentError("cannot summarize an empty operator list")
    total_time = _sum(ops, lambda op: op.stats.execution_time_ns)
    total_energy = _sum(ops, lambda op: op.stats.total_energy_J)
    if total_time <= 0:
        raise NativeExperimentError(f"{model} {phase} produced zero execution time")

    record: dict[str, Any] = {
        "model": model,
        "model_id": model_id,
        "phase": phase,
        "config": config,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "policy": policy,
        "threshold_pct": float(threshold_pct),
        "pg_strategy": pg_strategy,
        "operator_rows": len(ops),
        "expanded_operator_count": int(sum(op.stats.count for op in ops)),
        "total_exe_time_ns": total_time,
        "total_energy_J": total_energy,
        "avg_power_W": total_energy / (total_time / 1e9),
        "peak_power_W": max(float(op.stats.total_power_W) for op in ops),
    }
    components = {
        "sa": "sa",
        "vu": "vu",
        "sram": "sram",
        "hbm": "hbm",
        "ici": "ici",
        "other": "other",
    }
    for label, field in components.items():
        static = _sum(
            ops, lambda op, f=field: getattr(op.stats, f"static_energy_{f}_J")
        )
        dynamic = _sum(
            ops, lambda op, f=field: getattr(op.stats, f"dynamic_energy_{f}_J")
        )
        record[f"static_{label}_energy_J"] = static
        record[f"dynamic_{label}_energy_J"] = dynamic
        record[f"{label}_energy_J"] = static + dynamic
    record["total_static_energy_J"] = sum(
        record[f"static_{name}_energy_J"] for name in components
    )
    record["total_dynamic_energy_J"] = sum(
        record[f"dynamic_{name}_energy_J"] for name in components
    )

    times = {
        "sa": "sa_time_ns",
        "vu": "vu_time_ns",
        "vmem": "vmem_time_ns",
        "hbm": "memory_time_ns",
        "ici": "ici_time_ns",
    }
    for label, field in times.items():
        value = _sum(ops, lambda op, f=field: getattr(op.stats, f))
        record[f"{label}_time_ns"] = value
        record[f"{label}_temp_util"] = value / total_time
    record["sram_time_ns"] = record["vmem_time_ns"]
    record["sram_temp_util"] = record["vmem_temp_util"]
    record["model_flops_util"] = (
        sum(
            float(op.stats.flops_util) * op.stats.execution_time_ns * op.stats.count
            for op in ops
        )
        / total_time
    )
    record["hbm_bw_util"] = (
        sum(
            float(op.stats.hbm_bw_util) * op.stats.execution_time_ns * op.stats.count
            for op in ops
        )
        / total_time
    )
    return record


def write_operator_csv(
    path: Path,
    ops: Sequence[Any],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    from neusim.npusim.frontend.Operator import to_csv_dict

    rows = [to_csv_dict(op) for op in ops]
    if metadata:
        for row in rows:
            row.update(metadata)
    if not rows:
        raise NativeExperimentError(f"refusing to write an empty trace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def legacy_policy_token(policy: str) -> str:
    return POLICY_VALUES[policy]


def legacy_csv_path(
    root: Path, trace: PhaseTrace, policy: str, threshold_pct: float
) -> Path:
    token = legacy_policy_token(policy)
    if policy != "NoDVFS":
        token += f"_{threshold_pct / 100:g}"
    return (
        root
        / f"raw_energy_{token}"
        / f"{trace.model}_{trace.input_tokens}_{trace.output_tokens}"
        / trace.config_label
        / f"inference-v5p_{trace.phase}.csv"
    )


def transplant_modal_schedule(
    source_ops: Sequence[Any], destination_ops: Sequence[Any]
) -> list[Any]:
    """Transplant a schedule by semantic node key, independent of fusion indices."""
    attrs = (
        "dvfs_sa",
        "dvfs_vu",
        "dvfs_sram",
        "dvfs_hbm_mc",
        "dvfs_hbm_die",
        "dvfs_hbm_io",
        "dvfs_ici_mc",
        "dvfs_ici_phy",
    )

    def keys(ops: Sequence[Any]) -> dict[str, tuple[str, int]]:
        counts: dict[str, int] = {}
        result: dict[str, tuple[str, int]] = {}
        for op in ops:
            if op.name in result:
                continue
            stable = re.sub(r"\d+", "", op.name or "")
            ordinal = counts.get(stable, 0)
            counts[stable] = ordinal + 1
            result[op.name] = stable, ordinal
        return result

    source_keys = keys(source_ops)
    candidates: dict[tuple[str, int], Counter[tuple[Any, ...]]] = defaultdict(Counter)
    configurations: dict[tuple[tuple[str, int], tuple[Any, ...]], dict[str, Any]] = {}
    for op in source_ops:
        key = source_keys[op.name]
        signature = tuple(
            (
                getattr(getattr(op, name), "frequency_GHz", None),
                getattr(getattr(op, name), "voltage_V", None),
            )
            for name in attrs
        )
        candidates[key][signature] += max(1, int(getattr(op.stats, "count", 1)))
        configurations[(key, signature)] = {
            name: deepcopy(getattr(op, name)) for name in attrs
        }

    def conservative_mode(counts: Counter[tuple[Any, ...]]) -> tuple[Any, ...]:
        def tie_key(item: tuple[tuple[Any, ...], int]) -> tuple[Any, ...]:
            signature, count = item
            numeric = tuple(
                -1.0 if value is None else float(value)
                for domain in signature
                for value in domain
            )
            return count, numeric

        return max(counts.items(), key=tie_key)[0]

    schedule = {
        key: configurations[(key, conservative_mode(counts))]
        for key, counts in candidates.items()
    }
    output = deepcopy(destination_ops)
    destination_keys = keys(output)
    for op in output:
        values = schedule.get(destination_keys[op.name])
        if values:
            for name, value in values.items():
                setattr(op, name, deepcopy(value))
    return output


def current_state_space() -> dict[str, Any]:
    from neusim.npusim.backend.dvfs_power_getter import (
        HBM_MC_POINTS,
        ICI_MC_POINTS,
        SA_POINTS,
        SRAM_POINTS,
        VU_POINTS,
    )

    domains = [
        len(SA_POINTS),
        len(VU_POINTS),
        len(SRAM_POINTS),
        len(HBM_MC_POINTS),
        len(ICI_MC_POINTS),
    ]
    return {
        "domain_order": ["SA", "VU", "SRAM", "HBM_MC", "ICI_MC"],
        "domain_state_counts": domains,
        "exact_product": math.prod(domains),
    }
