from __future__ import annotations

import csv
import dataclasses
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest

from artifact_eval.micro26.experiments import generate_figure_22_dvfs_cache as cache
from neusim.fleetsim import dvfs_scheduler


def write_azure_trace(path: Path, rows: list[tuple[str, int, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("TIMESTAMP", "ContextTokens", "GeneratedTokens"),
        )
        writer.writeheader()
        for timestamp, input_tokens, output_tokens in rows:
            writer.writerow(
                {
                    "TIMESTAMP": timestamp,
                    "ContextTokens": input_tokens,
                    "GeneratedTokens": output_tokens,
                }
            )


def test_extract_shapes_uses_current_fleetsim_padding_and_decode_minimum(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "azure.csv"
    write_azure_trace(
        trace,
        [
            ("2024-01-01T00:00:00Z", 1, 1),
            ("2024-01-01T00:00:01Z", 129, 33),
            ("2024-01-01T00:00:02Z", 1, 2),
        ],
    )

    shapes, record = cache.extract_padded_shapes(trace)

    # GeneratedTokens=1 is normalized to 2 by FleetSim, then padded to 4.
    assert shapes == [(32, 4), (192, 48)]
    assert record["rows"] == 3
    assert record["unique_padded_shapes"] == 2
    assert record["padded_shapes_sha256"] == cache.canonical_json_sha256(shapes)


def test_identity_binds_trace_source_policy_and_diagnostic_scope() -> None:
    padding = cache.current_padding_schedule()
    common = {
        "trace_record": {
            "sha256": "trace-a",
            "bytes": 10,
            "rows": 2,
            "padded_shapes_sha256": "shapes",
            "unique_padded_shapes": 1,
        },
        "shapes": [(32, 4)],
        "source_record": {"sha256": "source-a"},
        "runtime": {"python": "3.12"},
        "padding": padding,
        "allocation": {"sha256": "allocation-a"},
        "coverage_scope": {
            "mode": "diagnostic_sorted_prefix",
            "complete_trace_coverage": False,
            "full_trace_shape_pairs": 2,
            "selected_shape_pairs": 1,
            "max_pairs_argument": 1,
            "selection": "sorted(input_seqlen, output_seqlen) prefix",
        },
    }

    basis, digest = cache.build_identity(policy="DVFSC", **common)
    changed_basis, changed_digest = cache.build_identity(policy="CustomAll", **common)

    assert basis["algorithm"]["candidate_points_shared_across_budgets"] is False
    assert basis["algorithm"]["dvfs_ga_vectorized"] is False
    assert basis["algorithm"]["detailed_dvfs_power_model"] is True
    assert not basis["data_policy"]["preexisting_dvfs_lookup_cache_consumed"]
    assert basis["coverage_scope"]["complete_trace_coverage"] is False
    assert digest != changed_digest
    assert basis["policy"] != changed_basis["policy"]


def test_manifest_identity_comparison_accepts_json_tuple_round_trip() -> None:
    padding = cache.current_padding_schedule()
    identity = {
        "trace": {"sha256": "trace"},
        "source_tree_sha256": "source",
        "padding": dataclasses.asdict(padding),
    }
    identity_sha256 = cache.canonical_json_sha256(identity)
    manifest = {
        "identity_sha256": identity_sha256,
        "identity": json.loads(json.dumps(identity)),
        "provenance": {
            "trace": {"sha256": "trace"},
            "source_tree": {"sha256": "source"},
        },
    }

    # Padding is tuple-valued in memory and list-valued after JSON loading.
    assert manifest["identity"] != identity
    cache.assert_manifest_matches(manifest, identity, identity_sha256)


@dataclass
class FakeStats:
    count: int
    execution_time_ns: int = 100
    total_energy_J: float = 10.0


@dataclass
class FakeOp:
    stats: FakeStats
    description: str


class FakeConfig:
    def __init__(self, output_seqlen: int = 4) -> None:
        self.output_seqlen = output_seqlen
        self.enable_dvfs = True

    def model_copy(self, *, deep: bool = False) -> FakeConfig:
        return deepcopy(self) if deep else FakeConfig(self.output_seqlen)


class FakeGenerator:
    def __init__(self, config: FakeConfig) -> None:
        self.config = config

    def generate(self, **_kwargs):
        prefill = [FakeOp(FakeStats(count=1), "Prefill")]
        # Two layer-equivalent executions per each of four output tokens.
        decode = [FakeOp(FakeStats(count=8), "Decode")]
        return prefill + decode, prefill, decode


def test_grouped_document_calls_optimizer_independently_and_normalizes_decode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, int, int, int]] = []

    def fake_analyze(ops, config, *, dvfs_config, **_kwargs):
        calls.append(
            (dvfs_config, id(ops[0]), ops[0].stats.count, config.output_seqlen)
        )
        budget = 0.0 if "_" not in dvfs_config else float(dvfs_config.split("_")[1])
        ops[0].stats.execution_time_ns = round(100 * (1 + budget))
        ops[0].stats.total_energy_J = 10 * (1 - budget / 2)
        return ops

    monkeypatch.setattr(cache, "build_llama_config", lambda *_args: FakeConfig())
    monkeypatch.setattr(cache, "LLMOpsGenerator", FakeGenerator)
    monkeypatch.setattr(cache, "analyze_all_operator_energy", fake_analyze)
    task = cache.CacheTask(
        repo_root=str(tmp_path),
        policy_root=str(tmp_path / "cache" / "DVFSC"),
        policy="DVFSC",
        input_seqlen=32,
        output_seqlen=4,
        phase="decode",
        manifest_identity_sha256="identity",
    )

    document = cache.grouped_document_for_task(task)

    assert [call[0] for call in calls] == [
        "DVFSC",
        "DVFSC_0.01",
        "DVFSC_0.02",
        "DVFSC_0.05",
        "DVFSC_0.1",
        "DVFSC_0.15",
        "DVFSC_0.2",
        "DVFSC_0.25",
        "DVFSC_0.3",
    ]
    assert len({call[1] for call in calls}) == len(cache.BUDGETS)
    assert all(call[2:] == (2, 1) for call in calls)
    assert set(document["points"]) == {str(value) for value in cache.BUDGETS}
    assert document["metadata"]["independent_optimizer_invocations"] == 9
    assert document["metadata"]["candidate_points_shared_across_budgets"] is False
    assert document["metadata"]["detailed_dvfs_power_model"] is True


def make_valid_document(task: cache.CacheTask) -> dict:
    return {
        "schema_version": cache.SCHEMA_VERSION,
        "metadata": {
            "model": cache.MODEL,
            "input_seqlen": task.input_seqlen,
            "output_seqlen": task.output_seqlen,
            "version": cache.VERSION,
            "phase": task.phase,
            "batch_size": 1,
            "dvfs_policy": task.policy,
            "budgets": list(task.budgets),
            "num_chips": 4,
            "dp": 1,
            "tp": 4,
            "pp": 1,
            "detailed_dvfs_power_model": True,
            "manifest_identity_sha256": task.manifest_identity_sha256,
            "candidate_points_shared_across_budgets": False,
            "dvfs_ga_vectorized": False,
        },
        "points": {
            str(value): {
                "time_ns_per_stage": round(100 * (1 + value)),
                "energy_J_per_chip": 1.0 - value / 2,
                "requested_perf_degrad": value,
                "actual_perf_degrad_vs_peak": value,
                "energy_saving_vs_peak": value / 2,
            }
            for value in task.budgets
        },
    }


def test_resume_and_coverage_require_exact_identity_and_all_budget_points(
    tmp_path: Path,
) -> None:
    policy_root = tmp_path / "DVFSC"
    tasks = cache.build_tasks(
        repo_root=tmp_path,
        policy_root=policy_root,
        policy="DVFSC",
        shapes=[(32, 4)],
        identity_sha256="identity-a",
    )
    for task in tasks:
        cache.atomic_write_json(cache.cache_file_path(task), make_valid_document(task))

    coverage = cache.validate_coverage(policy_root, tasks, complete_trace=False)
    assert coverage["status"] == "diagnostic_incomplete"
    assert coverage["grouped_files"] == 2
    assert coverage["optimizer_invocations"] == 18

    document_path = cache.cache_file_path(tasks[0])
    document = json.loads(document_path.read_text(encoding="utf-8"))
    document["metadata"]["manifest_identity_sha256"] = "identity-b"
    cache.atomic_write_json(document_path, document)
    valid, errors = cache.validate_cache_file(document_path, tasks[0])
    assert not valid
    assert any("manifest_identity_sha256" in error for error in errors)
    with pytest.raises(cache.CacheValidationError):
        cache.validate_coverage(policy_root, tasks, complete_trace=False)


def test_grouped_files_are_consumed_by_strict_scheduler_loader(
    tmp_path: Path,
) -> None:
    policy_root = tmp_path / "CustomAll"
    tasks = cache.build_tasks(
        repo_root=tmp_path,
        policy_root=policy_root,
        policy="CustomAll",
        shapes=[(32, 4)],
        identity_sha256="identity",
    )
    for task in tasks:
        cache.atomic_write_json(
            cache.cache_file_path(task),
            make_valid_document(task),
        )

    try:
        assert (
            dvfs_scheduler.load_dvfs_lookup_cache(str(policy_root), strict=True) == 18
        )
        assert dvfs_scheduler.get_dvfs_lookup_stats() == {
            "entries": 2,
            "points": 18,
            "hits": 0,
            "misses": 0,
            "plans_applied": 0,
            "plans_rejected_nonbeneficial": 0,
        }
    finally:
        dvfs_scheduler.reset_dvfs_lookup_cache()


def test_manifest_match_rejects_trace_or_source_mixing() -> None:
    identity = {
        "trace": {"sha256": "trace"},
        "source_tree_sha256": "source",
    }
    manifest = {
        "identity_sha256": "digest",
        "identity": identity,
        "provenance": {
            "trace": {"sha256": "trace"},
            "source_tree": {"sha256": "source"},
        },
    }
    cache.assert_manifest_matches(manifest, identity, "digest")

    mixed = deepcopy(manifest)
    mixed["provenance"]["source_tree"]["sha256"] = "old-source"
    with pytest.raises(cache.CacheValidationError, match="source hash"):
        cache.assert_manifest_matches(mixed, identity, "digest")
