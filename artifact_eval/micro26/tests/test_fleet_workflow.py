from __future__ import annotations

import csv
import hashlib
import sys
import time
from pathlib import Path

import pytest

from artifact_eval.micro26.experiments import run_fleet as workflow


def _write_result_trace(path: Path, *, blank_column: str | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=workflow.EXPECTED_RESULT_COLUMNS)
        writer.writeheader()
        for index in range(2):
            row = {column: index + 1 for column in workflow.EXPECTED_RESULT_COLUMNS}
            row.update(
                {
                    "enqueue_timestamp": index,
                    "prefill_start_timestamp": index + 1,
                    "prefill_end_timestamp": index + 2,
                    "decode_start_timestamp": index + 3,
                    "decode_end_timestamp": index + 4,
                }
            )
            if blank_column is not None and index == 1:
                row[blank_column] = ""
            writer.writerow(row)


def test_full_result_validation_checks_every_expected_cell(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "request_trace.csv"
    _write_result_trace(trace)

    record = workflow.validate_result_trace(trace, expected_rows=2)

    assert record["rows"] == 2
    assert record["columns"] == list(workflow.EXPECTED_RESULT_COLUMNS)
    assert record["all_expected_cells_numeric_and_finite"] is True
    assert record["last_completion_hours"] == 5 / (3600 * 1e9)


def test_full_result_validation_rejects_blank_energy(tmp_path: Path) -> None:
    trace = tmp_path / "request_trace.csv"
    _write_result_trace(trace, blank_column="decode_energy_J")

    with pytest.raises(workflow.WorkflowValidationError, match="empty decode_energy_J"):
        workflow.validate_result_trace(trace, expected_rows=2)


def test_policy_command_pins_paper_safeguards_and_strict_cache(
    tmp_path: Path,
) -> None:
    command = workflow.policy_command(
        policy="DVFSC",
        trace_path=tmp_path / "azure.csv",
        run_dir=tmp_path / "run",
        backend_cache=tmp_path / "backend",
        lookup_cache=tmp_path / "lookup" / "DVFSC",
    )

    assert "--enable_dvfs=true" in command
    assert "--enable_dvfs_power_model=true" in command
    assert "--dvfs_policy=DVFSC" in command
    assert "--dvfs_max_perf_degrad=1.0" in command
    assert "--dvfs_safeguard_window_minutes=5" in command
    assert "--dvfs_safeguard_violation_threshold=0.007" in command
    assert "--dvfs_require_cache_hit=true" in command
    assert "--slo_multiplier=5x" in command
    assert f"--static_vpod_allocation={workflow.ALLOCATION_CONFIG}" in command
    assert f"--dvfs_lookup_cache_dir={tmp_path / 'lookup' / 'DVFSC'}" in command


def test_asymmetric_cache_defaults_match_48_core_reference() -> None:
    assert workflow.default_cache_worker_counts(48) == (28, 11)
    assert sum(workflow.default_cache_worker_counts(24)) < 24


def test_reviewer_defaults_use_packaged_trace_and_lookup_cache() -> None:
    args = workflow.parse_args([])

    assert args.trace_file == workflow.DEFAULT_TRACE_FILE
    assert args.lookup_cache_dir == workflow.DEFAULT_LOOKUP_CACHE_DIR
    assert args.output_dir == workflow.DEFAULT_OUTPUT_DIR
    assert args.regenerate_lookup_cache is False


def test_regeneration_is_mutually_exclusive_with_supplied_cache(
    tmp_path: Path,
) -> None:
    regenerated = workflow.parse_args(["--regenerate-lookup-cache"])
    assert regenerated.regenerate_lookup_cache is True

    with pytest.raises(SystemExit):
        workflow.parse_args(
            [
                "--regenerate-lookup-cache",
                f"--lookup-cache-dir={tmp_path}",
            ]
        )


def test_parallel_wave_fails_fast_and_stops_sibling(tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(workflow.ParallelStageError) as raised:
        workflow.launch_commands(
            [
                (
                    "fails",
                    [sys.executable, "-c", "raise SystemExit(3)"],
                    tmp_path / "fails.log",
                ),
                (
                    "sibling",
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    tmp_path / "sibling.log",
                ),
            ],
            workflow.common_environment(),
        )

    assert time.monotonic() - started < 5
    assert raised.value.returncodes["fails"] == 3
    assert raised.value.returncodes["sibling"] != 0


def test_nonempty_root_requires_explicit_resume(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    (output / "existing").write_text("evidence", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--resume"):
        workflow.prepare_output_root(output, resume=False)

    assert workflow.prepare_output_root(output, resume=True) == output.resolve()


def test_cache_manifest_requires_fresh_exact_coverage(tmp_path: Path) -> None:
    cache_root = tmp_path / "dvfs_lookup"
    policy_root = cache_root / "DVFSC"
    tree_digest = hashlib.sha256()
    tree_files: list[dict[str, object]] = []
    for index in range(1574):
        cache_file = (
            policy_root
            / "llama3-70b"
            / f"{index}_1"
            / "5p"
            / ("prefill" if index % 2 == 0 else "decode")
            / "bs1.json"
        )
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text("{}\n", encoding="utf-8")
        relative = cache_file.relative_to(policy_root).as_posix()
        content_sha256 = workflow.sha256_file(cache_file)
        tree_files.append(
            {
                "path": relative,
                "bytes": cache_file.stat().st_size,
                "sha256": content_sha256,
            }
        )
    tree_files.sort(key=lambda record: str(record["path"]))
    for record in tree_files:
        tree_digest.update(str(record["path"]).encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(str(record["sha256"]).encode("ascii"))
        tree_digest.update(b"\n")

    identity = {
        "model": "llama3-70b",
        "version": "5p",
        "policy": "DVFSC",
        "phases": ["prefill", "decode"],
        "budgets": [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "topology": {
            "prefill_vpods": 20,
            "decode_vpods": 8,
            "num_chips": 4,
            "batch_size": 1,
            "dp": 1,
            "tp": 4,
            "pp": 1,
            "ep": 1,
        },
        "trace": {
            "sha256": workflow.run_figure_05.REFERENCE_TRACE_SHA256,
            "rows": workflow.EXPECTED_REQUEST_ROWS,
        },
        "data_policy": {
            "original_trace_util_result_data_consumed": False,
            "preexisting_dvfs_lookup_cache_consumed": False,
        },
        "algorithm": {
            "detailed_dvfs_power_model": True,
        },
    }
    manifest = {
        "identity_sha256": workflow.canonical_json_sha256(identity),
        "identity": identity,
        "execution": {"state": "complete"},
        "coverage": {
            "status": "complete",
            "shape_pairs": 787,
            "grouped_files": 1574,
            "budgets_per_file": 9,
            "detailed_dvfs_power_model": True,
            "files_with_budget_variation": 1574,
            "files_with_positive_energy_saving": 1574,
            "output_tree": {
                "sha256": tree_digest.hexdigest(),
                "files": tree_files,
            },
        },
    }
    workflow.write_json(policy_root / "manifest.json", manifest)
    trace = {
        "sha256": workflow.run_figure_05.REFERENCE_TRACE_SHA256,
        "rows": workflow.EXPECTED_REQUEST_ROWS,
    }

    record = workflow.validate_cache_manifest(cache_root, "DVFSC", trace)

    assert record["coverage"]["grouped_files"] == 1574
    assert record["identity_sha256"] == workflow.canonical_json_sha256(identity)

    corrupted = policy_root / str(tree_files[0]["path"])
    corrupted.write_text('{"corrupt": true}\n', encoding="utf-8")
    with pytest.raises(
        workflow.WorkflowValidationError,
        match="differs from its cache manifest record",
    ):
        workflow.validate_cache_manifest(cache_root, "DVFSC", trace)
