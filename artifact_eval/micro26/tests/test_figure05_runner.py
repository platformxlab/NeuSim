from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from artifact_eval.micro26.experiments import run_figure_05


def test_figure05_uses_one_minute_rolling_window() -> None:
    assert run_figure_05.ROLLING_WINDOW_MINUTES == 1.0


def test_generated_output_names_are_configuration_neutral() -> None:
    assert run_figure_05.RUN_NAME == "NoDVFS"
    assert run_figure_05.FIGURE_NAME == "figure_05_slo_slack.pdf"
    assert "p20d8" not in str(run_figure_05.REFERENCE_CHIP_OUTPUT).lower()


def test_figure05_allocation_config_is_explicit() -> None:
    allocation = json.loads(run_figure_05.ALLOCATION_CONFIG.read_text(encoding="utf-8"))
    assert allocation["prefill"] == {
        "count": 20,
        "npu_type": "5p",
        "num_chips": 4,
        "batch_size": 1,
        "dp": 1,
        "tp": 4,
        "pp": 1,
    }
    assert allocation["decode"] == {
        "count": 8,
        "npu_type": "5p",
        "num_chips": 4,
        "batch_size": 1,
        "dp": 1,
        "tp": 4,
        "pp": 1,
    }
    assert run_figure_05.ALLOCATION_COUNTS == {"prefill": 20, "decode": 8}


def test_inspect_azure_trace_records_hash_rows_and_span(tmp_path: Path) -> None:
    trace = tmp_path / "azure.csv"
    with trace.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=run_figure_05.AZURE_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "TIMESTAMP": "2024-05-10T00:00:00+00:00",
                "ContextTokens": 128,
                "GeneratedTokens": 16,
            }
        )
        writer.writerow(
            {
                "TIMESTAMP": "2024-05-11T00:00:00+00:00",
                "ContextTokens": 256,
                "GeneratedTokens": 32,
            }
        )

    record = run_figure_05.inspect_azure_trace(trace)

    assert record["rows"] == 2
    assert record["span_hours"] == 24.0
    assert len(record["sha256"]) == 64
    assert record["path"] == str(trace.resolve())


def test_reference_trace_validation_is_exact() -> None:
    reference = {
        "rows": run_figure_05.REFERENCE_TRACE_ROWS,
        "span_hours": run_figure_05.REFERENCE_TRACE_SPAN_HOURS,
        "sha256": run_figure_05.REFERENCE_TRACE_SHA256,
    }
    run_figure_05.validate_reference_trace(reference)

    for key, value in (
        ("rows", run_figure_05.REFERENCE_TRACE_ROWS - 1),
        ("span_hours", run_figure_05.REFERENCE_TRACE_SPAN_HOURS - 0.01),
        ("sha256", "0" * 64),
    ):
        candidate = dict(reference)
        candidate[key] = value
        with pytest.raises(ValueError, match="exact unsampled Azure Code"):
            run_figure_05.validate_reference_trace(candidate)


def test_result_trace_uses_minimum_enqueue_not_completion_order(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "request_trace.csv"
    with trace.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=run_figure_05.RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "enqueue_timestamp": 20,
                "prefill_end_timestamp": 30,
                "decode_end_timestamp": 40,
                "TTFT_ns": 10,
                "TPOT_ns": 1,
            }
        )
        writer.writerow(
            {
                "enqueue_timestamp": 10,
                "prefill_end_timestamp": 50,
                "decode_end_timestamp": 60,
                "TTFT_ns": 40,
                "TPOT_ns": 2,
            }
        )

    record = run_figure_05.inspect_result_trace(trace)

    assert record["first_enqueue_hours"] == 10 / (3600 * 1e9)
    assert record["last_enqueue_hours"] == 20 / (3600 * 1e9)


def test_static_vpod_validation_uses_configured_counts(tmp_path: Path) -> None:
    static_vpods = tmp_path / "static_vpods.json"
    entry = {
        "npu_type": "5p",
        "num_chips": 4,
        "pcfg": "bs1-dp1-tp4-pp1",
    }
    static_vpods.write_text(
        json.dumps(
            {
                "prefill": [dict(entry) for _ in range(20)],
                "decode": [dict(entry) for _ in range(8)],
            }
        ),
        encoding="utf-8",
    )

    validated = run_figure_05.validate_static_vpod_stats(static_vpods)

    assert validated == {
        "expected": {"prefill": 20, "decode": 8},
        "observed": {"prefill": 20, "decode": 8},
    }


def test_reference_output_is_protected() -> None:
    with pytest.raises(ValueError, match="reference output"):
        run_figure_05.prepare_output_directory(
            run_figure_05.REFERENCE_CHIP_OUTPUT / "figure05"
        )
