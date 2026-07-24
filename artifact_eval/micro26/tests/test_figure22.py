from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from artifact_eval.micro26 import build_figure_22
from artifact_eval.micro26.plots import figure_22
from artifact_eval.micro26.plots.figure_05 import SLOTarget


def write_slo_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "percentile": 33,
                        "input_seqlen": 100,
                        "prefill": {"slo_TTFT_sec": {"5x": 0.724832}},
                        "decode": {
                            "representative_seqlen": 120,
                            "slo_TPOT_ms": {"5x": 91.1996},
                        },
                    },
                    {
                        "percentile": 100,
                        "input_seqlen": 200,
                        "prefill": {"slo_TTFT_sec": {"5x": 4.405492}},
                        "decode": {
                            "representative_seqlen": 240,
                            "slo_TPOT_ms": {"5x": 99.6988},
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def write_trace(path: Path, *, energy_scale: float = 1.0) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(figure_22.REQUIRED_TRACE_COLUMNS)
        )
        writer.writeheader()
        for index, minute in enumerate((0, 2, 6, 12 * 60, 12 * 60 + 2)):
            writer.writerow(
                {
                    "enqueue_timestamp": minute * 60 * 1e9,
                    "input_seqlen": 100 if index % 2 == 0 else 150,
                    "output_seqlen": 20,
                    "TTFT_ns": (0.5 if index % 2 == 0 else 2.5) * 1e9,
                    "TPOT_ns": (5 if index % 2 == 0 else 25) * 1e6,
                    "prefill_energy_J": energy_scale * (10 + index),
                    "decode_energy_J": energy_scale * (20 + index),
                }
            )


def test_load_run_uses_slo_buckets_and_aggregate_decode_energy(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "request_trace.csv"
    write_trace(trace)
    targets = [
        SLOTarget(100, 1.0, 120, 10.0),
        SLOTarget(200, 2.0, 240, 20.0),
    ]

    frame = figure_22.load_run(trace, targets)

    assert frame["prefill_pass"].tolist()[:2] == [True, False]
    assert frame["decode_pass"].tolist()[:2] == [True, False]
    assert frame["decode_energy"].tolist()[:2] == [20.0, 21.0]
    # The aggregate energy is not multiplied by output length.
    assert frame["decode_energy"].iloc[0] != 20.0 * 20


def test_paper_plot_rounds_slo_targets_in_seconds_only(
    tmp_path: Path,
) -> None:
    exact_targets = [
        SLOTarget(1291, 0.724832, 1313, 91.1996),
        SLOTarget(2777, 1.488405, 2802, 92.9012),
        SLOTarget(7691, 4.405492, 9071, 99.6988),
    ]
    rounded = figure_22.round_slo_targets_for_paper_plot(exact_targets)

    assert [target.ttft_seconds for target in rounded] == [
        0.725,
        1.488,
        4.405,
    ]
    assert [target.tpot_milliseconds / 1000.0 for target in rounded] == [
        0.091,
        0.093,
        0.1,
    ]

    trace = tmp_path / "rounding_boundary.csv"
    with trace.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(figure_22.REQUIRED_TRACE_COLUMNS)
        )
        writer.writeheader()
        writer.writerow(
            {
                "enqueue_timestamp": 0,
                "input_seqlen": 1291,
                "output_seqlen": 22,
                # Passes rounded .725 s, but would fail exact .724832 s.
                "TTFT_ns": 0.7249 * 1e9,
                # Fails rounded .091 s, but would pass exact .0911996 s.
                "TPOT_ns": 0.0911 * 1e9,
                "prefill_energy_J": 1.0,
                "decode_energy_J": 1.0,
            }
        )

    frame = figure_22.load_run(trace, exact_targets)
    assert frame["prefill_pass"].tolist() == [True]
    assert frame["decode_pass"].tolist() == [False]


def test_original_rolling_and_normalization_semantics(
    tmp_path: Path,
) -> None:
    traces: dict[str, Path] = {}
    for name, scale in (("baseline", 1.0), ("DVFSC", 0.8), ("CustomAll", 0.6)):
        traces[name] = tmp_path / f"{name}.csv"
        write_trace(traces[name], energy_scale=scale)
    targets = [
        SLOTarget(100, 1.0, 120, 10.0),
        SLOTarget(200, 2.0, 240, 20.0),
    ]

    grid = figure_22.compute_grids(traces, targets)

    assert tuple(grid) == figure_22.RUN_ORDER
    np.testing.assert_allclose(grid["baseline"]["prefill_energy_norm"].dropna(), 1.0)
    np.testing.assert_allclose(grid["DVFSC"]["prefill_energy_norm"].dropna(), 0.8)
    np.testing.assert_allclose(grid["CustomAll"]["decode_energy_norm"].dropna(), 0.6)
    # Requests at t=0 and t=2 minutes share the five-minute trailing window.
    assert grid["baseline"].loc[pd.Timedelta(minutes=2), "prefill_slo"] == 0.5

    figure, axes = figure_22.create_figure(grid)
    try:
        assert figure.get_size_inches().tolist() == [6.8, 2.2]
        assert axes[0, 0].get_xlim() == (12.0, 24.0)
        assert axes[0, 0].get_ylim() == (95.0, 100.3)
        assert axes[0, 1].get_ylim() == (99.88, 100.01)
        assert axes[1, 0].get_ylim() == (0.7, 1.05)
        np.testing.assert_allclose(axes[0, 0].get_yticks(), [95, 96, 97, 98, 99, 100])
        np.testing.assert_allclose(axes[0, 1].get_yticks(), [99.90, 100.00])
        np.testing.assert_allclose(axes[0, 1].get_yticks(minor=True), [99.95])
        assert [label.get_text() for label in axes[0, 1].get_yticklabels()] == [
            "99.9%",
            "100%",
        ]
        assert axes[0, 1].yaxis.get_ticks_position() == "left"
        assert axes[1, 1].yaxis.get_ticks_position() == "left"
        assert all(label.get_visible() for label in axes[1, 1].get_yticklabels())
        assert axes[0, 1].get_legend() is None
        assert axes[1, 1].get_legend() is None
        assert len(figure.legends) == 1
        legend = figure.legends[0]
        assert legend.get_frame_on()
        np.testing.assert_allclose(
            legend.get_frame().get_edgecolor(), (0.0, 0.0, 0.0, 1.0)
        )
        assert [text.get_text() for text in legend.get_texts()] == [
            figure_22.LABELS[name] for name in figure_22.RUN_ORDER
        ]
        assert len(axes[0, 0].lines) == 3
        assert len(axes[1, 0].lines) == 4
        assert [line.get_color() for line in axes[0, 0].lines] == [
            figure_22.COLORS[name] for name in figure_22.RUN_ORDER
        ]
    finally:
        plt.close(figure)


def test_review_builder_writes_all_outputs_and_documents_energy_quirk(
    tmp_path: Path,
) -> None:
    slo = tmp_path / "slo.json"
    write_slo_config(slo)
    traces = {}
    for name, scale in (("baseline", 1.0), ("dvfsc", 0.8), ("enpu", 0.6)):
        traces[name] = tmp_path / f"{name}.csv"
        write_trace(traces[name], energy_scale=scale)
    output = tmp_path / "review"

    outputs, report, provenance = build_figure_22.build(
        traces["baseline"],
        traces["dvfsc"],
        traces["enpu"],
        slo,
        output,
    )

    for path in (outputs.pdf, outputs.png, outputs.csv, report, provenance):
        assert path.is_file()
        assert path.stat().st_size > 0
    text = report.read_text(encoding="utf-8")
    assert "Static allocation: 20 prefill vPods and 8 decode vPods" in text
    assert "aggregate energy per completed request" in text
    assert "does not divide by input or output token count" in text
    assert "service-level DVFS scheduler use the exact `5x` targets" in text
    assert "rounded both TTFT and TPOT targets in seconds" in text
    document = json.loads(provenance.read_text(encoding="utf-8"))
    assert document["allocation"]["prefill_vpods"] == 20
    assert document["allocation"]["decode_vpods"] == 8
    slo = document["aggregation"]["slo_satisfaction"]
    assert slo["scheduler"]["thresholds"][0]["ttft_seconds"] == 0.724832
    assert slo["scheduler"]["thresholds"][0]["tpot_seconds"] == 0.0911996
    assert slo["paper_plot"]["decimal_places_seconds"] == 3
    assert slo["paper_plot"]["thresholds"][0]["ttft_seconds"] == 0.725
    assert slo["paper_plot"]["thresholds"][0]["tpot_seconds"] == 0.091
    assert not document["data_policy"]["original_trace_util_result_data_consumed"]
