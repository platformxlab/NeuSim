from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from artifact_eval.micro26.plots import figure_05


class TestFigure05(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    @staticmethod
    def _targets() -> list[figure_05.SLOTarget]:
        return [
            figure_05.SLOTarget(100, 1.0, 120, 10.0),
            figure_05.SLOTarget(200, 2.0, 240, 20.0),
            figure_05.SLOTarget(300, 3.0, 360, 30.0),
        ]

    def test_sequence_length_binning_uses_ceiling_and_last_bucket(self) -> None:
        targets = self._targets()
        self.assertEqual(figure_05.assign_ttft_target(1, targets), 1.0)
        self.assertEqual(figure_05.assign_ttft_target(100, targets), 1.0)
        self.assertEqual(figure_05.assign_ttft_target(101, targets), 2.0)
        self.assertEqual(figure_05.assign_ttft_target(999, targets), 3.0)
        self.assertEqual(figure_05.assign_tpot_target(120, targets), 10.0)
        self.assertEqual(figure_05.assign_tpot_target(121, targets), 20.0)
        self.assertEqual(figure_05.assign_tpot_target(999, targets), 30.0)

    def test_rolling_statistics_use_one_minute_trailing_window(self) -> None:
        summary = figure_05.rolling_slack_summary(
            np.array([6.0 / 60.0, 0.0, 4.0 / 60.0]),
            np.array([30.0, 10.0, 20.0]),
        )

        np.testing.assert_allclose(summary.time_hours, [0.0, 4.0 / 60.0, 0.1])
        np.testing.assert_allclose(summary.mean, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(summary.p1, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(summary.p25, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(summary.p75, [10.0, 20.0, 30.0])
        np.testing.assert_allclose(summary.p99, [10.0, 20.0, 30.0])

    def test_fresh_trace_plot_smoke(self) -> None:
        slo_config = self.root / "slo.json"
        slo_config.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "percentile": 100,
                            "input_seqlen": 1000,
                            "prefill": {"slo_TTFT_sec": {"5x": 1.0}},
                            "decode": {
                                "representative_seqlen": 2000,
                                "slo_TPOT_ms": {"5x": 100.0},
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        request_trace = self.root / "request_trace.csv"
        with request_trace.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(figure_05.REQUIRED_TRACE_COLUMNS)
            )
            writer.writeheader()
            for index, minute in enumerate((0.0, 2.0, 7.0)):
                writer.writerow(
                    {
                        "input_seqlen": 100 + index,
                        "output_seqlen": 20,
                        "prefill_end_timestamp": minute * 60 * 1e9,
                        "decode_end_timestamp": (minute * 60 + 1) * 1e9,
                        "TTFT_ns": (0.2 + 0.1 * index) * 1e9,
                        "TPOT_ns": (20 + 10 * index) * 1e6,
                    }
                )

        output = self.root / "figure_05.pdf"
        figure_05.plot(request_trace, slo_config, output)

        self.assertTrue(output.is_file())
        self.assertGreater(output.stat().st_size, 1000)
        targets = figure_05.load_slo_targets(slo_config)
        ttft, tpot = figure_05.compute_slack_summaries(request_trace, targets)
        figure, axes = figure_05.create_figure(ttft, tpot)
        self.addCleanup(plt.close, figure)
        self.assertEqual(axes[0].get_xlim(), (0.0, 24.0))
        self.assertEqual(axes[2].get_xlim(), (0.0, 24.0))
        self.assertEqual(len(axes[0].lines), 5)
        self.assertEqual(len(axes[2].lines), 5)
        self.assertEqual(len(axes[3].lines), 6)
        self.assertEqual(axes[2].get_ylim(), (55.0, 100.0))
        np.testing.assert_allclose(axes[2].get_yticks(), [60.0, 80.0, 100.0])
        panel_gap = axes[1].get_position().y0 - axes[2].get_position().y1
        self.assertGreater(panel_gap, 0.04)
        self.assertEqual(axes[3].get_ylim(), (-10.0, 10.0))


if __name__ == "__main__":
    unittest.main()
