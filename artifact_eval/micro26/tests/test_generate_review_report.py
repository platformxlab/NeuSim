from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from artifact_eval.micro26 import generate_review_report


class TestGenerateReviewReport(unittest.TestCase):
    @staticmethod
    def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("model", "phase", "policy", "Execution time"),
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_figure_4_uses_only_six_nodvfs_panel_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            operator_records = Path(directory) / "operator_records"
            no_dvfs = operator_records / "raw_energy_None"
            self._write_rows(
                no_dvfs / "panels.csv",
                [
                    {
                        "model": model,
                        "phase": phase,
                        "policy": "NoDVFS",
                        "Execution time": "1",
                    }
                    for model, phase in generate_review_report.FIGURE_4_PANELS
                ]
                + [
                    {
                        "model": "deepseekv2-236b",
                        "phase": "decode",
                        "policy": "NoDVFS",
                        "Execution time": "1",
                    },
                    {
                        "model": "llama3-70b",
                        "phase": "prefill",
                        "policy": "DVFS-C",
                        "Execution time": "1",
                    },
                ],
            )
            self._write_rows(
                operator_records / "raw_energy_0" / "optimized.csv",
                [
                    {
                        "model": "llama3-70b",
                        "phase": "prefill",
                        "policy": "eNPU-All",
                        "Execution time": "1",
                    }
                ],
            )

            source, rows = generate_review_report.figure_4_records(operator_records)

        self.assertEqual(source, no_dvfs)
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {(str(row["model"]), str(row["phase"])) for row in rows},
            generate_review_report.FIGURE_4_PANELS,
        )


if __name__ == "__main__":
    unittest.main()
