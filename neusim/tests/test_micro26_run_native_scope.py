from __future__ import annotations

import argparse
import json
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path

from artifact_eval.micro26.experiments import run_native


class TestMicro26NativeScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        manifest = repo_root / "artifact_eval/micro26/config/paper_experiments.json"
        cls.paper = json.loads(manifest.read_text(encoding="utf-8"))

    def test_repeatable_selectors_resolve_in_canonical_order(self) -> None:
        parser = run_native.build_parser()
        args = parser.parse_args(
            [
                "--group",
                "standard_sweep",
                "--output-dir",
                "/tmp/micro26-scope-test",
                "--model",
                "deepseekv3_671b",
                "--model",
                "llama3_1_405b",
                "--phase",
                "decode",
            ]
        )
        run_native._validate_scope_selection(args, parser)

        self.assertEqual(
            run_native._selected_models(args, self.paper),
            ["llama3_1_405b", "deepseekv3_671b"],
        )
        self.assertEqual(run_native._selected_phases(args), ("decode",))
        self.assertEqual(run_native._trace_task_count(args, self.paper), 2)

    def test_quick_explicit_model_overrides_default_first_model(self) -> None:
        args = argparse.Namespace(
            quick=True,
            models=["llama3_1_405b"],
            phases=["decode"],
            group="standard_sweep",
        )
        self.assertEqual(
            run_native._selected_models(args, self.paper), ["llama3_1_405b"]
        )
        self.assertEqual(run_native._selected_phases(args), ("decode",))

    def test_scope_is_rejected_for_unrelated_group(self) -> None:
        parser = run_native.build_parser()
        args = parser.parse_args(
            [
                "--group",
                "power_gating",
                "--output-dir",
                "/tmp/micro26-scope-test",
                "--phase",
                "decode",
            ]
        )
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            run_native._validate_scope_selection(args, parser)

    def test_duplicate_scope_is_rejected(self) -> None:
        parser = run_native.build_parser()
        args = parser.parse_args(
            [
                "--group",
                "domain_count",
                "--output-dir",
                "/tmp/micro26-scope-test",
                "--model",
                "llama3_1_405b",
                "--model",
                "llama3_1_405b",
            ]
        )
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            run_native._validate_scope_selection(args, parser)

    def test_configuration_source_override_and_default(self) -> None:
        default_contract = run_native._table_3_configuration_contract(self.paper)
        self.assertEqual(default_contract["source"], self.paper["configuration_source"])

        without_source = deepcopy(self.paper)
        without_source.pop("configuration_source")
        fallback_contract = run_native._table_3_configuration_contract(without_source)
        self.assertEqual(
            fallback_contract["source"],
            "paper_experiments.json Table 3 transcription",
        )

        variant = deepcopy(self.paper)
        variant["configuration_source"] = "TP32 decode sensitivity"
        variant_contract = run_native._table_3_configuration_contract(variant)
        self.assertEqual(variant_contract["source"], "TP32 decode sensitivity")

    def test_full_manifest_is_validated_outside_selected_scope(self) -> None:
        invalid = deepcopy(self.paper)
        invalid["models"][-1]["phases"]["prefill"]["chips"] = 31
        with self.assertRaisesRegex(
            run_native.NativeExperimentError,
            "DP/TP/PP/EP product does not equal chips",
        ):
            run_native._table_3_configuration_contract(invalid)


if __name__ == "__main__":
    unittest.main()
