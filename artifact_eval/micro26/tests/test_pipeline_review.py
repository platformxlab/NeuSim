from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from artifact_eval.micro26 import pipeline


class TestPipelineReview(unittest.TestCase):
    def _context(
        self, root: Path, *, selected: tuple[str, ...], quick: bool = False
    ) -> pipeline.Context:
        config = {
            "default_selection": "2,3",
            "figures": {
                "2": {"experiment_groups": []},
                "3": {"experiment_groups": []},
            },
        }
        return pipeline.Context(
            config=config,
            paper={},
            selected=selected,
            stages=("review",),
            results_dir=root,
            output_dir=root,
            paper_pdf=None,
            python="python",
            jobs=1,
            trace_workers=1,
            group_trace_worker_overrides=(),
            allow_current_ideal=False,
            verbose_simulator=False,
            quick=quick,
            dry_run=True,
            resume=False,
        )

    def test_review_only_invokes_figure_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contexts = (
                self._context(root, selected=("2",)),
                self._context(root, selected=("2", "3")),
                self._context(root, selected=("2", "3"), quick=True),
            )
            for context in contexts:
                with self.subTest(selected=context.selected, quick=context.quick):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        pipeline.review(context)
                    rendered = output.getvalue()
                    self.assertEqual(rendered.count("[Numerical figure review]"), 1)
                    self.assertEqual(rendered.count("generate_review_report.py"), 1)
                    self.assertIn("--results-dir", rendered)
                    self.assertIn("--output-dir", rendered)

    def test_parser_defaults_to_self_contained_bundle(self) -> None:
        config = {"default_selection": "2", "figures": {"2": {}}}
        args = pipeline.build_parser(config).parse_args([])
        self.assertEqual(args.results_dir, pipeline.DEFAULT_BUNDLE)
        self.assertIsNone(args.output_dir)
        self.assertEqual(args.output_dir or args.results_dir, pipeline.DEFAULT_BUNDLE)

        custom = pipeline.build_parser(config).parse_args(
            ["--results-dir", "/tmp/neusim-custom-results"]
        )
        self.assertIsNone(custom.output_dir)
        self.assertEqual(custom.output_dir or custom.results_dir, custom.results_dir)


if __name__ == "__main__":
    unittest.main()
