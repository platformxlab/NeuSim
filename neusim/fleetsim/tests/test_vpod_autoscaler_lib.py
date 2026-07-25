import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import neusim.fleetsim.vPodAutoScaler_lib as vPodAutoScalerLib
from neusim.configs.models.LLMConfig import LLMConfig
from neusim.configs.systems.NPUFleetConfig import NPUFleetConfig
from neusim.configs.workloads.LLMInferenceWorkloadConfig import RequestPatternType


class TestvPodAutoScalerLib(unittest.TestCase):
    def setUp(self):
        data_dir = Path(__file__).parent / "data"
        self.test_csv_path = str(
            data_dir / "traces" / "AzureLLMInferenceTrace_code_test.csv"
        )
        self.req_lookup_cache_dir_test = str(data_dir / "request_lookup_cache")
        self.config = NPUFleetConfig()
        self.config.workload_config.request_pattern = RequestPatternType.TRACE
        self.config.workload_config.trace_file_path = self.test_csv_path
        self.config.workload_config.request_results_cache_dir = (
            self.req_lookup_cache_dir_test
        )

    def test_num_chips_to_shape_2D(self):
        self.assertEqual(vPodAutoScalerLib.num_chips_to_shape_2D(1), [1, 1])
        self.assertEqual(vPodAutoScalerLib.num_chips_to_shape_2D(12), [3, 4])
        self.assertEqual(vPodAutoScalerLib.num_chips_to_shape_2D(4096), [64, 64])

    def test_num_chips_to_shape_3D(self):
        self.assertEqual(vPodAutoScalerLib.num_chips_to_shape_3D(1), [1, 1, 1])
        self.assertEqual(vPodAutoScalerLib.num_chips_to_shape_3D(12), [3, 2, 2])
        self.assertEqual(vPodAutoScalerLib.num_chips_to_shape_3D(4096), [16, 16, 16])

    def test_shape_to_num_chips(self):
        self.assertEqual(
            vPodAutoScalerLib.shape_to_num_chips((2, 4, 8)),
            64,
        )
        self.assertEqual(
            vPodAutoScalerLib.shape_to_num_chips((1, 4, 8)),
            32,
        )
        self.assertEqual(
            vPodAutoScalerLib.shape_to_num_chips([1]),
            1,
        )

    def test_get_vPod_create_delay_ns(self):
        self.assertEqual(
            vPodAutoScalerLib.get_vPod_create_delay_ns(
                self.config.workload_config.llm_config
            ),
            60_000_000_000,
        )

    def test_pad_to(self):
        self.assertEqual(
            vPodAutoScalerLib.pad_to(10, 64),
            64,
        )
        self.assertEqual(
            vPodAutoScalerLib.pad_to(64, 64),
            64,
        )
        self.assertEqual(
            vPodAutoScalerLib.pad_to(65, 64),
            128,
        )
        self.assertEqual(
            vPodAutoScalerLib.pad_to(128, 64),
            128,
        )
        self.assertEqual(
            vPodAutoScalerLib.pad_to(129, 35),
            140,
        )

    def test_pad_seqlen(self):
        self.assertEqual(
            vPodAutoScalerLib.pad_seqlen(
                10,
                [4],
                [],
            ),
            12,
        )
        self.assertEqual(
            vPodAutoScalerLib.pad_seqlen(
                10,
                [4, 8],
                [16],
            ),
            12,
        )
        self.assertEqual(
            vPodAutoScalerLib.pad_seqlen(
                33,
                [4, 8, 16],
                [32, 64],
            ),
            40,
        )
        self.assertEqual(
            vPodAutoScalerLib.pad_seqlen(
                65,
                [4, 8, 16],
                [32, 64],
            ),
            80,
        )

    def test_load_lookup_cache(self):
        # Just test if it runs without error
        prefill_cache = vPodAutoScalerLib.load_lookup_cache(
            "llama3-70b",
            1024,
            32,
            ("5p", "6e"),
            self.req_lookup_cache_dir_test,
            "energy",
            "prefill",
        )
        self.assertGreater(len(prefill_cache), 0)

        decode_cache = vPodAutoScalerLib.load_lookup_cache(
            "llama3-70b",
            1024,
            32,
            ("5p", "6e"),
            self.req_lookup_cache_dir_test,
            "energy",
            "decode",
        )
        self.assertGreater(len(decode_cache), 0)

    def test_load_lookup_cache_accepts_runtime_json_without_operator_csv(self):
        """Compact runtime caches need stats/config JSON, not unused op rows."""
        source = (
            Path(self.req_lookup_cache_dir_test)
            / "energy"
            / "llama3-70b"
            / "1024_32"
            / "5p"
            / "prefill"
            / "1.json"
        )
        with tempfile.TemporaryDirectory() as temporary:
            leaf = (
                Path(temporary) / "energy" / "llama3-70b" / "1024_32" / "5p" / "prefill"
            )
            leaf.mkdir(parents=True)
            (leaf / "1.json").write_text(
                json.dumps(json.loads(source.read_text(encoding="utf-8"))),
                encoding="utf-8",
            )
            vPodAutoScalerLib.load_lookup_cache.cache_clear()
            loaded = vPodAutoScalerLib.load_lookup_cache(
                "llama3-70b",
                1024,
                32,
                ("5p",),
                temporary,
                "energy",
                "prefill",
            )

        self.assertEqual(loaded["5p"][0][0], [])
        self.assertEqual(loaded["5p"][0][1]["sim_config"]["name"], "5p")

    def test_get_cost_metric_name(self):
        self.assertEqual(
            vPodAutoScalerLib.get_cost_metric_name("energy"),
            "avg_power_efficiency_tkn_per_joule",
        )
        self.assertEqual(
            vPodAutoScalerLib.get_cost_metric_name("monetary"),
            "monetary_cost_tkn_per_dollar",
        )
        with self.assertRaisesRegex(ValueError, "Unknown optimization goal"):
            vPodAutoScalerLib.get_cost_metric_name("latency")

    def test_get_optimal_vPod_config(self):
        self.config.workload_config.llm_config.input_seqlen = 1024
        self.config.workload_config.llm_config.output_seqlen = 32
        prefill_cfg = vPodAutoScalerLib.get_optimal_vPod_config(
            self.config,
            "prefill",
            ["5p", "6e"],
        )
        decode_cfg = vPodAutoScalerLib.get_optimal_vPod_config(
            self.config,
            "decode",
            ["5p", "6e"],
        )
        self.assertGreater(len(prefill_cfg), 0)
        self.assertGreater(len(decode_cfg), 0)

    def test_find_percentile_pair(self):
        pairs = [(10, 100), (20, 20), (30, 5), (40, 1)]
        self.assertEqual(
            vPodAutoScalerLib.find_percentile_pair(pairs, 50, "prefill"),
            (20, 20),
        )
        self.assertEqual(
            vPodAutoScalerLib.find_percentile_pair(pairs, 50, "decode"),
            (20, 20),
        )
        with self.assertRaisesRegex(ValueError, "prefill.*decode"):
            vPodAutoScalerLib.find_percentile_pair(pairs, 50, "training")

    def test_recommend_seqlen_pair_by_percentile(self):
        pairs = [(128, 8), (512, 32), (1024, 64)]
        self.assertEqual(
            vPodAutoScalerLib.recommend_seqlen_pair_by_percentile(
                pairs, 100, "prefill"
            ),
            (1024, 64),
        )

    def test_recommend_seqlen_by_regression_prediction(self):
        self.assertEqual(
            vPodAutoScalerLib.recommend_seqlen_by_regression_prediction(
                [(128, 8)], self.config, "prefill"
            ),
            (128, 8),
        )
        pairs = [(512, 8), (128, 64), (256, 32)]
        self.assertEqual(
            vPodAutoScalerLib.recommend_seqlen_by_regression_prediction(
                pairs, self.config, "prefill", num_samples=3
            ),
            (512, 8),
        )
        self.assertEqual(
            vPodAutoScalerLib.recommend_seqlen_by_regression_prediction(
                pairs, self.config, "decode", num_samples=3
            ),
            (512, 8),
        )
        with self.assertRaisesRegex(ValueError, "Unknown mode"):
            vPodAutoScalerLib.recommend_seqlen_by_regression_prediction(
                pairs, self.config, "training", num_samples=3
            )

    def test_get_seqlen_to_config_mapping(self):
        # Just test if it runs without error
        mapping = vPodAutoScalerLib.get_seqlen_to_config_mapping(
            [(1024, 32)],
            self.config,
            "prefill",
        )
        self.assertGreater(len(mapping), 0)
        mapping = vPodAutoScalerLib.get_seqlen_to_config_mapping(
            [(1024, 32)],
            self.config,
            "decode",
        )
        self.assertGreater(len(mapping), 0)

    def test_get_config_to_seqlen_set_mapping(self):
        # Just test if it runs without error
        mapping = vPodAutoScalerLib.get_config_to_seqlen_set_mapping(
            [(1024, 32)],
            self.config,
            "prefill",
        )
        self.assertGreater(len(mapping), 0)
        mapping = vPodAutoScalerLib.get_config_to_seqlen_set_mapping(
            [(1024, 32)],
            self.config,
            "decode",
        )
        self.assertGreater(len(mapping), 0)

    def test_get_seqlen_range_for_config(self):
        # Test when no config matches the seqlens
        seqlen_range = vPodAutoScalerLib.get_seqlen_range_for_config(
            [(1024, 32)],
            self.config.workload_config.llm_config,
            self.config,
            "prefill",
        )
        self.assertEqual(seqlen_range, (-1, -1))
        seqlen_range = vPodAutoScalerLib.get_seqlen_range_for_config(
            [(1024, 32)],
            self.config.workload_config.llm_config,
            self.config,
            "decode",
        )
        self.assertEqual(seqlen_range, (-1, -1))

        # Test when some config matches the seqlens
        self.config.workload_config.llm_config.input_seqlen = 1024
        self.config.workload_config.llm_config.output_seqlen = 32
        optimal_cfg = vPodAutoScalerLib.get_optimal_vPod_config(
            self.config,
            "prefill",
            ["5p", "6e"],
        )[0]
        seqlen_range = vPodAutoScalerLib.get_seqlen_range_for_config(
            [(1024, 32)],
            optimal_cfg,
            self.config,
            "prefill",
        )
        self.assertEqual(seqlen_range, (1024, 1024))
        optimal_cfg = vPodAutoScalerLib.get_optimal_vPod_config(
            self.config,
            "decode",
            ["5p", "6e"],
        )[0]
        seqlen_range = vPodAutoScalerLib.get_seqlen_range_for_config(
            [(1024, 32)],
            optimal_cfg,
            self.config,
            "decode",
        )
        self.assertEqual(seqlen_range, (1024 + 32, 1024 + 32))

    def test_get_available_cache_seqlens(self):
        # Clear lru_cache to avoid stale entries from other tests
        vPodAutoScalerLib.get_available_cache_seqlens.cache_clear()

        seqlens = vPodAutoScalerLib.get_available_cache_seqlens(
            self.req_lookup_cache_dir_test,
            "energy",
            "llama3-70b",
        )
        # Test cache has energy/llama3-70b/1024_32/
        self.assertIsInstance(seqlens, list)
        self.assertIn((1024, 32), seqlens)
        # All entries should be (int, int) tuples
        for pair in seqlens:
            self.assertIsInstance(pair, tuple)
            self.assertEqual(len(pair), 2)
            self.assertIsInstance(pair[0], int)
            self.assertIsInstance(pair[1], int)
        # List should be sorted
        self.assertEqual(seqlens, sorted(seqlens))

        # Verify caching: second call should return same object (cached)
        seqlens2 = vPodAutoScalerLib.get_available_cache_seqlens(
            self.req_lookup_cache_dir_test,
            "energy",
            "llama3-70b",
        )
        self.assertIs(seqlens, seqlens2)

        # Non-existent model should return empty list
        seqlens_empty = vPodAutoScalerLib.get_available_cache_seqlens(
            self.req_lookup_cache_dir_test,
            "energy",
            "nonexistent-model",
        )
        self.assertEqual(seqlens_empty, [])

    def test_get_optimal_vPod_config_returns_empty(self):
        """get_optimal_vPod_config returns [] when no valid config exists (not a default fallback)."""
        cfg = deepcopy(self.config)
        cfg.workload_config.request_results_cache_dir = self.req_lookup_cache_dir_test
        # Use a seqlen that does NOT exist in the test cache (only 1024_32 exists)
        cfg.workload_config.llm_config.input_seqlen = 8192
        cfg.workload_config.llm_config.output_seqlen = 64
        result = vPodAutoScalerLib.get_optimal_vPod_config(cfg, "prefill", ["5p", "6e"])
        self.assertEqual(result, [])

    def test_get_optimal_vPod_config_with_seqlen_fallback_no_fallback_needed(self):
        """When the primary seqlen has a valid config, return it directly without fallback."""
        cfg = deepcopy(self.config)
        cfg.workload_config.request_results_cache_dir = self.req_lookup_cache_dir_test
        cfg.workload_config.llm_config.input_seqlen = 1024
        cfg.workload_config.llm_config.output_seqlen = 32
        result = vPodAutoScalerLib.get_optimal_vPod_config_with_seqlen_fallback(
            cfg, "prefill", ["5p", "6e"]
        )
        self.assertGreater(len(result), 0)

    def test_get_optimal_vPod_config_with_seqlen_fallback_success(self):
        """When the primary seqlen has no config but a smaller one does, return the fallback."""
        vPodAutoScalerLib.get_available_cache_seqlens.cache_clear()

        cfg = deepcopy(self.config)
        cfg.workload_config.request_results_cache_dir = self.req_lookup_cache_dir_test
        # 2048_64 doesn't exist in test cache, but 1024_32 does
        cfg.workload_config.llm_config.input_seqlen = 2048
        cfg.workload_config.llm_config.output_seqlen = 64
        result = vPodAutoScalerLib.get_optimal_vPod_config_with_seqlen_fallback(
            cfg, "prefill", ["5p", "6e"]
        )
        self.assertGreater(len(result), 0)
        # The returned config should come from the 1024_32 cache entry
        self.assertEqual(result[0].input_seqlen, 1024)
        self.assertEqual(result[0].output_seqlen, 32)

    def test_prefill_fallback_accepts_equal_input_with_different_output(self):
        """Prefill lookup is reusable when only the cached output length differs."""
        vPodAutoScalerLib.get_available_cache_seqlens.cache_clear()

        cfg = deepcopy(self.config)
        cfg.workload_config.llm_config.input_seqlen = 1024
        cfg.workload_config.llm_config.output_seqlen = 4
        result = vPodAutoScalerLib.get_optimal_vPod_config_with_seqlen_fallback(
            cfg, "prefill", ["5p", "6e"]
        )

        self.assertGreater(len(result), 0)
        self.assertEqual(result[0].input_seqlen, 1024)
        self.assertEqual(result[0].output_seqlen, 32)

    def test_fallback_skips_candidate_without_requested_phase_or_version(self):
        """Try lower-ranked sequence pairs when the closest cache leaf is empty."""
        cfg = deepcopy(self.config)
        cfg.workload_config.llm_config.input_seqlen = 1024
        cfg.workload_config.llm_config.output_seqlen = 16
        valid_config = LLMConfig(input_seqlen=1024, output_seqlen=4)
        attempted: list[tuple[int, int]] = []

        def fake_lookup(
            candidate: NPUFleetConfig,
            phase: str,
            npu_types: list[str] | None,
        ) -> list[LLMConfig]:
            pair = (
                candidate.workload_config.llm_config.input_seqlen,
                candidate.workload_config.llm_config.output_seqlen,
            )
            attempted.append(pair)
            self.assertEqual(phase, "prefill")
            self.assertEqual(npu_types, ["5p", "6e"])
            return [valid_config] if pair == (1024, 4) else []

        with (
            patch.object(
                vPodAutoScalerLib,
                "get_available_cache_seqlens",
                return_value=[(1024, 4), (1024, 8)],
            ),
            patch.object(
                vPodAutoScalerLib,
                "get_optimal_vPod_config",
                side_effect=fake_lookup,
            ),
        ):
            result = vPodAutoScalerLib.get_optimal_vPod_config_with_seqlen_fallback(
                cfg, "prefill", ["5p", "6e"]
            )

        self.assertEqual(attempted, [(1024, 16), (1024, 8), (1024, 4)])
        self.assertEqual(result, [valid_config])


class TestFindBestFitConfig(unittest.TestCase):
    """Tests for find_best_fit_config()."""

    def _make_cfg(self, name: str, num_chips: int = 64) -> LLMConfig:
        """Create a distinct LLMConfig for testing."""
        cfg = LLMConfig()
        cfg.name = name
        cfg.num_chips = num_chips
        return cfg

    def test_single_covering_config(self):
        """Returns the covering config when eff_seqlen falls within its range."""
        cfg_a = self._make_cfg("a")
        ranges = {cfg_a: (100, 500)}
        result = vPodAutoScalerLib.find_best_fit_config(300, ranges)
        self.assertIs(result, cfg_a)

    def test_tightest_covering_config(self):
        """When multiple configs cover eff_seqlen, returns the one with the smallest span."""
        cfg_wide = self._make_cfg("wide")
        cfg_tight = self._make_cfg("tight")
        ranges = {
            cfg_wide: (100, 1000),  # span = 900
            cfg_tight: (200, 400),  # span = 200, tighter
        }
        result = vPodAutoScalerLib.find_best_fit_config(300, ranges)
        self.assertIs(result, cfg_tight)

    def test_covering_preferred_over_nearest(self):
        """A covering config is always preferred even if a non-covering config is closer."""
        cfg_covering = self._make_cfg("covering")
        cfg_close = self._make_cfg("close")
        ranges = {
            cfg_covering: (100, 1000),  # covers 300, but wide span
            cfg_close: (301, 310),  # very close, doesn't cover 300
        }
        result = vPodAutoScalerLib.find_best_fit_config(300, ranges)
        self.assertIs(result, cfg_covering)

    def test_nearest_larger_config(self):
        """When no covering config exists, returns the nearest config (larger preferred)."""
        cfg_a = self._make_cfg("a")
        ranges = {cfg_a: (500, 800)}
        result = vPodAutoScalerLib.find_best_fit_config(300, ranges)
        self.assertIs(result, cfg_a)

    def test_nearest_smaller_config(self):
        """Falls back to a smaller config when no larger config is available."""
        cfg_a = self._make_cfg("a")
        ranges = {cfg_a: (100, 200)}
        result = vPodAutoScalerLib.find_best_fit_config(300, ranges)
        self.assertIs(result, cfg_a)

    def test_prefers_larger_over_smaller_at_equal_distance(self):
        """When two non-covering configs are equidistant, prefers the larger one."""
        cfg_smaller = self._make_cfg("smaller")
        cfg_larger = self._make_cfg("larger")
        ranges = {
            cfg_smaller: (100, 200),  # distance = 100 (from max=200 to 300)
            cfg_larger: (400, 600),  # distance = 100 (from min=400 to 300)
        }
        result = vPodAutoScalerLib.find_best_fit_config(300, ranges)
        self.assertIs(result, cfg_larger)

    def test_nearer_smaller_over_farther_larger(self):
        """When a smaller config is closer, it wins over a farther larger one."""
        cfg_smaller = self._make_cfg("smaller")
        cfg_larger = self._make_cfg("larger")
        ranges = {
            cfg_smaller: (250, 290),  # distance = 10
            cfg_larger: (500, 800),  # distance = 200
        }
        result = vPodAutoScalerLib.find_best_fit_config(300, ranges)
        self.assertIs(result, cfg_smaller)

    def test_skips_invalid_configs(self):
        """Configs with range (-1, -1) are skipped entirely."""
        cfg_invalid = self._make_cfg("invalid")
        cfg_valid = self._make_cfg("valid")
        ranges = {
            cfg_invalid: (-1, -1),
            cfg_valid: (500, 800),
        }
        result = vPodAutoScalerLib.find_best_fit_config(300, ranges)
        self.assertIs(result, cfg_valid)

    def test_all_invalid_returns_none(self):
        """Returns None when all configs have range (-1, -1)."""
        cfg_a = self._make_cfg("a")
        cfg_b = self._make_cfg("b")
        ranges = {
            cfg_a: (-1, -1),
            cfg_b: (-1, -1),
        }
        result = vPodAutoScalerLib.find_best_fit_config(300, ranges)
        self.assertIsNone(result)

    def test_empty_input_returns_none(self):
        """Returns None when no configs are provided."""
        result = vPodAutoScalerLib.find_best_fit_config(300, {})
        self.assertIsNone(result)

    def test_exact_boundary_match(self):
        """eff_seqlen exactly at range_min or range_max counts as covering."""
        cfg_a = self._make_cfg("a")
        ranges = {cfg_a: (300, 500)}
        self.assertIs(vPodAutoScalerLib.find_best_fit_config(300, ranges), cfg_a)
        self.assertIs(vPodAutoScalerLib.find_best_fit_config(500, ranges), cfg_a)

    def test_multiple_non_covering_picks_closest(self):
        """Among multiple non-covering configs, picks the one with smallest distance."""
        cfg_far = self._make_cfg("far")
        cfg_near = self._make_cfg("near")
        ranges = {
            cfg_far: (600, 900),  # distance = 300
            cfg_near: (350, 400),  # distance = 50
        }
        result = vPodAutoScalerLib.find_best_fit_config(300, ranges)
        self.assertIs(result, cfg_near)


def test_monetary_regression_prediction_uses_shared_pipeline_cost(monkeypatch):
    import ray

    class _ImmediateRemote:
        def __init__(self, function):
            self.function = function

        def remote(self, *args, **kwargs):
            return self.function(*args, **kwargs)

    config = NPUFleetConfig()
    config.workload_config.optimization_goal = "monetary"
    pod_config = config.workload_config.llm_config.model_copy(
        update={
            "name": "4",
            "num_chips": 8,
            "microbatch_size_ici": 4,
            "pipeline_parallelism_degree": 4,
        }
    )
    op = SimpleNamespace(stats=SimpleNamespace(execution_time_ns=5, count=2))
    helper_calls = []
    backend_batch_sizes = []
    fit_costs = []

    def run_batch(_config, requests):
        backend_batch_sizes.append(len(requests))
        return [op], [op]

    def pipeline_batch_cost(ops, cfg):
        helper_calls.append((ops, cfg))
        return 40.0

    def capture_polyfit(_seqlens, costs, degree):
        fit_costs.append(list(costs))
        assert degree == 2
        return [0.0, 0.0, costs[0]]

    monkeypatch.setattr(
        vPodAutoScalerLib,
        "get_optimal_vPod_config_with_seqlen_fallback",
        lambda *_args, **_kwargs: [pod_config],
    )
    monkeypatch.setattr(
        vPodAutoScalerLib.sim_util,
        "sample_from_list",
        lambda values, count: list(values)[:count],
    )
    monkeypatch.setattr(
        vPodAutoScalerLib.npusim_backend,
        "run_inference_request_batch",
        run_batch,
    )
    monkeypatch.setattr(
        vPodAutoScalerLib.cost_model,
        "pipeline_batch_monetary_cost_dollars",
        pipeline_batch_cost,
    )
    monkeypatch.setattr(vPodAutoScalerLib.np, "polyfit", capture_polyfit)
    monkeypatch.setattr(ray, "remote", lambda function: _ImmediateRemote(function))
    monkeypatch.setattr(ray, "get", lambda futures: futures)

    seqlen_pairs = [(64, 4), (128, 4), (256, 4), (512, 4)]
    recommendation = vPodAutoScalerLib.recommend_seqlen_by_regression_prediction(
        seqlen_pairs,
        config,
        "prefill",
        num_samples=3,
    )

    assert recommendation in seqlen_pairs
    assert backend_batch_sizes == [pod_config.microbatch_size_ici] * 3
    assert len(helper_calls) == 3
    assert all(cfg is pod_config for _ops, cfg in helper_calls)
    assert fit_costs == [[10.0, 10.0, 10.0]] * len(seqlen_pairs)
