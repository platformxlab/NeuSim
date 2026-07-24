import tempfile
import unittest
from math import ceil, inf
from pathlib import Path

from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
    RequestPatternType,
)
from neusim.fleetsim.LoadGenerator import LLMRequest, LoadGenerator

STATIC_ALLOCATION = {
    "prefill": {
        "count": 1, "npu_type": "4", "num_chips": 1, "batch_size": 1
    },
    "decode": {
        "count": 1, "npu_type": "4", "num_chips": 1, "batch_size": 1
    },
}


class TestLLMRequest(unittest.TestCase):
    def test_llm_request_initialization(self):
        request = LLMRequest(input_seqlen=512, output_seqlen=128, timestamp=1000)
        self.assertEqual(request.input_seqlen, 512)
        self.assertEqual(request.output_seqlen, 128)
        self.assertEqual(request.enqueue_timestamp, 1000)

    def test_llm_request_total_seqlen(self):
        request = LLMRequest(input_seqlen=512, output_seqlen=128, timestamp=1000)
        self.assertEqual(request.total_seqlen, 512+128)

    def test_llm_request_mark_progress_and_get_latency(self):
        request = LLMRequest(input_seqlen=512, output_seqlen=128, timestamp=1000)

        request.mark_prefill_started(2000)
        self.assertEqual(request.prefill_start_timestamp, 2000)
        self.assertEqual(request.current_decode_step, 0)
        self.assertTrue(request.is_prefill_started())
        self.assertRaises(AssertionError, request.TTFT_ns)

        request.mark_prefill_finished(3000)
        self.assertEqual(request.prefill_end_timestamp, 3000)
        self.assertEqual(request.current_decode_step, 1)
        self.assertTrue(request.is_prefill_finished())
        self.assertEqual(request.TTFT_ns(), 3000 - 1000)

        request.mark_decode_iteration_started(4000)
        self.assertEqual(request.decode_start_timestamp, 4000)
        self.assertEqual(request.current_decode_step, 1)
        self.assertTrue(request.is_decode_started())
        self.assertRaises(AssertionError, request.TPOT_ns)

        request.mark_decode_iteration_finished(4100, num_iterations=64)
        self.assertEqual(request.decode_end_timestamp, -1)
        self.assertEqual(request.current_decode_step, 65)
        self.assertFalse(request.is_decode_finished())
        self.assertRaises(AssertionError, request.TPOT_ns)

        request.mark_decode_iteration_started(4105)
        self.assertEqual(request.decode_start_timestamp, 4000)

        request.mark_decode_iteration_finished(4200, num_iterations=63)
        self.assertEqual(request.decode_end_timestamp, 4200)
        self.assertEqual(request.current_decode_step, 128)
        self.assertTrue(request.is_decode_finished())
        self.assertEqual(request.TPOT_ns(), ceil((4200 - 3000) / (128 - 1)))

        self.assertEqual(request.total_latency_ns(), 4200 - 1000)
        self.assertEqual(request.prefill_queuing_delay_ns(), 2000 - 1000)
        self.assertEqual(request.prefill_latency_ns(), 3000 - 2000)
        self.assertEqual(request.decode_latency_ns(), 4200 - 4000)

        request.ideal_TPOT_ns = ceil((4200 - 4005) / 128)
        self.assertEqual(
            request.decode_queuing_delay_per_iteration_ns(),
            ceil((4200 - 3000) / (128 - 1)) - ceil((4200 - 4005) / 128),
        )

    def test_llm_request_rejects_decode_iteration_overshoot(self):
        request = LLMRequest(input_seqlen=32, output_seqlen=4)
        request.mark_prefill_started(0)
        request.mark_prefill_finished(10)
        request.mark_decode_iteration_started(10)

        with self.assertRaisesRegex(ValueError, "must be positive"):
            request.mark_decode_iteration_finished(20, num_iterations=0)
        with self.assertRaisesRegex(ValueError, "only 3 remaining"):
            request.mark_decode_iteration_finished(20, num_iterations=4)

        request.mark_decode_iteration_finished(20, num_iterations=3)
        self.assertEqual(request.current_decode_step, request.output_seqlen)

    def test_llm_request_rejects_invalid_sequence_lengths_and_timestamp(self):
        with self.assertRaisesRegex(ValueError, "input_seqlen"):
            LLMRequest(input_seqlen=0, output_seqlen=2)
        with self.assertRaisesRegex(ValueError, "output_seqlen"):
            LLMRequest(input_seqlen=1, output_seqlen=1)
        with self.assertRaisesRegex(ValueError, "timestamp"):
            LLMRequest(input_seqlen=1, output_seqlen=2, timestamp=-1)


class TestLoadGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        data_dir = Path(self.temp_dir.name)

        self.azure_csv_path = data_dir / "Azure_test.csv"
        self.azure_csv_path.write_text(
            "TIMESTAMP,ContextTokens,GeneratedTokens\n"
            "2023-11-16 18:17:03.9799600,4808,10\n"
            "2023-11-16 18:17:04.0319600,3180,1\n"
            "2023-11-16 18:17:05.9799600,2679,222\n",
            encoding="utf-8",
        )
        self.burst_csv_path = data_dir / "BurstGPT_test.csv"
        self.burst_csv_path.write_text(
            "Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type\n"
            "5,ChatGPT,472,18,490,Conversation log\n"
            "45,ChatGPT,1087,230,1317,Conversation log\n"
            "105,GPT-4,417,276,693,Conversation log\n",
            encoding="utf-8",
        )
        self.config = LLMInferenceWorkloadConfig(
            static_vpod_allocation=STATIC_ALLOCATION,
            request_pattern=RequestPatternType.TRACE,
            trace_file_path=str(self.azure_csv_path),
        )

    def test_load_requests_from_trace_Azure(self):
        generator = LoadGenerator(self.config)
        requests = generator.load_requests_from_trace()

        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0].enqueue_timestamp, 0)
        self.assertEqual(requests[1].enqueue_timestamp, 52_000_000)
        self.assertEqual(requests[-1].enqueue_timestamp, 2_000_000_000)
        self.assertEqual(requests[0].input_seqlen, 5120)
        self.assertEqual(requests[1].output_seqlen, 4)

    def test_load_requests_from_trace_BurstGPT(self):
        self.config.trace_file_path = str(self.burst_csv_path)
        generator = LoadGenerator(self.config)
        requests = generator.load_requests_from_trace()

        self.assertEqual(len(requests), 3)
        self.assertEqual(requests[0].enqueue_timestamp, 0)
        self.assertEqual(requests[1].enqueue_timestamp, 40_000_000_000)
        self.assertEqual(requests[-1].enqueue_timestamp, 100_000_000_000)


class TestSyntheticLoadGenerator(unittest.TestCase):
    def _make_config(self, **kwargs):
        defaults = dict(
            static_vpod_allocation=STATIC_ALLOCATION,
            request_pattern=RequestPatternType.SYNTHETIC,
            trace_file_path="",
            pad_seqlen_loadgen=False,
        )
        defaults.update(kwargs)
        return LLMInferenceWorkloadConfig(**defaults)

    def test_fixed_seqlens_poisson(self):
        config = self._make_config(
            synthetic_num_requests=100,
            synthetic_request_rate=10.0,
            synthetic_input_len=512,
            synthetic_output_len=128,
            synthetic_seed=42,
        )
        gen = LoadGenerator(config)
        reqs = gen.generate()

        self.assertEqual(len(reqs), 100)
        # All seqlens should be exactly the specified values
        for r in reqs:
            self.assertEqual(r.input_seqlen, 512)
            self.assertEqual(r.output_seqlen, 128)
        # First timestamp is 0
        self.assertEqual(reqs[0].enqueue_timestamp, 0)
        # Timestamps should be monotonically non-decreasing
        for i in range(1, len(reqs)):
            self.assertGreaterEqual(reqs[i].enqueue_timestamp, reqs[i - 1].enqueue_timestamp)
        # With rate=10 RPS, last timestamp should be roughly around (100/10)=10 seconds
        last_ts_sec = reqs[-1].enqueue_timestamp / 1e9
        self.assertGreater(last_ts_sec, 1.0)  # at least 1 second
        self.assertLess(last_ts_sec, 30.0)  # not unreasonably large

    def test_inf_rate_all_at_zero(self):
        config = self._make_config(
            synthetic_num_requests=50,
            synthetic_request_rate=inf,
            synthetic_input_len=256,
            synthetic_output_len=64,
        )
        gen = LoadGenerator(config)
        reqs = gen.generate()

        self.assertEqual(len(reqs), 50)
        for r in reqs:
            self.assertEqual(r.enqueue_timestamp, 0)
            self.assertEqual(r.input_seqlen, 256)
            self.assertEqual(r.output_seqlen, 64)

    def test_variable_seqlens(self):
        config = self._make_config(
            synthetic_num_requests=200,
            synthetic_request_rate=20.0,
            synthetic_input_len=512,
            synthetic_input_len_std=100,
            synthetic_output_len=128,
            synthetic_output_len_std=30,
            synthetic_seed=123,
        )
        gen = LoadGenerator(config)
        reqs = gen.generate()

        self.assertEqual(len(reqs), 200)
        input_lens = [r.input_seqlen for r in reqs]
        output_lens = [r.output_seqlen for r in reqs]
        # There should be variation in seqlens
        self.assertGreater(len(set(input_lens)), 1)
        self.assertGreater(len(set(output_lens)), 1)
        # Min bounds should be respected
        self.assertGreaterEqual(min(input_lens), 1)
        self.assertGreaterEqual(min(output_lens), 2)

    def test_reproducibility(self):
        config = self._make_config(
            synthetic_num_requests=50,
            synthetic_request_rate=5.0,
            synthetic_input_len=1024,
            synthetic_input_len_std=200,
            synthetic_output_len=256,
            synthetic_output_len_std=50,
            synthetic_seed=99,
        )
        reqs1 = LoadGenerator(config).generate()
        reqs2 = LoadGenerator(config).generate()

        self.assertEqual(len(reqs1), len(reqs2))
        for r1, r2 in zip(reqs1, reqs2, strict=False):
            self.assertEqual(r1.input_seqlen, r2.input_seqlen)
            self.assertEqual(r1.output_seqlen, r2.output_seqlen)
            self.assertEqual(r1.enqueue_timestamp, r2.enqueue_timestamp)
