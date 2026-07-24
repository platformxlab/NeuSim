"""Tests for static-fleet request completion metrics."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from neusim.fleetsim.LLMInferenceEvents import LLMInferenceDecodeIterationEndEvent
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.fleetsim.MetricsServer import MetricsServer


def _make_request(
    enqueue_ts: int,
    input_seqlen: int = 512,
    output_seqlen: int = 128,
) -> LLMRequest:
    return LLMRequest(
        input_seqlen=input_seqlen,
        output_seqlen=output_seqlen,
        timestamp=enqueue_ts,
    )


def _make_decode_in_progress(
    *,
    enqueue_ts: int = 0,
    prefill_start_ts: int = 100,
    prefill_end_ts: int = 200,
    decode_start_ts: int = 300,
    decode_steps: int = 1,
    output_seqlen: int = 128,
) -> LLMRequest:
    request = _make_request(enqueue_ts, output_seqlen=output_seqlen)
    request.mark_prefill_started(prefill_start_ts)
    request.mark_prefill_finished(prefill_end_ts)
    request.mark_decode_iteration_started(decode_start_ts)
    request.mark_decode_iteration_finished(
        decode_start_ts + 100,
        num_iterations=decode_steps,
    )
    return request


def _make_decode_finished(
    *,
    enqueue_ts: int = 0,
    prefill_start_ts: int = 100,
    prefill_end_ts: int = 200,
    decode_start_ts: int = 300,
    decode_end_ts: int = 400,
    output_seqlen: int = 10,
) -> LLMRequest:
    request = _make_request(enqueue_ts, output_seqlen=output_seqlen)
    request.mark_prefill_started(prefill_start_ts)
    request.mark_prefill_finished(prefill_end_ts)
    request.mark_decode_iteration_started(decode_start_ts)
    request.mark_decode_iteration_finished(
        decode_end_ts,
        num_iterations=output_seqlen - 1,
    )
    return request


def _make_metrics_server(output_dir: str = "/tmp") -> MetricsServer:
    simulator = MagicMock()
    simulator.config.workload_config = MagicMock()
    simulator.config.output_dir = output_dir
    simulator.name = "test_sim"
    simulator.timestamp = 0
    simulator.event_queue_length.return_value = 0
    simulator.num_event_listeners.return_value = 0
    simulator.llm_inference_endpoint.decode_request_queue = []
    simulator.llm_inference_endpoint.prefill_request_queue = []
    simulator.llm_inference_endpoint.prefill_vpods = {}
    simulator.llm_inference_endpoint.decode_vpods = {}
    return MetricsServer(simulator, name="test_metrics_server")


class TestRequestStatsSummary(unittest.TestCase):
    def test_throughput_uses_request_completion_not_timer_horizon(self) -> None:
        metrics = _make_metrics_server()
        metrics.simulator.timestamp = int(30 * 60 * 1e9)
        request = _make_decode_finished(output_seqlen=2, decode_end_ts=20)
        request.prefill_energy_J = 2.0
        request.decode_energy_J = 1.0
        request.prefill_cost_dollars = 0.5
        request.decode_cost_dollars = 0.25
        metrics.request_trace = [request]

        stats = metrics.get_request_stats_summary()

        self.assertEqual(stats["throughput_rps"], 1 / (20 / 1e9))
        self.assertEqual(stats["prefill_throughput_tps"], 512 / (20 / 1e9))
        self.assertEqual(stats["decode_throughput_tps"], 1 / (20 / 1e9))
        self.assertEqual(stats["decode_token_per_joule"], 1.0)
        self.assertEqual(stats["decode_token_per_dollar"], 4.0)

    def test_empty_summary_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "No completed requests"):
            _make_metrics_server().get_request_stats_summary()


class TestStaticMetrics(unittest.TestCase):
    def test_decode_end_records_only_finished_requests(self) -> None:
        metrics = _make_metrics_server()
        finished = _make_decode_finished()
        unfinished = _make_decode_in_progress(decode_steps=5)
        event = LLMInferenceDecodeIterationEndEvent(
            requests=[finished, unfinished],
            timestamp=400,
            worker_id="worker",
            num_iterations=5,
        )

        metrics.decode_iteration_end(event)

        self.assertEqual(metrics.request_trace, [finished])

    def test_safeguard_window_tracks_violations_in_constant_time(self) -> None:
        metrics = _make_metrics_server()
        metrics.dvfs_slo_window_ns = 100
        metrics.workload_config.dvfs_safeguard_violation_threshold = 0.5
        outcomes = iter([False, True, False, True])
        metrics._check_request_slo = lambda _request: next(outcomes)
        request = _make_request(0)

        metrics.update_dvfs_slo_window(0, request)
        self.assertTrue(metrics.dvfs_locked_to_peak)
        self.assertEqual(metrics.dvfs_slo_violation_count, 1)

        metrics.update_dvfs_slo_window(50, request)
        self.assertFalse(metrics.dvfs_locked_to_peak)  # exactly 50%, not above

        metrics.update_dvfs_slo_window(100, request)
        self.assertTrue(metrics.dvfs_locked_to_peak)
        self.assertEqual(metrics.dvfs_slo_violation_count, 2)

        # At t=151 both t=0 and t=50 are stale. The violation at t=100 and
        # the new compliant request remain, giving exactly 50% again.
        metrics.update_dvfs_slo_window(151, request)
        self.assertFalse(metrics.dvfs_locked_to_peak)
        self.assertEqual(list(metrics.dvfs_slo_records), [(100, True), (151, False)])
        self.assertEqual(metrics.dvfs_slo_violation_count, 1)

    def test_request_trace_csv_contains_figure_5_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metrics = _make_metrics_server(directory)
            request = _make_decode_finished(output_seqlen=2)
            request.prefill_energy_J = 2.0
            request.decode_energy_J = 1.0
            request.prefill_cost_dollars = 0.5
            request.decode_cost_dollars = 0.25
            request.ideal_TTFT_ns = request.TTFT_ns()
            request.ideal_TPOT_ns = request.TPOT_ns()
            metrics.request_trace = [request]
            output = Path(directory) / "request_trace.csv"

            metrics.dump_request_trace_to_csv(str(output))

            with output.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                row = next(reader)
            required = {
                "input_seqlen",
                "output_seqlen",
                "prefill_end_timestamp",
                "decode_end_timestamp",
                "TTFT_ns",
                "TPOT_ns",
                "effective_TPOT_ns",
                "prefill_energy_J",
                "decode_energy_J",
                "decode_energy_per_token_J",
            }
            self.assertTrue(required.issubset(row))
            self.assertEqual(float(row["decode_energy_J"]), 1.0)
            self.assertEqual(float(row["decode_energy_per_token_J"]), 0.5)


if __name__ == "__main__":
    unittest.main()
