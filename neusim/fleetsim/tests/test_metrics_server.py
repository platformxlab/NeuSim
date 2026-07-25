"""Unit tests for MetricsServer time window logic.

Tests the three key behaviors introduced by the in-flight request retention fix:
1. evict_stale_from_time_window() only evicts requests whose phase is finished.
2. decode_request_queue_size_change() adds decode requests at first iteration start.
3. decode_iteration_end() no longer adds requests to the decode time window.
"""

import unittest
from collections import deque
from unittest.mock import MagicMock

from neusim.fleetsim.LLMInferenceEvents import (
    LLMInferenceDecodeIterationEndEvent,
    LLMInferenceDecodeIterationStartEvent,
)
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.fleetsim.MetricsServer import MetricsServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(enqueue_ts: int, input_seqlen: int = 512, output_seqlen: int = 128) -> LLMRequest:
    """Create a fresh LLMRequest with a given enqueue timestamp."""
    return LLMRequest(input_seqlen=input_seqlen, output_seqlen=output_seqlen, timestamp=enqueue_ts)


def _make_prefill_in_progress(enqueue_ts: int, prefill_start_ts: int) -> LLMRequest:
    """Create a request that has started but not finished prefill."""
    req = _make_request(enqueue_ts)
    req.mark_prefill_started(prefill_start_ts)
    assert not req.is_prefill_finished()
    return req


def _make_prefill_finished(enqueue_ts: int, prefill_start_ts: int, prefill_end_ts: int) -> LLMRequest:
    """Create a request that has finished prefill but not started decode."""
    req = _make_request(enqueue_ts)
    req.mark_prefill_started(prefill_start_ts)
    req.mark_prefill_finished(prefill_end_ts)
    assert req.is_prefill_finished()
    assert not req.is_decode_started()
    return req


def _make_decode_in_progress(
    enqueue_ts: int, prefill_start_ts: int, prefill_end_ts: int,
    decode_start_ts: int, decode_steps: int = 1, output_seqlen: int = 128,
) -> LLMRequest:
    """Create a request that is mid-decode (some iterations done, not finished)."""
    req = _make_request(enqueue_ts, output_seqlen=output_seqlen)
    req.mark_prefill_started(prefill_start_ts)
    req.mark_prefill_finished(prefill_end_ts)
    req.mark_decode_iteration_started(decode_start_ts)
    if decode_steps > 0:
        req.mark_decode_iteration_finished(decode_start_ts + 100, num_iterations=decode_steps)
    assert req.is_decode_started()
    assert not req.is_decode_finished()
    return req


def _make_decode_first_iteration(
    enqueue_ts: int, prefill_start_ts: int, prefill_end_ts: int,
    decode_start_ts: int, output_seqlen: int = 128,
) -> LLMRequest:
    """Create a request at the very start of its first decode iteration.

    After mark_decode_iteration_started, current_decode_step is still 1
    (set by mark_prefill_finished). This is what MetricsServer sees at
    priority 999 for the DecodeIterationStartEvent.
    """
    req = _make_request(enqueue_ts, output_seqlen=output_seqlen)
    req.mark_prefill_started(prefill_start_ts)
    req.mark_prefill_finished(prefill_end_ts)
    req.mark_decode_iteration_started(decode_start_ts)
    assert req.current_decode_step == 1
    return req


def _make_decode_finished(
    enqueue_ts: int, prefill_start_ts: int, prefill_end_ts: int,
    decode_start_ts: int, decode_end_ts: int, output_seqlen: int = 128,
) -> LLMRequest:
    """Create a fully finished request."""
    req = _make_request(enqueue_ts, output_seqlen=output_seqlen)
    req.mark_prefill_started(prefill_start_ts)
    req.mark_prefill_finished(prefill_end_ts)
    req.mark_decode_iteration_started(decode_start_ts)
    req.mark_decode_iteration_finished(
        decode_end_ts, num_iterations=output_seqlen - 1
    )
    assert req.is_decode_finished()
    return req


def _mock_simulator(hs_window_min: float = 1.0, vs_window_min: float = 1.0):
    """Build a minimal mock simulator for MetricsServer construction."""
    sim = MagicMock()
    sim.config.workload_config.hs_window_minutes = hs_window_min
    sim.config.workload_config.vs_window_minutes = vs_window_min
    sim.name = "test_sim"
    sim.timestamp = 0
    sim.llm_inference_endpoint.decode_request_queue = []
    sim.llm_inference_endpoint.prefill_request_queue = []
    return sim


def _make_metrics_server(hs_window_min: float = 1.0, vs_window_min: float = 1.0) -> MetricsServer:
    """Construct a MetricsServer backed by a mock simulator."""
    sim = _mock_simulator(hs_window_min, vs_window_min)
    return MetricsServer(sim, name="test_metrics_server")


class TestRequestStatsSummary(unittest.TestCase):
    def test_throughput_uses_request_completion_not_simulator_timer_horizon(self):
        metrics = _make_metrics_server()
        metrics.simulator.timestamp = int(30 * 60 * 1e9)
        request = _make_decode_finished(0, 0, 10, 10, 20, output_seqlen=2)
        request.prefill_energy_J = 2.0
        request.decode_energy_per_token_J = [1.0]
        request.prefill_cost_dollars = 0.5
        request.decode_cost_per_token_dollars = [0.25]
        metrics.request_trace = [request]

        stats = metrics.get_request_stats_summary()

        self.assertEqual(stats["throughput_rps"], 1 / (20 / 1e9))
        self.assertEqual(stats["prefill_throughput_tps"], 512 / (20 / 1e9))
        self.assertEqual(stats["decode_throughput_tps"], 1 / (20 / 1e9))
        self.assertEqual(stats["decode_token_per_joule"], 1.0)
        self.assertEqual(stats["decode_token_per_dollar"], 4.0)

    def test_empty_summary_has_clear_error(self):
        metrics = _make_metrics_server()
        with self.assertRaisesRegex(ValueError, "No completed requests"):
            metrics.get_request_stats_summary()


# ===========================================================================
# Tests for evict_stale_from_time_window()
# ===========================================================================

class TestUpdateTimeWindowRequests(unittest.TestCase):
    """Tests for the completion-guarded eviction in evict_stale_from_time_window()."""

    def setUp(self):
        self.ms = _make_metrics_server()
        self.window_ns = 1_000  # 1 μs window for easy arithmetic

    # --- basic behavior ---

    def test_empty_queue_append(self):
        """New requests are appended to an empty queue."""
        q: deque[LLMRequest] = deque()
        req = _make_request(enqueue_ts=100)
        q.extend([req])
        self.assertEqual(len(q), 1)
        self.assertIs(q[0], req)

    def test_within_window_kept(self):
        """Requests within the time window are never evicted regardless of state."""
        q: deque[LLMRequest] = deque()
        # Finished prefill request, still inside window
        req = _make_prefill_finished(enqueue_ts=500, prefill_start_ts=600, prefill_end_ts=700)
        q.append(req)
        self.ms.evict_stale_from_time_window(q, current_timestamp=1000, time_window_ns=self.window_ns, prefill_or_decode="prefill")
        self.assertEqual(len(q), 1, "Request inside window should not be evicted")

    # --- prefill eviction ---

    def test_prefill_finished_outside_window_evicted(self):
        """Finished prefill request outside the window is evicted."""
        q: deque[LLMRequest] = deque()
        req = _make_prefill_finished(enqueue_ts=100, prefill_start_ts=200, prefill_end_ts=300)
        q.append(req)
        self.ms.evict_stale_from_time_window(q, current_timestamp=5000, time_window_ns=self.window_ns, prefill_or_decode="prefill")
        self.assertEqual(len(q), 0, "Finished prefill request outside window should be evicted")

    def test_prefill_in_progress_outside_window_retained(self):
        """In-progress prefill request outside the window is kept (in-flight retention)."""
        q: deque[LLMRequest] = deque()
        req = _make_prefill_in_progress(enqueue_ts=100, prefill_start_ts=200)
        q.append(req)
        self.ms.evict_stale_from_time_window(q, current_timestamp=5000, time_window_ns=self.window_ns, prefill_or_decode="prefill")
        self.assertEqual(len(q), 1, "In-progress prefill request must be retained")

    def test_prefill_not_started_outside_window_retained(self):
        """Request that hasn't started prefill outside the window is kept."""
        q: deque[LLMRequest] = deque()
        req = _make_request(enqueue_ts=100)  # not started at all
        q.append(req)
        self.ms.evict_stale_from_time_window(q, current_timestamp=5000, time_window_ns=self.window_ns, prefill_or_decode="prefill")
        self.assertEqual(len(q), 1, "Not-started request must be retained")

    # --- decode eviction ---

    def test_decode_finished_outside_window_evicted(self):
        """Finished decode request outside the window is evicted."""
        q: deque[LLMRequest] = deque()
        req = _make_decode_finished(enqueue_ts=100, prefill_start_ts=200, prefill_end_ts=300,
                                     decode_start_ts=400, decode_end_ts=500, output_seqlen=10)
        q.append(req)
        self.ms.evict_stale_from_time_window(q, current_timestamp=5000, time_window_ns=self.window_ns, prefill_or_decode="decode")
        self.assertEqual(len(q), 0, "Finished decode request outside window should be evicted")

    def test_decode_in_progress_outside_window_retained(self):
        """In-progress decode request outside the window is kept (in-flight retention)."""
        q: deque[LLMRequest] = deque()
        req = _make_decode_in_progress(enqueue_ts=100, prefill_start_ts=200, prefill_end_ts=300,
                                        decode_start_ts=400, decode_steps=5, output_seqlen=128)
        q.append(req)
        self.ms.evict_stale_from_time_window(q, current_timestamp=5000, time_window_ns=self.window_ns, prefill_or_decode="decode")
        self.assertEqual(len(q), 1, "In-progress decode request must be retained")

    # --- mixed ordering / stop-at-first-unfinished ---

    def test_eviction_skips_unfinished_evicts_finished_behind(self):
        """Eviction skips in-flight requests but still evicts finished ones behind them."""
        q: deque[LLMRequest] = deque()
        # Request A: finished, outside window → should be evicted
        req_a = _make_decode_finished(enqueue_ts=100, prefill_start_ts=200, prefill_end_ts=300,
                                       decode_start_ts=400, decode_end_ts=500, output_seqlen=10)
        # Request B: in-progress, outside window → should NOT be evicted
        req_b = _make_decode_in_progress(enqueue_ts=200, prefill_start_ts=300, prefill_end_ts=400,
                                          decode_start_ts=500, decode_steps=5, output_seqlen=128)
        # Request C: finished, outside window → should be evicted (not blocked by B)
        req_c = _make_decode_finished(enqueue_ts=300, prefill_start_ts=400, prefill_end_ts=500,
                                       decode_start_ts=600, decode_end_ts=700, output_seqlen=10)
        q.extend([req_a, req_b, req_c])
        self.ms.evict_stale_from_time_window(q, current_timestamp=5000, time_window_ns=self.window_ns, prefill_or_decode="decode")
        self.assertEqual(len(q), 1, "Should evict A and C, keep only in-flight B")
        self.assertIs(q[0], req_b)

    def test_multiple_finished_evicted(self):
        """Multiple consecutive finished requests outside the window are all evicted."""
        q: deque[LLMRequest] = deque()
        for i in range(5):
            req = _make_decode_finished(
                enqueue_ts=100 + i, prefill_start_ts=200 + i, prefill_end_ts=300 + i,
                decode_start_ts=400 + i, decode_end_ts=500 + i, output_seqlen=10)
            q.append(req)
        # One in-progress at the end
        req_inflight = _make_decode_in_progress(
            enqueue_ts=200, prefill_start_ts=300, prefill_end_ts=400,
            decode_start_ts=500, decode_steps=5, output_seqlen=128)
        q.append(req_inflight)

        self.ms.evict_stale_from_time_window(q, current_timestamp=5000, time_window_ns=self.window_ns, prefill_or_decode="decode")
        self.assertEqual(len(q), 1, "All finished should be evicted, in-flight kept")
        self.assertIs(q[0], req_inflight)

    def test_new_requests_appended_after_eviction(self):
        """New requests are appended even when some old ones are evicted."""
        q: deque[LLMRequest] = deque()
        old = _make_decode_finished(enqueue_ts=100, prefill_start_ts=200, prefill_end_ts=300,
                                     decode_start_ts=400, decode_end_ts=500, output_seqlen=10)
        q.append(old)
        new = _make_request(enqueue_ts=4500)
        self.ms.evict_stale_from_time_window(q, current_timestamp=5000, time_window_ns=self.window_ns, prefill_or_decode="decode")
        q.extend([new])
        self.assertEqual(len(q), 1)
        self.assertIs(q[0], new)

    # --- cross-phase: prefill mode ignores decode state ---

    def test_prefill_mode_evicts_when_prefill_done_even_if_decode_not_done(self):
        """In prefill mode, a request with prefill finished but decode in progress is evictable."""
        q: deque[LLMRequest] = deque()
        req = _make_decode_in_progress(enqueue_ts=100, prefill_start_ts=200, prefill_end_ts=300,
                                        decode_start_ts=400, decode_steps=5, output_seqlen=128)
        assert req.is_prefill_finished()
        assert not req.is_decode_finished()
        q.append(req)
        self.ms.evict_stale_from_time_window(q, current_timestamp=5000, time_window_ns=self.window_ns, prefill_or_decode="prefill")
        self.assertEqual(len(q), 0, "Prefill mode should evict once prefill is done")


# ===========================================================================
# Tests for decode_request_queue_size_change()
# ===========================================================================

class TestDecodeRequestQueueSizeChange(unittest.TestCase):
    """Tests that decode requests enter the time window at first iteration start."""

    def setUp(self):
        self.ms = _make_metrics_server(hs_window_min=1.0, vs_window_min=1.0)

    def test_first_iteration_added_to_time_window(self):
        """Requests on their first decode iteration (current_decode_step==1) are added."""
        req = _make_decode_first_iteration(
            enqueue_ts=0, prefill_start_ts=100, prefill_end_ts=200,
            decode_start_ts=300, output_seqlen=128)
        event = LLMInferenceDecodeIterationStartEvent(
            requests=[req], timestamp=300, worker_id="w1", num_iterations=4)
        self.ms.decode_request_queue_size_change(event)
        self.assertEqual(len(self.ms.decode_requests_in_mpa_time_window), 1)
        self.assertIs(self.ms.decode_requests_in_mpa_time_window[0], req)

    def test_subsequent_iteration_not_readded(self):
        """Requests past their first iteration (current_decode_step>1) are NOT re-added."""
        req = _make_decode_in_progress(
            enqueue_ts=0, prefill_start_ts=100, prefill_end_ts=200,
            decode_start_ts=300, decode_steps=4, output_seqlen=128)
        assert req.current_decode_step > 1
        event = LLMInferenceDecodeIterationStartEvent(
            requests=[req], timestamp=600, worker_id="w1", num_iterations=4)
        self.ms.decode_request_queue_size_change(event)
        self.assertEqual(len(self.ms.decode_requests_in_mpa_time_window), 0,
                         "Requests past first iteration should not be re-added")

    def test_mixed_batch_only_first_iteration_added(self):
        """In a batch with mixed iteration counts, only first-iteration requests are added."""
        req_first = _make_decode_first_iteration(
            enqueue_ts=0, prefill_start_ts=100, prefill_end_ts=200,
            decode_start_ts=300, output_seqlen=128)
        req_later = _make_decode_in_progress(
            enqueue_ts=0, prefill_start_ts=100, prefill_end_ts=200,
            decode_start_ts=300, decode_steps=10, output_seqlen=128)
        event = LLMInferenceDecodeIterationStartEvent(
            requests=[req_first, req_later], timestamp=400, worker_id="w1", num_iterations=4)
        self.ms.decode_request_queue_size_change(event)
        self.assertEqual(len(self.ms.decode_requests_in_mpa_time_window), 1)
        self.assertIs(self.ms.decode_requests_in_mpa_time_window[0], req_first)

    def test_end_event_triggers_cleanup_only(self):
        """DecodeIterationEndEvent marks window dirty; eviction happens on read."""
        # Pre-populate the deque with an old finished request outside the window
        old_req = _make_decode_finished(
            enqueue_ts=0, prefill_start_ts=100, prefill_end_ts=200,
            decode_start_ts=300, decode_end_ts=400, output_seqlen=10)
        self.ms.decode_requests_in_mpa_time_window.append(old_req)
        # The MPA window is max(1,1) min = 60s = 60_000_000_000 ns
        # Set timestamp far enough in the future to expire old_req
        far_future_ts = 120_000_000_000  # 120s
        event = LLMInferenceDecodeIterationEndEvent(
            requests=[old_req], timestamp=far_future_ts, worker_id="w1", num_iterations=10)
        self.ms.decode_request_queue_size_change(event)
        # With lazy eviction, the deque still has the stale entry
        self.assertEqual(len(self.ms.decode_requests_in_mpa_time_window), 1)
        # But calling evict_stale_mpa_requests cleans it up
        self.ms.evict_stale_mpa_requests(far_future_ts, "decode")
        self.assertEqual(len(self.ms.decode_requests_in_mpa_time_window), 0,
                         "Eviction on read should remove finished expired requests")

    def test_end_event_does_not_add_finished_requests(self):
        """DecodeIterationEndEvent does not add any requests, even finished ones."""
        req = _make_decode_finished(
            enqueue_ts=1000, prefill_start_ts=1100, prefill_end_ts=1200,
            decode_start_ts=1300, decode_end_ts=1400, output_seqlen=10)
        event = LLMInferenceDecodeIterationEndEvent(
            requests=[req], timestamp=1400, worker_id="w1", num_iterations=10)
        self.ms.decode_request_queue_size_change(event)
        self.assertEqual(len(self.ms.decode_requests_in_mpa_time_window), 0,
                         "End event must not add finished requests to the window")

    def test_queue_lengths_recorded_for_both_event_types(self):
        """Both start and end events record to decode_queue_lengths."""
        req = _make_decode_first_iteration(
            enqueue_ts=0, prefill_start_ts=100, prefill_end_ts=200,
            decode_start_ts=300, output_seqlen=128)
        start_event = LLMInferenceDecodeIterationStartEvent(
            requests=[req], timestamp=300, worker_id="w1", num_iterations=4)
        self.ms.decode_request_queue_size_change(start_event)

        end_event = LLMInferenceDecodeIterationEndEvent(
            requests=[req], timestamp=400, worker_id="w1", num_iterations=4)
        self.ms.decode_request_queue_size_change(end_event)

        self.assertEqual(len(self.ms.decode_queue_lengths), 2)
        self.assertEqual(self.ms.decode_queue_lengths[0][0], 300)
        self.assertEqual(self.ms.decode_queue_lengths[1][0], 400)


# ===========================================================================
# Tests for decode_iteration_end()
# ===========================================================================

class TestDecodeIterationEnd(unittest.TestCase):
    """Tests that decode_iteration_end() records finished requests to request_trace
    but does NOT add them to the decode time window."""

    def setUp(self):
        self.ms = _make_metrics_server()

    def test_finished_request_added_to_trace(self):
        """Finished requests are appended to request_trace."""
        req = _make_decode_finished(
            enqueue_ts=0, prefill_start_ts=100, prefill_end_ts=200,
            decode_start_ts=300, decode_end_ts=400, output_seqlen=10)
        event = LLMInferenceDecodeIterationEndEvent(
            requests=[req], timestamp=400, worker_id="w1", num_iterations=10)
        self.ms.decode_iteration_end(event)
        self.assertEqual(len(self.ms.request_trace), 1)
        self.assertIs(self.ms.request_trace[0], req)

    def test_finished_request_not_added_to_decode_window(self):
        """Finished requests must NOT be added to the decode time window deque."""
        req = _make_decode_finished(
            enqueue_ts=0, prefill_start_ts=100, prefill_end_ts=200,
            decode_start_ts=300, decode_end_ts=400, output_seqlen=10)
        event = LLMInferenceDecodeIterationEndEvent(
            requests=[req], timestamp=400, worker_id="w1", num_iterations=10)
        self.ms.decode_iteration_end(event)
        self.assertEqual(len(self.ms.decode_requests_in_mpa_time_window), 0,
                         "decode_iteration_end should not add to decode window")

    def test_unfinished_requests_ignored(self):
        """Requests that are not yet finished are NOT added to request_trace."""
        req = _make_decode_in_progress(
            enqueue_ts=0, prefill_start_ts=100, prefill_end_ts=200,
            decode_start_ts=300, decode_steps=5, output_seqlen=128)
        event = LLMInferenceDecodeIterationEndEvent(
            requests=[req], timestamp=500, worker_id="w1", num_iterations=5)
        self.ms.decode_iteration_end(event)
        self.assertEqual(len(self.ms.request_trace), 0)

    def test_mixed_batch_only_finished_in_trace(self):
        """In a batch, only finished requests go to request_trace."""
        finished = _make_decode_finished(
            enqueue_ts=0, prefill_start_ts=100, prefill_end_ts=200,
            decode_start_ts=300, decode_end_ts=400, output_seqlen=10)
        in_progress = _make_decode_in_progress(
            enqueue_ts=0, prefill_start_ts=100, prefill_end_ts=200,
            decode_start_ts=300, decode_steps=5, output_seqlen=128)
        event = LLMInferenceDecodeIterationEndEvent(
            requests=[finished, in_progress], timestamp=400, worker_id="w1", num_iterations=10)
        self.ms.decode_iteration_end(event)
        self.assertEqual(len(self.ms.request_trace), 1)
        self.assertIs(self.ms.request_trace[0], finished)
        self.assertEqual(len(self.ms.decode_requests_in_mpa_time_window), 0)


# ===========================================================================
# End-to-end scenario test
# ===========================================================================

class TestTimeWindowEndToEnd(unittest.TestCase):
    """Scenario test: walk a request through its full lifecycle and verify
    time window membership at each stage."""

    def test_request_lifecycle(self):
        ms = _make_metrics_server(hs_window_min=0.001, vs_window_min=0.001)
        # MPA window = max(0.001, 0.001) min = 0.06s = 60_000_000 ns
        window_ns = ms.mpa_time_window_ns
        self.assertEqual(window_ns, 60_000_000)

        req = _make_request(enqueue_ts=0, output_seqlen=10)

        # 1) Prefill start at t=10ms → request enters prefill window
        req.mark_prefill_started(10_000_000)
        ms.update_all_autoscaling_time_windows_prefill(10_000_000, [req])
        self.assertEqual(len(ms.prefill_requests_in_mpa_time_window), 1)
        self.assertEqual(len(ms.decode_requests_in_mpa_time_window), 0)

        # 2) Prefill end at t=20ms → request stays (within window, finished)
        req.mark_prefill_finished(20_000_000)
        ms.update_all_autoscaling_time_windows_prefill(20_000_000, [])
        ms.evict_stale_mpa_requests(20_000_000, "prefill")
        self.assertEqual(len(ms.prefill_requests_in_mpa_time_window), 1)

        # 3) Prefill window at t=100ms → request outside 60ms window AND prefill finished → evicted
        ms.update_all_autoscaling_time_windows_prefill(100_000_000, [])
        ms.evict_stale_mpa_requests(100_000_000, "prefill")
        self.assertEqual(len(ms.prefill_requests_in_mpa_time_window), 0)

        # 4) First decode iteration start at t=30ms (simulate the MetricsServer handler)
        req2 = _make_request(enqueue_ts=5_000_000, output_seqlen=10)
        req2.mark_prefill_started(10_000_000)
        req2.mark_prefill_finished(20_000_000)
        req2.mark_decode_iteration_started(30_000_000)
        assert req2.current_decode_step == 1
        start_event = LLMInferenceDecodeIterationStartEvent(
            requests=[req2], timestamp=30_000_000, worker_id="w1", num_iterations=4)
        ms.decode_request_queue_size_change(start_event)
        self.assertEqual(len(ms.decode_requests_in_mpa_time_window), 1)

        # 5) At t=100ms (70ms after enqueue=5ms → outside 60ms window),
        #    request is still in-progress → must NOT be evicted
        ms.update_all_autoscaling_time_windows_decode(100_000_000, [])
        ms.evict_stale_mpa_requests(100_000_000, "decode")
        self.assertEqual(len(ms.decode_requests_in_mpa_time_window), 1,
                         "In-flight decode request must survive past the time window")

        # 6) Finish decode, then cleanup → now it should be evicted
        req2.mark_decode_iteration_finished(110_000_000, num_iterations=9)
        assert req2.is_decode_finished()
        ms.update_all_autoscaling_time_windows_decode(120_000_000, [])
        ms.evict_stale_mpa_requests(120_000_000, "decode")
        self.assertEqual(len(ms.decode_requests_in_mpa_time_window), 0,
                         "Finished decode request outside the window should be evicted")

    def test_in_flight_retention_duration(self):
        """An in-flight decode request is retained across many cleanup cycles,
        then evicted once it finishes."""
        ms = _make_metrics_server(hs_window_min=0.001, vs_window_min=0.001)

        req = _make_decode_first_iteration(
            enqueue_ts=0, prefill_start_ts=10_000_000, prefill_end_ts=20_000_000,
            decode_start_ts=30_000_000, output_seqlen=100)

        # Add at first decode iteration start
        start_event = LLMInferenceDecodeIterationStartEvent(
            requests=[req], timestamp=30_000_000, worker_id="w1", num_iterations=4)
        ms.decode_request_queue_size_change(start_event)
        self.assertEqual(len(ms.decode_requests_in_mpa_time_window), 1)

        # Simulate many cleanup cycles well past the time window
        for t_ms in range(100, 600, 50):
            t_ns = t_ms * 1_000_000
            ms.update_all_autoscaling_time_windows_decode(t_ns, [])
            ms.evict_stale_mpa_requests(t_ns, "decode")
            self.assertEqual(len(ms.decode_requests_in_mpa_time_window), 1,
                             f"In-flight request must survive cleanup at t={t_ms}ms")

        # Finish the request
        req.mark_decode_iteration_finished(600_000_000, num_iterations=99)
        assert req.is_decode_finished()

        # Now cleanup should evict it (enqueue_ts=0 is far outside window at t=700ms)
        ms.update_all_autoscaling_time_windows_decode(700_000_000, [])
        ms.evict_stale_mpa_requests(700_000_000, "decode")
        self.assertEqual(len(ms.decode_requests_in_mpa_time_window), 0)


if __name__ == "__main__":
    unittest.main()
