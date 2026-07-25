import csv
import gzip
import json
import os
import pickle
import typing
from collections import deque
from copy import deepcopy

import numpy as np
from absl import logging

from neusim.fleetsim.LLMInferenceEvents import (
    LLMInferenceDecodeIterationEndEvent,
    LLMInferenceDecodeIterationStartEvent,
    LLMInferencePrefillEndEvent,
    LLMInferencePrefillStartEvent,
    LLMInferenceRequestEnqueueEvent,
)
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.fleetsim.SimObject import SimObject

if typing.TYPE_CHECKING:
    from neusim.fleetsim.NPUFleetSimulator import NPUFleetSimulator
from collections.abc import Iterable

from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
)


class MetricsServer(SimObject):
    """
    Metrics server for monitoring the metrics used for auto-scaling.
    """

    def __init__(self, simulator: "NPUFleetSimulator", name: str | None = None):
        super().__init__(name or "metrics_server", simulator)

        self.workload_config: LLMInferenceWorkloadConfig = deepcopy(
            simulator.config.workload_config
        )

        self.request_trace: list[LLMRequest] = []
        """List of completed LLMRequest objects for trace analysis."""

        self.prefill_requests_in_hs_time_window: deque[LLMRequest] = deque()
        """A deque of LLMRequests within the last HS time window for calculating autoscaling decisions."""
        self.prefill_requests_in_vs_time_window: deque[LLMRequest] = deque()
        """A deque of LLMRequests within the last VS time window for calculating autoscaling decisions."""
        self.prefill_requests_in_mpa_time_window: deque[LLMRequest] = deque()
        """A deque of LLMRequests within the last MPA time window for calculating autoscaling decisions."""
        self._prefill_mpa_dirty: bool = False
        """Flag indicating the prefill MPA window has un-evicted stale entries."""
        self.decode_requests_in_hs_time_window: deque[LLMRequest] = deque()
        """A deque of LLMRequests within the last HS time window for calculating autoscaling decisions."""
        self.decode_requests_in_vs_time_window: deque[LLMRequest] = deque()
        """A deque of LLMRequests within the last VS time window for calculating autoscaling decisions."""
        self.decode_requests_in_mpa_time_window: deque[LLMRequest] = deque()
        """A deque of LLMRequests within the last MPA time window for calculating autoscaling decisions."""
        self._decode_mpa_dirty: bool = False
        """Flag indicating the decode MPA window has un-evicted stale entries."""

        # monitoring
        self.prefill_queue_lengths: list[tuple[int, int]] = []
        """[(timestamp, queue_length), ...]. Updated for every LLMInferenceRequestEnqueueEvent."""
        self.decode_queue_lengths: list[tuple[int, int]] = []
        """[(timestamp, queue_length), ...]. Updated for every LLMInferenceDecodeIterationStartEvent."""
        self.prefill_seqlens: list[tuple[int, int]] = []
        """[(timestamp, seq_length), ...]. Updated for every LLMInferenceRequestEnqueueEvent."""
        self.decode_seqlens: list[tuple[int, int]] = []
        """[(timestamp, seq_length), ...]. Updated for every LLMInferenceDecodeIterationStartEvent."""

        # autoscaling time windows
        self.hs_time_window_ns: int = int(
            simulator.config.workload_config.hs_window_minutes * 60 * 1e9
        )
        self.vs_time_window_ns: int = int(
            simulator.config.workload_config.vs_window_minutes * 60 * 1e9
        )
        self.mpa_time_window_ns: int = max(
            self.hs_time_window_ns, self.vs_time_window_ns
        )

    def initialize(self):
        """
        Initialize the metrics server by registering event listeners for metrics updates.
        """
        self.simulator.add_event_listener(
            LLMInferenceDecodeIterationEndEvent.get_type_listener(
                self.decode_iteration_end
            )
        )
        self.simulator.add_event_listener(
            LLMInferenceRequestEnqueueEvent.get_type_listener(
                self.prefill_request_queue_size_change
            )
        )
        self.simulator.add_event_listener(
            LLMInferencePrefillStartEvent.get_type_listener(
                self.prefill_request_queue_size_change
            )
        )
        self.simulator.add_event_listener(
            LLMInferencePrefillEndEvent.get_type_listener(
                self.prefill_request_queue_size_change
            )
        )
        self.simulator.add_event_listener(
            LLMInferenceDecodeIterationStartEvent.get_type_listener(
                self.decode_request_queue_size_change
            )
        )
        self.simulator.add_event_listener(
            LLMInferenceDecodeIterationEndEvent.get_type_listener(
                self.decode_request_queue_size_change
            )
        )

    def evict_stale_from_time_window(
        self,
        time_window_queue: deque[LLMRequest],
        current_timestamp: int,
        time_window_ns: int,
        prefill_or_decode: str = "prefill",
    ):
        """
        Evict requests that are outside the time window AND whose phase is finished.
        In-flight requests are kept even if their enqueue_timestamp is outside the window,
        so that the autoscaler counts capacity still consumed by in-progress requests.

        Called lazily — only when the autoscaler actually reads the window,
        not on every decode/prefill event.
        """
        window_cutoff = current_timestamp - time_window_ns
        # Scan and remove all expired+finished requests.
        new_queue: deque[LLMRequest] = deque()
        for req in time_window_queue:
            if req.enqueue_timestamp < window_cutoff:
                # Outside time window — keep only if still in-flight
                if prefill_or_decode == "prefill" and req.is_prefill_finished():
                    continue  # evict
                elif prefill_or_decode == "decode" and req.is_decode_finished():
                    continue  # evict
            new_queue.append(req)
        time_window_queue.clear()
        time_window_queue.extend(new_queue)

    def evict_stale_mpa_requests(self, current_timestamp: int, prefill_or_decode: str):
        """
        Evict stale requests from the MPA time window for the given mode.
        Should be called before the autoscaler reads the window.
        """
        if prefill_or_decode == "prefill" and self._prefill_mpa_dirty:
            self.evict_stale_from_time_window(
                self.prefill_requests_in_mpa_time_window,
                current_timestamp,
                self.mpa_time_window_ns,
                prefill_or_decode="prefill",
            )
            self._prefill_mpa_dirty = False
        elif prefill_or_decode == "decode" and self._decode_mpa_dirty:
            self.evict_stale_from_time_window(
                self.decode_requests_in_mpa_time_window,
                current_timestamp,
                self.mpa_time_window_ns,
                prefill_or_decode="decode",
            )
            self._decode_mpa_dirty = False

    def update_all_autoscaling_time_windows_prefill(
        self, current_timestamp: int, new_requests: Iterable[LLMRequest]
    ):
        # Lazy append: just extend the deque with new requests (O(1) when empty).
        # Eviction is deferred to evict_stale_mpa_requests(), called before autoscaler reads.
        self.prefill_requests_in_mpa_time_window.extend(new_requests)
        self._prefill_mpa_dirty = True

    def update_all_autoscaling_time_windows_decode(
        self, current_timestamp: int, new_requests: Iterable[LLMRequest]
    ):
        # Lazy append: just extend the deque with new requests (O(1) when empty).
        # Eviction is deferred to evict_stale_mpa_requests(), called before autoscaler reads.
        self.decode_requests_in_mpa_time_window.extend(new_requests)
        self._decode_mpa_dirty = True

    def prefill_request_queue_size_change(
        self,
        event: LLMInferenceRequestEnqueueEvent
        | LLMInferencePrefillStartEvent
        | LLMInferencePrefillEndEvent,
    ):
        """
        Event listener for prefill request enqueue events.
        """
        requests = self.simulator.llm_inference_endpoint.prefill_request_queue  # type: ignore
        item = (event.timestamp, len(requests))  # type: ignore
        self.prefill_queue_lengths.append(item)
        # logging.debug("MetricsServer: Recorded prefill queue length at timestamp %d: %d", item[0], item[1])
        item = (
            event.timestamp,
            int(np.mean([req.input_seqlen for req in requests]))
            if len(requests) > 0
            else 0,
        )
        self.prefill_seqlens.append(item)
        if isinstance(event, LLMInferencePrefillStartEvent):
            self.update_all_autoscaling_time_windows_prefill(
                event.timestamp, event.requests
            )
        elif isinstance(event, LLMInferencePrefillEndEvent):
            self.update_all_autoscaling_time_windows_prefill(event.timestamp, [])

    def decode_request_queue_size_change(
        self,
        event: LLMInferenceDecodeIterationStartEvent
        | LLMInferenceDecodeIterationEndEvent,
    ):
        """
        Event listener for decode iteration start/end events.
        """
        requests = self.simulator.llm_inference_endpoint.decode_request_queue  # type: ignore
        item = (event.timestamp, len(requests))  # type: ignore
        self.decode_queue_lengths.append(item)
        if isinstance(event, LLMInferenceDecodeIterationStartEvent):
            # Add requests starting their first decode iteration to the time window.
            # By the time MetricsServer sees this event (priority=999),
            # mark_decode_iteration_started() has already been called (priority=10),
            # setting decode_start_timestamp. But current_decode_step is still 1
            # for first-iteration requests (mark_decode_iteration_started doesn't change it).
            new_decode_requests = [
                r for r in event.requests if r.current_decode_step == 1
            ]
            self.update_all_autoscaling_time_windows_decode(
                event.timestamp, new_decode_requests
            )
        elif isinstance(event, LLMInferenceDecodeIterationEndEvent):
            # Trigger cleanup only (requests were added at start)
            self.update_all_autoscaling_time_windows_decode(event.timestamp, [])

    def decode_iteration_end(self, event: LLMInferenceDecodeIterationEndEvent):
        """
        Event listener for decode iteration end events.
        Simply add the request to the trace list after the request is finished.
        The latency stats are recorded in the request object itself.
        """
        requests = event.requests
        finished_requests = [
            request for request in requests if request.is_decode_finished()
        ]
        if len(finished_requests) > 0:
            self.request_trace += finished_requests
            if (
                len(self.request_trace) % 100 == 0
                or event.timestamp % int(1e9 * 3600) == 0
            ):  # print every 100 requests or every hour
                logging.info(
                    "%s: Processed %d requests. # prefill vPods: %d. # decode vPods: %d. Prefill queue size: %d. Decode queue size: %d. Remaining events left in sim queue: %d.",
                    self.simulator.name,
                    len(self.request_trace),
                    len(self.simulator.llm_inference_endpoint.prefill_vpods),  # type: ignore
                    len(self.simulator.llm_inference_endpoint.decode_vpods),  # type: ignore
                    len(self.simulator.llm_inference_endpoint.prefill_request_queue),  # type: ignore
                    len(self.simulator.llm_inference_endpoint.decode_request_queue),  # type: ignore
                    self.simulator.event_queue_length(),
                )
                logging.info(
                    "%s: Current timestamp: %s hours.",
                    self.simulator.name,
                    self.simulator.timestamp / (3600 * int(1e9)),
                )
                logging.info(
                    "%s: # event listeners: %d",
                    self.simulator.name,
                    self.simulator.num_event_listeners(),
                )
                if logging.level_debug():
                    # print event listener count by event type
                    logging.debug(
                        "%s: event listener count by event type:\n%s",
                        self.simulator.name,
                        json.dumps(
                            {
                                etype.__name__: len(listeners)
                                for etype, listeners in self.simulator.event_listeners.items()
                            },
                            indent=4,
                        ),
                    )
                    # also print event queue breakdown by event type
                    logging.debug(
                        "%s: Event queue breakdown by event type:\n%s",
                        self.simulator.name,
                        json.dumps(
                            self.simulator.get_event_queue_breakdown_snapshot(),
                            indent=4,
                        ),
                    )
                # also print time window request counts
                logging.info(
                    "%s: Time window request counts: MPA prefill: %d, MPA decode: %d.",
                    self.simulator.name,
                    len(self.prefill_requests_in_mpa_time_window),
                    len(self.decode_requests_in_mpa_time_window),
                )

    def dump_request_trace_to_csv(self, filepath: str | None = None):
        """
        Dump self.request_trace to a CSV file.
        """
        if filepath is None:
            filepath = os.path.join(
                self.simulator.config.output_dir, "request_trace.csv"
            )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(
                [
                    "request_id",
                    "input_seqlen",
                    "output_seqlen",
                    "enqueue_timestamp",
                    "prefill_start_timestamp",
                    "prefill_end_timestamp",
                    "decode_start_timestamp",
                    "decode_end_timestamp",
                    "prefill_queuing_delay_ns",
                    "prefill_latency_ns",
                    "TTFT_ns",
                    "decode_queuing_delay_per_iteration_ns",
                    "TPOT_ns",
                    "ideal_TTFT_ns",
                    "ideal_TPOT_ns",
                    "prefill_energy_J",
                    "decode_energy_per_token_J",
                    "prefill_cost_dollars",
                    "decode_cost_per_token_dollars",
                    "config_prefill_npu_type",
                    "config_prefill_input_seqlen",
                    "config_prefill_batch_size",
                    "config_prefill_num_chips",
                    "config_prefill_pcfg",
                    "config_decode_npu_types",
                    "config_decode_input_seqlens",
                    "config_decode_output_seqlens",
                    "config_decode_batch_sizes",
                    "config_decode_num_chips",
                    "config_decode_pcfgs",
                ]
            )
            # Write data
            for request in self.request_trace:
                # compute decode energy per token
                num_decode_iterations = request.output_seqlen - 1
                total_energy = sum(request.decode_energy_per_token_J)
                decode_energy_per_token_J = total_energy / num_decode_iterations
                total_cost = sum(request.decode_cost_per_token_dollars)
                decode_cost_per_token_dollars = total_cost / num_decode_iterations

                writer.writerow(
                    [
                        request.id,
                        request.input_seqlen,
                        request.output_seqlen,
                        request.enqueue_timestamp,
                        request.prefill_start_timestamp,
                        request.prefill_end_timestamp,
                        request.decode_start_timestamp,
                        request.decode_end_timestamp,
                        request.prefill_queuing_delay_ns(),
                        request.prefill_latency_ns(),
                        request.TTFT_ns(),
                        request.decode_queuing_delay_per_iteration_ns(),
                        request.TPOT_ns(),
                        request.ideal_TTFT_ns,
                        request.ideal_TPOT_ns,
                        request.prefill_energy_J,
                        decode_energy_per_token_J,
                        request.prefill_cost_dollars,
                        decode_cost_per_token_dollars,
                        request.config_prefill_npu_type,
                        request.config_prefill_input_seqlen,
                        request.config_prefill_batch_size,
                        request.config_prefill_num_chips,
                        request.config_prefill_pcfg,
                        "/".join(request.config_decode_npu_types),
                        "/".join(map(str, request.config_decode_input_seqlens)),
                        "/".join(map(str, request.config_decode_output_seqlens)),
                        "/".join(map(str, request.config_decode_batch_sizes)),
                        "/".join(map(str, request.config_decode_num_chips)),
                        "/".join(request.config_decode_pcfgs),
                    ]
                )

        logging.info("Request trace dumped to %s", filepath)

    def get_request_stats_summary(self) -> dict:
        """
        Return a summary of the request trace statistics in a dictionary.
        """
        if not self.request_trace:
            raise ValueError("No completed requests are available for the summary.")

        stats = {}
        stats["total_requests"] = len(self.request_trace)

        # min, max, average of input sequence length
        input_seqlens = [request.input_seqlen for request in self.request_trace]
        stats["input_seqlen_min"] = min(input_seqlens)
        stats["input_seqlen_max"] = max(input_seqlens)
        stats["input_seqlen_avg"] = np.mean(input_seqlens)

        # min, max, average of output sequence length
        output_seqlens = [request.output_seqlen for request in self.request_trace]
        decode_token_counts = [output_seqlen - 1 for output_seqlen in output_seqlens]
        stats["output_seqlen_min"] = min(output_seqlens)
        stats["output_seqlen_max"] = max(output_seqlens)
        stats["output_seqlen_avg"] = np.mean(output_seqlens)

        # min, max, average of TTFT in seconds
        TTFTs = [request.TTFT_ns() for request in self.request_trace]
        stats["TTFT_min_s"] = np.min(TTFTs) / 1e9
        stats["TTFT_max_s"] = np.max(TTFTs) / 1e9
        stats["TTFT_avg_s"] = np.mean(TTFTs) / 1e9

        # min, max, average of TPOT in seconds
        TPOTs = [request.TPOT_ns() for request in self.request_trace]
        stats["TPOT_min_s"] = np.min(TPOTs) / 1e9
        stats["TPOT_max_s"] = np.max(TPOTs) / 1e9
        stats["TPOT_avg_s"] = np.mean(TPOTs) / 1e9

        # request rate in requests per second
        first_enqueue_time = min(req.enqueue_timestamp for req in self.request_trace)
        last_enqueue_time = max(req.enqueue_timestamp for req in self.request_trace)
        arrival_window_ns = last_enqueue_time - first_enqueue_time
        if arrival_window_ns > 0:
            stats["request_rate_rps"] = len(self.request_trace) / (
                arrival_window_ns / 1e9
            )
        else:
            stats["request_rate_rps"] = float("inf")

        # Use the workload completion horizon. Control-plane timers may remain
        # queued after the final request and must not dilute throughput.
        last_completion_time = max(
            req.decode_end_timestamp for req in self.request_trace
        )
        completion_window_ns = last_completion_time - first_enqueue_time
        if completion_window_ns <= 0:
            raise ValueError(
                "Completed requests must end after the first enqueue timestamp."
            )
        completion_window_s = completion_window_ns / 1e9
        stats["throughput_rps"] = len(self.request_trace) / completion_window_s
        # prefill throughput in tokens per second
        stats["prefill_throughput_tps"] = sum(input_seqlens) / completion_window_s
        # Decode iterations produce tokens 2..N; token 1 is emitted by prefill.
        stats["decode_throughput_tps"] = sum(decode_token_counts) / completion_window_s

        # cost efficiencies
        stats["prefill_token_per_joule"] = sum(input_seqlens) / sum(
            req.prefill_energy_J for req in self.request_trace
        )
        stats["decode_token_per_joule"] = sum(decode_token_counts) / sum(
            sum(req.decode_energy_per_token_J) for req in self.request_trace
        )
        stats["total_token_per_joule"] = (sum(input_seqlens) + sum(output_seqlens)) / (
            sum(req.prefill_energy_J for req in self.request_trace)
            + sum(sum(req.decode_energy_per_token_J) for req in self.request_trace)
        )
        stats["prefill_token_per_dollar"] = sum(input_seqlens) / sum(
            req.prefill_cost_dollars for req in self.request_trace
        )
        stats["decode_token_per_dollar"] = sum(decode_token_counts) / sum(
            sum(req.decode_cost_per_token_dollars) for req in self.request_trace
        )
        stats["total_token_per_dollar"] = (sum(input_seqlens) + sum(output_seqlens)) / (
            sum(req.prefill_cost_dollars for req in self.request_trace)
            + sum(sum(req.decode_cost_per_token_dollars) for req in self.request_trace)
        )

        return stats

    def print_and_dump_request_stats_summary(self):
        """
        Print a summary of the request trace statistics.
        """
        stats = self.get_request_stats_summary()
        logging.info("Request trace summary:")
        logging.info(json.dumps(stats, indent=4))

        # dump to JSON file
        output_filepath = os.path.join(self.simulator.config.output_dir, "stats.json")
        with open(output_filepath, "w") as f:
            json.dump(stats, f, indent=4)
        logging.info("Simulation stats summary dumped to %s", output_filepath)

    def dump_simulation_stats(self):
        self.dump_request_trace_to_csv()
        self.print_and_dump_request_stats_summary()

    def save_to_checkpoint(self, checkpoint_path: str):
        cpt = {
            "request_trace": self.request_trace,
            "workload_config": self.workload_config,
            "prefill_queue_lengths": self.prefill_queue_lengths,
            "decode_queue_lengths": self.decode_queue_lengths,
            "prefill_seqlens": self.prefill_seqlens,
            "decode_seqlens": self.decode_seqlens,
        }
        with gzip.open(checkpoint_path, "wb") as f:
            pickle.dump(cpt, f)

    def load_from_checkpoint(self, checkpoint_path: str):
        with gzip.open(checkpoint_path, "rb") as f:
            cpt = pickle.load(f)
            self.request_trace = cpt["request_trace"]
            self.workload_config = cpt["workload_config"]
            self.prefill_queue_lengths = cpt["prefill_queue_lengths"]
            self.decode_queue_lengths = cpt["decode_queue_lengths"]
            self.prefill_seqlens = cpt["prefill_seqlens"]
            self.decode_seqlens = cpt["decode_seqlens"]
