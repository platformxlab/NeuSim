import csv
import json
import os
import typing
from collections import deque

import numpy as np
from absl import logging

from neusim.fleetsim.LLMInferenceEvents import LLMInferenceDecodeIterationEndEvent
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.fleetsim.SimObject import SimObject

if typing.TYPE_CHECKING:
    from neusim.fleetsim.NPUFleetSimulator import NPUFleetSimulator


class MetricsServer(SimObject):
    """Collect completed-request latency, energy, and queue metrics."""

    def __init__(self, simulator: "NPUFleetSimulator", name: str | None = None):
        super().__init__(name or "metrics_server", simulator)
        self.request_trace: list[LLMRequest] = []
        """List of completed LLMRequest objects for trace analysis."""
        self.workload_config = simulator.config.workload_config
        self.dvfs_enabled = self.workload_config.enable_dvfs is True
        self.slo_targets: list[dict[str, float | int]] = []
        if self.dvfs_enabled:
            self.slo_targets = self._load_slo_targets()

        self.dvfs_slo_window_ns = int(
            getattr(self.workload_config, "dvfs_safeguard_window_minutes", 5.0)
            * 60
            * 1e9
        )
        self.dvfs_slo_records: deque[tuple[int, bool]] = deque()
        self.dvfs_slo_violation_count = 0
        self.dvfs_locked_to_peak = False

    def _load_slo_targets(self) -> list[dict[str, float | int]]:
        """Load and validate the paper's percentile-bucket SLO targets."""
        path = self.workload_config.slo_json_path
        multiplier = self.workload_config.slo_multiplier
        if not path:
            raise ValueError("slo_json_path is required when DVFS is enabled")
        with open(path) as slo_file:
            data = json.load(slo_file)
        results = data.get("results")
        if not isinstance(results, list) or not results:
            raise ValueError(f"SLO JSON has no nonempty results list: {path}")

        targets: list[dict[str, float | int]] = []
        for result in sorted(results, key=lambda item: item["percentile"]):
            try:
                targets.append(
                    {
                        "input_seqlen": int(result["input_seqlen"]),
                        "decode_repr_seqlen": int(
                            result["decode"]["representative_seqlen"]
                        ),
                        "ttft_target": float(
                            result["prefill"]["slo_TTFT_sec"][multiplier]
                        ),
                        "tpot_target": float(
                            result["decode"]["slo_TPOT_ms"][multiplier]
                        ),
                    }
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid SLO target for multiplier {multiplier!r} in {path}"
                ) from exc
        return targets

    def assign_ttft_target(self, input_seqlen: int) -> float:
        """Assign the first percentile-bucket TTFT target that covers a request."""
        for target in self.slo_targets:
            if input_seqlen <= target["input_seqlen"]:
                return float(target["ttft_target"])
        return float(self.slo_targets[-1]["ttft_target"])

    def assign_tpot_target(self, total_seqlen: int) -> float:
        """Assign the first percentile-bucket TPOT target that covers a request."""
        for target in self.slo_targets:
            if total_seqlen <= target["decode_repr_seqlen"]:
                return float(target["tpot_target"])
        return float(self.slo_targets[-1]["tpot_target"])

    def _check_request_slo(self, request: LLMRequest) -> bool:
        ttft_target = self.assign_ttft_target(request.input_seqlen)
        tpot_target = self.assign_tpot_target(request.total_seqlen)
        return (
            request.TTFT_ns() / 1e9 <= ttft_target
            and request.TPOT_ns() / 1e6 <= tpot_target
        )

    def update_dvfs_slo_window(self, timestamp: int, request: LLMRequest) -> None:
        """Update safeguard 2 in O(1) per completed request."""
        violated = not self._check_request_slo(request)
        self.dvfs_slo_records.append((timestamp, violated))
        self.dvfs_slo_violation_count += int(violated)
        cutoff = timestamp - self.dvfs_slo_window_ns
        while self.dvfs_slo_records and self.dvfs_slo_records[0][0] < cutoff:
            _, stale_violated = self.dvfs_slo_records.popleft()
            self.dvfs_slo_violation_count -= int(stale_violated)
        violation_rate = (
            self.dvfs_slo_violation_count / len(self.dvfs_slo_records)
            if self.dvfs_slo_records
            else 0.0
        )
        self.dvfs_locked_to_peak = (
            violation_rate > self.workload_config.dvfs_safeguard_violation_threshold
        )

    def initialize(self):
        """Register only the completion listener needed by the static artifact."""
        self.simulator.add_event_listener(
            LLMInferenceDecodeIterationEndEvent.get_type_listener(
                self.decode_iteration_end
            )
        )

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
            if self.dvfs_enabled:
                for request in finished_requests:
                    self.update_dvfs_slo_window(event.timestamp, request)
            if (
                len(self.request_trace) % 10_000 == 0
                or event.timestamp % int(1e9 * 3600) == 0
            ):  # progress only; avoid I/O on the full 2.49M-request trace
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

    def dump_request_trace_to_csv(self, filepath: str | None = None):
        """Dump the compact request timing trace consumed by Figure 5."""
        if filepath is None:
            filepath = os.path.join(
                self.simulator.config.output_dir, "request_trace.csv"
            )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
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
                    "effective_TPOT_ns",
                    "prefill_energy_J",
                    "decode_energy_J",
                    "decode_energy_per_token_J",
                    "prefill_cost_dollars",
                    "decode_cost_dollars",
                ]
            )
            for request in self.request_trace:
                writer.writerow(
                    [
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
                        request.TPOT_ns(),
                        request.prefill_energy_J,
                        request.decode_energy_J,
                        request.decode_energy_J / request.output_seqlen,
                        request.prefill_cost_dollars,
                        request.decode_cost_dollars,
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

        stats["effective_TPOT_min_s"] = stats["TPOT_min_s"]
        stats["effective_TPOT_max_s"] = stats["TPOT_max_s"]
        stats["effective_TPOT_avg_s"] = stats["TPOT_avg_s"]
        if self.dvfs_enabled:
            from neusim.fleetsim.dvfs_scheduler import get_dvfs_lookup_stats

            stats["dvfs_lookup"] = get_dvfs_lookup_stats()

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
            req.decode_energy_J for req in self.request_trace
        )
        stats["total_token_per_joule"] = (sum(input_seqlens) + sum(output_seqlens)) / (
            sum(req.prefill_energy_J for req in self.request_trace)
            + sum(req.decode_energy_J for req in self.request_trace)
        )
        stats["prefill_token_per_dollar"] = sum(input_seqlens) / sum(
            req.prefill_cost_dollars for req in self.request_trace
        )
        stats["decode_token_per_dollar"] = sum(decode_token_counts) / sum(
            req.decode_cost_dollars for req in self.request_trace
        )
        stats["total_token_per_dollar"] = (sum(input_seqlens) + sum(output_seqlens)) / (
            sum(req.prefill_cost_dollars for req in self.request_trace)
            + sum(req.decode_cost_dollars for req in self.request_trace)
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
