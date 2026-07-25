import gzip
import math
import pickle
import typing
import uuid
from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
from absl import logging

from neusim.configs.models.LLMConfig import LLMConfig
from neusim.configs.models.ModelConfig import ModelConfig
from neusim.eventsim.Event import Event
from neusim.fleetsim.LLMInferenceEvents import (
    LLMInferencePrefillEndEvent,
    LLMInferenceRequestEnqueueEvent,
    vPodHorizontalScalingRecommendationEvent,
    vPodMultiDimensionalScalingRecommendationEvent,
    vPodVerticalScalingRecommendationEvent,
)
from neusim.fleetsim.util import get_pstr

# from neusim.fleetsim.util import ListMap
if typing.TYPE_CHECKING:
    from neusim.fleetsim.LLMInferenceEndpoint import (
        LLMInferenceInstanceBase,
        vPodInstance,
    )
    from neusim.fleetsim.NPUFleetSimulator import NPUFleetSimulator
import neusim.fleetsim.vPodAutoScaler_lib as autoscaler_lib
import neusim.npusim.frontend.memory_footprint_analysis_lib as mem_footprint_lib
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.fleetsim.MetricsServer import MetricsServer
from neusim.fleetsim.SimObject import SimObject


class vPodAutoScaler(SimObject):
    """
    vPod auto scaler.
    This class implements the logic for horizontal and vertical scaling recommendations of vPods.
    ***Only implements recommendation. Does not actually fire events to scale vPods.***
    The actual scaling operation should be managed by another entity such as the LLM inference endpoint.
    """

    def __init__(
        self,
        simulator: "NPUFleetSimulator",
        name: str | None = None,
        metrics_server: MetricsServer | None = None,
        prefill_or_decode: str = "prefill",
        enable_mpa: bool = False,
        enable_autoscaling_events: bool = True,
    ):
        super().__init__(name or "autoscaler", simulator)
        self.id: str = uuid.uuid4().hex
        self.hs_interval_ns: int = int(
            simulator.config.workload_config.hs_interval_minutes * 60 * 1e9
        )
        self.vs_interval_ns: int = int(
            simulator.config.workload_config.vs_interval_minutes * 60 * 1e9
        )
        self.hs_window_ns: int = int(
            simulator.config.workload_config.hs_window_minutes * 60 * 1e9
        )
        self.vs_window_ns: int = int(
            simulator.config.workload_config.vs_window_minutes * 60 * 1e9
        )
        self.mpa_interval_ns: int = min(self.hs_interval_ns, self.vs_interval_ns)
        """Multi-dimensional auto-scaling time interval. Useful only if enable_mpa is True."""
        self.mpa_window_ns: int = max(self.hs_window_ns, self.vs_window_ns)
        """Multi-dimensional auto-scaling time window. Useful only if enable_mpa is True."""
        self.enable_mpa: bool = enable_mpa
        """If True, use multi-dimensional auto-scaling. Otherwise, use separate horizontal and vertical scaling."""
        self.metrics_server: MetricsServer = metrics_server or MetricsServer(
            simulator, f"{self.name}_metrics_server"
        )
        self.prefill_or_decode: str = prefill_or_decode
        self._ewma_state: dict[str, float] = {}
        self.queue_depth_driven_configs: set[tuple[LLMConfig, ...]] = set()
        self.last_recommendation_queue_driven: bool = False
        self._mpa_min_interval_ns: int = int(
            simulator.config.workload_config.mpa_min_interval_seconds * 1e9
        )
        self._last_nonroutine_mpa_timestamp: int = -1

        if not enable_autoscaling_events:
            return

        # add recursive event listener
        if not self.enable_mpa:
            self.simulator.add_event_listener(
                vPodHorizontalScalingRecommendationEvent.get_autoscaler_id_listener(
                    self.schedule_next_hs_event, self.id
                )
            )
            self.simulator.add_event_listener(
                vPodVerticalScalingRecommendationEvent.get_autoscaler_id_listener(
                    self.schedule_next_vs_event, self.id
                )
            )

            # trigger HS when a new request is enqueued
            def _schedule_HS_event(event: Event):
                self.schedule_hs_event(
                    event.timestamp, do_not_scale_down=True, is_routine=False
                )

            if prefill_or_decode == "prefill":
                self.simulator.add_event_listener(
                    LLMInferenceRequestEnqueueEvent.get_type_listener(
                        _schedule_HS_event
                    )
                )
            else:
                self.simulator.add_event_listener(
                    LLMInferencePrefillEndEvent.get_type_listener(_schedule_HS_event)
                )

            # schedule the first hs and vs event
            self.simulator.put(
                vPodHorizontalScalingRecommendationEvent(
                    self.simulator.timestamp + self.hs_interval_ns,
                    self.prefill_or_decode,
                    self.id,
                )
            )
            self.simulator.put(
                vPodVerticalScalingRecommendationEvent(
                    self.simulator.timestamp + self.vs_interval_ns,
                    self.prefill_or_decode,
                    self.id,
                )
            )
        else:  # use MPA scaling event
            self.simulator.add_event_listener(
                vPodMultiDimensionalScalingRecommendationEvent.get_autoscaler_id_listener(
                    self.schedule_next_mpa_event, self.id
                )
            )

            # trigger MPA when a new request is enqueued
            def _schedule_MPA_event(event: Event):
                self.schedule_mpa_event(
                    event.timestamp, do_not_scale_down=True, is_routine=False
                )

            if prefill_or_decode == "prefill":
                self.simulator.add_event_listener(
                    LLMInferenceRequestEnqueueEvent.get_type_listener(
                        _schedule_MPA_event
                    )
                )
            else:
                self.simulator.add_event_listener(
                    LLMInferencePrefillEndEvent.get_type_listener(_schedule_MPA_event)
                )

            # schedule the first mpa event
            self.simulator.put(
                vPodMultiDimensionalScalingRecommendationEvent(
                    self.simulator.timestamp + self.mpa_interval_ns,
                    self.prefill_or_decode,
                    self.id,
                )
            )

    def schedule_next_mpa_event(
        self, event: vPodMultiDimensionalScalingRecommendationEvent
    ):
        """
        Schedule the next multi-dimensional scaling event.
        """
        assert self.simulator.llm_inference_endpoint
        assert self.simulator.client
        assert self.simulator.metrics_server
        if not event.is_routine:
            return
        total_num_requests_from_trace = (
            self.simulator.client.total_num_enqueued_requests
        )
        processed_num_requests = len(self.simulator.metrics_server.request_trace)
        if logging.level_debug():
            logging.debug(
                f"Processed {processed_num_requests} out of {total_num_requests_from_trace} requests."
            )
        if processed_num_requests >= total_num_requests_from_trace:
            # No requests in the queue, do not schedule the next event
            logging.debug(
                f"Not scheduling next vertical scaling event for autoscaler {self.id} as all requests have been processed."
            )
            return
        self.schedule_mpa_event(event.timestamp + self.mpa_interval_ns)

    def schedule_mpa_event(
        self, timestamp: int, do_not_scale_down: bool = False, is_routine: bool = True
    ):
        """
        Schedule the next multi-dimensional scaling event at timestamp.
        """
        # Throttle non-routine (request-triggered) MPA events to at most once per
        # mpa_min_interval_seconds. During bursts many requests enqueue in quick
        # succession; processing MPA for each one is wasteful since the request
        # window barely changes between consecutive enqueues.
        if not is_routine:
            if (
                timestamp - self._last_nonroutine_mpa_timestamp
                < self._mpa_min_interval_ns
            ):
                return
            self._last_nonroutine_mpa_timestamp = timestamp
        self.simulator.put(
            vPodMultiDimensionalScalingRecommendationEvent(
                timestamp,
                self.prefill_or_decode,
                self.id,
                do_not_scale_down,
                is_routine,
            )
        )
        if logging.level_debug():
            logging.debug(
                "Scheduled next multi-dimensional scaling recommendation event for %s at %d",
                self.prefill_or_decode,
                timestamp,
            )

    def schedule_hs_event(
        self, timestamp: int, do_not_scale_down: bool = False, is_routine: bool = True
    ):
        """
        Schedule the next horizontal scaling event at timestamp.
        """
        self.simulator.put(
            vPodHorizontalScalingRecommendationEvent(
                timestamp,
                self.prefill_or_decode,
                self.id,
                do_not_scale_down,
                is_routine,
            )
        )
        if logging.level_debug():
            logging.debug(
                "Scheduled next horizontal scaling recommendation event for %s at %d",
                self.prefill_or_decode,
                timestamp,
            )

    def schedule_vs_event(self, timestamp: int):
        """
        Schedule the next vertical scaling event at timestamp.
        """
        self.simulator.put(
            vPodVerticalScalingRecommendationEvent(
                timestamp, self.prefill_or_decode, self.id
            )
        )
        if logging.level_debug():
            logging.debug(
                "Scheduled next vertical scaling recommendation event for %s at %d",
                self.prefill_or_decode,
                timestamp,
            )

    def schedule_next_hs_event(self, event: vPodHorizontalScalingRecommendationEvent):
        assert self.simulator.llm_inference_endpoint
        assert self.simulator.client
        assert self.simulator.metrics_server
        if not event.is_routine:
            return
        total_num_requests_from_trace = (
            self.simulator.client.total_num_enqueued_requests
        )
        processed_num_requests = len(self.simulator.metrics_server.request_trace)
        if logging.level_debug():
            logging.debug(
                f"Processed {processed_num_requests} out of {total_num_requests_from_trace} requests."
            )
        if processed_num_requests >= total_num_requests_from_trace:
            # No requests in the queue, do not schedule the next event
            logging.debug(
                f"Not scheduling next horizontal scaling event for autoscaler {self.id} as all requests have been processed."
            )
            return
        self.schedule_hs_event(event.timestamp + self.hs_interval_ns)

    def schedule_next_vs_event(self, event: vPodVerticalScalingRecommendationEvent):
        assert self.simulator.llm_inference_endpoint
        assert self.simulator.client
        assert self.simulator.metrics_server
        total_num_requests_from_trace = (
            self.simulator.client.total_num_enqueued_requests
        )
        processed_num_requests = len(self.simulator.metrics_server.request_trace)
        if logging.level_debug():
            logging.debug(
                f"Processed {processed_num_requests} out of {total_num_requests_from_trace} requests."
            )
        if processed_num_requests >= total_num_requests_from_trace:
            # No requests in the queue, do not schedule the next event
            logging.debug(
                f"Not scheduling next vertical scaling event for autoscaler {self.id} as all requests have been processed."
            )
            return
        self.schedule_vs_event(event.timestamp + self.vs_interval_ns)

    def save_to_checkpoint(self, checkpoint_path: str):
        simulator = self.simulator
        metrics_server = self.metrics_server
        self.simulator = None  # type: ignore
        self.metrics_server = None  # type: ignore
        with gzip.open(checkpoint_path, "wb") as f:
            pickle.dump(self, f)
        self.simulator = simulator
        self.metrics_server = metrics_server

    def load_from_checkpoint(self, checkpoint_path: str):
        simulator = self.simulator
        metrics_server = self.metrics_server
        with gzip.open(checkpoint_path, "rb") as f:
            obj = pickle.load(f)
            self.__dict__.update(obj.__dict__)
            self.simulator = simulator
            self.metrics_server = metrics_server

    @abstractmethod
    def get_initial_allocation(self) -> list[tuple[int, ModelConfig]]:
        raise NotImplementedError(
            "vPodAutoScaler.get_initial_allocation is not implemented yet."
        )

    def get_initial_allocation_alternatives(
        self,
    ) -> list[tuple[int, list[ModelConfig]]]:
        """
        Like get_initial_allocation, but each group exposes an ORDERED list of candidate
        configs (most-preferred / most-efficient first). The endpoint allocates each vPod
        on the first candidate whose NPU version still has capacity under
        max_chips_per_version, so the initial fleet spills onto a less-preferred version
        instead of silently over-subscribing the best one.

        Default: wrap every single config from get_initial_allocation() as a singleton
        list, i.e. no spill (preserves legacy behavior for autoscalers that do not provide
        version alternatives).
        """
        return [(count, [cfg]) for count, cfg in self.get_initial_allocation()]

    @abstractmethod
    def get_recommended_allocation(
        self, *args, **kwargs
    ) -> list[tuple[Sequence[ModelConfig], int]]:
        """
        Get the recommended allocation of vPods based on the current workload.
        Returns a list of tuples, where each tuple contains a list of vPod configurations and the number of vPods with that configuration.
        This function should be overridden and used when multi-dimensional auto-scaling is enabled.
        Otherwise, it defaults to returning a list of one tuple containing the recommended number of vPods and the recommended vPod configuration
        (based on HS and VS separately).
        """
        return [
            (
                self.get_recommended_vPod_config(*args, **kwargs),
                self.get_recommended_num_vPods(*args, **kwargs),
            )
        ]

    @abstractmethod
    def get_recommended_num_vPods(self, *args, **kwargs) -> int:
        """
        Horizontal scaling.
        Get the recommended number of vPods based on the current workload.
        This method should be called periodically to adjust the number of vPods.
        """
        raise NotImplementedError(
            "vPodAutoScaler.get_recommended_num_vPods is not implemented yet."
        )

    @abstractmethod
    def get_recommended_vPod_config(self, *args, **kwargs) -> list[ModelConfig]:
        """
        Vertical scaling.
        Get the recommended configuration for a vPod based on the current workload.
        This method should be called periodically to adjust the vPod configuration.
        """
        raise NotImplementedError(
            "vPodAutoScaler.get_recommended_vPod_config is not implemented yet."
        )

    def get_seqlens_for_config(
        self, seqlen_pairs: Sequence[tuple[int, int]], config: LLMConfig
    ) -> set[tuple[int, int]]:
        """Get seqlen pairs that should be routed to the given config.
        Default: delegates to autoscaler_lib.get_seqlens_for_config()."""
        return autoscaler_lib.get_seqlens_for_config(
            seqlen_pairs, config, self.simulator.config, self.prefill_or_decode
        )

    def _compute_ewma_peak_rate(
        self,
        requests: Sequence[LLMRequest],
        timestamp: int,
        window_ns: int,
        ewma_key: str,
    ) -> float:
        """Compute EWMA-smoothed peak request arrival rate.

        Bins requests by enqueue_timestamp into intervals of
        ewma_interval_seconds within [timestamp - window_ns, timestamp].
        Peak rate = max(bin_count) / interval_duration.
        Updates self._ewma_state[ewma_key] with smoothed value.
        """
        interval_s = self.simulator.config.workload_config.ewma_interval_seconds
        alpha = self.simulator.config.workload_config.ewma_alpha
        interval_ns = int(interval_s * 1e9)

        window_start = timestamp - window_ns
        window_end = timestamp

        # Count requests per interval bin
        bin_counts: dict[int, int] = {}
        for req in requests:
            t = req.enqueue_timestamp
            if t < window_start or t > window_end:
                continue
            bin_idx = int((t - window_start) // interval_ns)
            bin_counts[bin_idx] = bin_counts.get(bin_idx, 0) + 1

        if len(bin_counts) == 0:
            peak_rate = 0.0
        else:
            peak_count = max(bin_counts.values())
            peak_rate = peak_count / interval_s

        # EWMA update
        if ewma_key not in self._ewma_state:
            self._ewma_state[ewma_key] = peak_rate  # first observation, no smoothing
        else:
            self._ewma_state[ewma_key] = (
                alpha * peak_rate + (1 - alpha) * self._ewma_state[ewma_key]
            )

        if logging.level_debug():
            logging.debug(
                "(%s) EWMA key=%s: peak_rate=%.2f, ewma=%.2f, #requests=%d, #bins=%d",
                self.prefill_or_decode,
                ewma_key,
                peak_rate,
                self._ewma_state[ewma_key],
                len(list(requests)),
                len(bin_counts),
            )

        return self._ewma_state[ewma_key]

    def _compute_num_vpods_from_rate(
        self,
        ewma_peak_rate: float,
        batch_size: float,
        ideal_latency: float,
        prefill_or_decode: str,
        avg_output_seqlen: float = 1.0,
    ) -> int:
        """Compute num_vPods from EWMA peak rate and per-vPod throughput.

        For prefill: throughput = batch_size / ideal_latency  (ideal_latency in seconds)
        For decode:  throughput = batch_size / (avg_output_seqlen * ideal_latency / 1000)
                     (ideal_latency in ms -> seconds)
        Returns max(1, ceil(headroom * ewma_peak_rate / throughput)).
        """
        headroom = self.simulator.config.workload_config.scaling_headroom_factor

        if prefill_or_decode == "prefill":
            throughput = (
                batch_size / ideal_latency if ideal_latency > 0 else float("inf")
            )
        else:
            latency_s = ideal_latency / 1000.0  # ms -> s
            throughput = (
                batch_size / (avg_output_seqlen * latency_s)
                if latency_s > 0 and avg_output_seqlen > 0
                else float("inf")
            )

        if throughput <= 0 or throughput == float("inf"):
            return 1

        num_vpods = math.ceil(headroom * ewma_peak_rate / throughput)

        if logging.level_debug():
            logging.debug(
                "(%s) EWMA vpods: rate=%.2f, throughput=%.2f, headroom=%.2f, batch=%.2f, latency=%.4f, avg_oseq=%.1f -> %d vpods",
                prefill_or_decode,
                ewma_peak_rate,
                throughput,
                headroom,
                batch_size,
                ideal_latency,
                avg_output_seqlen,
                max(1, num_vpods),
            )

        return max(1, num_vpods)

    def _compute_num_vpods_from_queue_depth(
        self,
        queue_depth: int,
        batch_size: float,
        ideal_latency: float,
        prefill_or_decode: str,
        avg_output_seqlen: float = 1.0,
    ) -> int:
        """Compute num_vPods needed to drain queued requests within the target time.

        Uses the same throughput model as _compute_num_vpods_from_rate but
        derives the vPod count from queue_depth / (per_vpod_capacity * drain_target)
        instead of from arrival rate.

        Returns 0 when queue_depth == 0 or drain_target_seconds == 0 (disabled).
        """
        drain_target = self.simulator.config.workload_config.queue_drain_target_seconds
        if drain_target <= 0 or queue_depth <= 0:
            return 0

        if prefill_or_decode == "prefill":
            throughput = (
                batch_size / ideal_latency if ideal_latency > 0 else float("inf")
            )
        else:
            latency_s = ideal_latency / 1000.0  # ms -> s
            throughput = (
                batch_size / (avg_output_seqlen * latency_s)
                if latency_s > 0 and avg_output_seqlen > 0
                else float("inf")
            )

        if throughput <= 0 or throughput == float("inf"):
            return 0

        capacity_per_vpod = throughput * drain_target
        drain_vpods = math.ceil(queue_depth / capacity_per_vpod)

        if logging.level_debug():
            logging.debug(
                "(%s) Queue drain: depth=%d, throughput=%.2f, drain_target=%.1fs, capacity/vpod=%.1f -> %d vpods",
                prefill_or_decode,
                queue_depth,
                throughput,
                drain_target,
                capacity_per_vpod,
                drain_vpods,
            )

        return drain_vpods

    @abstractmethod
    def get_vPod_create_delay_ns(self, vPod: "vPodInstance") -> int:
        """
        Get the delay for creating a new vPod in nanoseconds.
        """
        raise NotImplementedError(
            "vPodAutoScaler.get_vPod_create_delay_ns is not implemented yet."
        )

    @abstractmethod
    def get_vPod_reconfig_delay_ns(
        self, old_config: ModelConfig, new_config: ModelConfig
    ) -> int:
        """
        Get the delay for reconfiguring a vPod in nanoseconds.
        """
        raise NotImplementedError(
            "vPodAutoScaler.get_vPod_reconfig_delay_ns is not implemented yet."
        )


class HorizontalAutoScaler(vPodAutoScaler):
    """
    Horizontal auto scaler.
    This class implements the logic for horizontal scaling of vPods.
    If a vPod config is given, it will be used to create the vPod replicas.
    Otherwise, the initial allocation result will be used to create the vPod replicas.
    """

    def __init__(
        self,
        simulator: "NPUFleetSimulator",
        name: str | None = None,
        metrics_server: MetricsServer | None = None,
        default_vPod_config: ModelConfig | None = None,
        prefill_or_decode: str = "prefill",
    ):
        super().__init__(simulator, name, metrics_server, prefill_or_decode)

        # find average seqlen of all requests
        assert self.simulator.client
        requests = self.simulator.client.requests

        if simulator.config.workload_config.hs_initial_alloc_sample_criteria == "max":
            avg_input_seqlen, avg_output_seqlen = max(
                [(req.input_seqlen, req.output_seqlen) for req in requests],
                key=lambda x: x[0] if prefill_or_decode == "prefill" else (x[0] + x[1]),
            )
        elif (
            simulator.config.workload_config.hs_initial_alloc_sample_criteria
            == "average"
        ):
            # avg_input_seqlen, avg_output_seqlen = (
            #     math.ceil(sum(req.input_seqlen for req in requests) / len(requests)),
            #     math.ceil(sum(req.output_seqlen for req in requests) / len(requests)),
            # )
            # use mean + 0.25 std to avoid under-provisioning
            input_seqlen_std = np.std([req.input_seqlen for req in requests])
            output_seqlen_std = np.std([req.output_seqlen for req in requests])
            avg_input_seqlen = math.ceil(
                np.mean([req.input_seqlen for req in requests])
                + 0.25 * input_seqlen_std
            )
            avg_output_seqlen = math.ceil(
                np.mean([req.output_seqlen for req in requests])
                + 0.25 * output_seqlen_std
            )
        else:
            raise ValueError(
                f"Invalid sample criteria: {simulator.config.workload_config.hs_initial_alloc_sample_criteria}"
            )

        input_seqlen = autoscaler_lib.pad_seqlen(
            avg_input_seqlen,
            self.simulator.config.workload_config.input_seqlen_padding_factors,
            self.simulator.config.workload_config.input_seqlen_padding_steps,
        )
        output_seqlen = autoscaler_lib.pad_seqlen(
            avg_output_seqlen
            if prefill_or_decode == "decode"
            else 4,  # do not care about output seqlen for prefill vPods
            self.simulator.config.workload_config.output_seqlen_padding_factors,
            self.simulator.config.workload_config.output_seqlen_padding_steps,
        )

        temp_cfg = deepcopy(self.simulator.config)
        temp_cfg.workload_config.llm_config.input_seqlen = input_seqlen
        temp_cfg.workload_config.llm_config.output_seqlen = output_seqlen

        # npu_types = ["v5p" if prefill_or_decode == "decode" else "v6e"]  # hardcode this for now.
        optimal_cfg_for_avg_seqlen = (
            autoscaler_lib.get_optimal_vPod_config_with_seqlen_fallback(
                temp_cfg,
                prefill_or_decode,  # npu_types
            )
        )
        logging.info(
            "(%s) (%s) Initial vPod Config: type=%s, num_chips=%s, microbatch_size_ici=%s, input_seqlen=%s, output_seqlen=%s",
            self.simulator.name,
            self.prefill_or_decode,
            optimal_cfg_for_avg_seqlen[0].name,
            optimal_cfg_for_avg_seqlen[0].num_chips,
            optimal_cfg_for_avg_seqlen[0].microbatch_size_ici,
            optimal_cfg_for_avg_seqlen[0].input_seqlen,
            optimal_cfg_for_avg_seqlen[0].output_seqlen,
        )

        self.vPod_config = optimal_cfg_for_avg_seqlen[0]

        # Right-size the INITIAL vPod count from the warm-up window instead of a hardcoded
        # constant (was 32). Base-Max is a single-version baseline: it never spills to other
        # NPU versions (it deliberately does NOT override get_initial_allocation_alternatives,
        # so it inherits the base singleton default). That means prefill and decode must SHARE
        # the capped best-fit version (e.g. 6e=32 chips); a static 32 prefill vPods would grab
        # the whole cap and starve decode. Sizing each phase to its warm-up demand lets them
        # share the cap, and once the cap is hit the endpoint can no longer place vPods
        # (queueing -> SLO violation) -- which is exactly what the single-version baseline is
        # meant to exhibit under a constrained cluster.
        warmup_requests = [
            r for r in requests if r.enqueue_timestamp <= self.hs_window_ns
        ]
        ewma_peak_rate = self._compute_ewma_peak_rate(
            warmup_requests, self.hs_window_ns, self.hs_window_ns, "hs_init"
        )
        ideal_latency = autoscaler_lib.get_ideal_latency_for_config(
            self.simulator.config, self.vPod_config, prefill_or_decode
        )
        config_batch_size = self.vPod_config.microbatch_size_ici
        if prefill_or_decode == "prefill":
            config_seqlen = self.vPod_config.input_seqlen
            avg_req_seqlen = (
                float(np.mean([r.input_seqlen for r in warmup_requests]))
                if warmup_requests
                else float(config_seqlen)
            )
            avg_output_seqlen = 1.0
        else:
            config_seqlen = (
                self.vPod_config.input_seqlen + self.vPod_config.output_seqlen
            )
            avg_req_seqlen = (
                float(np.mean([r.total_seqlen for r in warmup_requests]))
                if warmup_requests
                else float(config_seqlen)
            )
            avg_output_seqlen = (
                float(max(1.0, np.mean([r.output_seqlen for r in warmup_requests])))
                if warmup_requests
                else 1.0
            )
        token_budget = config_batch_size * config_seqlen
        batch_size = token_budget / max(1.0, avg_req_seqlen)
        self.initial_num_vpods = self._compute_num_vpods_from_rate(
            ewma_peak_rate,
            batch_size,
            ideal_latency,
            prefill_or_decode,
            avg_output_seqlen,
        )
        logging.info(
            "(%s) (%s) Initial vPod count right-sized from warm-up: %d vPods (warm-up peak rate=%.2f req/s, %d warm-up reqs)",
            self.simulator.name,
            prefill_or_decode,
            self.initial_num_vpods,
            ewma_peak_rate,
            len(warmup_requests),
        )

    def get_initial_allocation(self) -> list[tuple[int, ModelConfig]]:
        # Single best-fit config (e.g. 6e); count right-sized from the warm-up window in
        # __init__. Base-Max intentionally does NOT override get_initial_allocation_-
        # alternatives, so it inherits the base singleton default and never spills.
        return [(self.initial_num_vpods, deepcopy(self.vPod_config))]

    def get_recommended_num_vPods(self, timestamp: int = -1, *args, **kwargs) -> int:
        """
        @timestamp: The current timestamp.
        Scale based on EWMA-smoothed peak request arrival rate vs per-vPod throughput.
        """
        assert timestamp != -1

        # Get requests from the MPA time window deque, filtered to HS window
        if self.prefill_or_decode == "prefill":
            all_requests = self.metrics_server.prefill_requests_in_mpa_time_window
        else:
            all_requests = self.metrics_server.decode_requests_in_mpa_time_window
        window_requests = [
            r
            for r in all_requests
            if r.enqueue_timestamp >= timestamp - self.hs_window_ns
        ]

        # Compute EWMA peak rate
        ewma_peak_rate = self._compute_ewma_peak_rate(
            window_requests, timestamp, self.hs_window_ns, "hs_global"
        )

        # Look up ideal latency for the vPod config
        ideal_latency = autoscaler_lib.get_ideal_latency_for_config(
            self.simulator.config, self.vPod_config, self.prefill_or_decode
        )

        # Token-budget effective batch size: how many average-sized requests fit in the token budget
        config_batch_size = self.vPod_config.microbatch_size_ici
        if self.prefill_or_decode == "prefill":
            config_seqlen = self.vPod_config.input_seqlen
            avg_req_seqlen = (
                float(np.mean([r.input_seqlen for r in window_requests]))
                if len(window_requests) > 0
                else float(config_seqlen)
            )
        else:
            config_seqlen = (
                self.vPod_config.input_seqlen + self.vPod_config.output_seqlen
            )
            avg_req_seqlen = (
                float(np.mean([r.total_seqlen for r in window_requests]))
                if len(window_requests) > 0
                else float(config_seqlen)
            )

        token_budget = config_batch_size * config_seqlen
        batch_size = token_budget / max(1.0, avg_req_seqlen)

        # For decode, compute avg output seqlen from window requests
        avg_output_seqlen = 1.0
        if self.prefill_or_decode == "decode" and len(window_requests) > 0:
            avg_output_seqlen = float(
                max(1.0, np.mean([r.output_seqlen for r in window_requests]))
            )

        num_vPods = self._compute_num_vpods_from_rate(
            ewma_peak_rate,
            batch_size,
            ideal_latency,
            self.prefill_or_decode,
            avg_output_seqlen,
        )

        # Queue-drain signal from MetricsServer
        queue_lengths = (
            self.metrics_server.prefill_queue_lengths
            if self.prefill_or_decode == "prefill"
            else self.metrics_server.decode_queue_lengths
        )
        current_queue_depth = queue_lengths[-1][1] if queue_lengths else 0

        num_vPods_drain = self._compute_num_vpods_from_queue_depth(
            current_queue_depth,
            batch_size,
            ideal_latency,
            self.prefill_or_decode,
            avg_output_seqlen,
        )
        if num_vPods_drain > num_vPods:
            self.last_recommendation_queue_driven = True
            num_vPods = num_vPods_drain
        else:
            self.last_recommendation_queue_driven = False

        if logging.level_debug():
            logging.debug(
                "(%s) (timestamp=%d) EWMA HS: peak_rate=%.2f, ideal_latency=%.4f, batch=%.2f (config=%d, config_seqlen=%d, avg_req_seqlen=%.1f), avg_oseq=%.1f, queue_depth=%d -> %d vPods",
                self.prefill_or_decode,
                timestamp,
                ewma_peak_rate,
                ideal_latency,
                batch_size,
                config_batch_size,
                config_seqlen,
                avg_req_seqlen,
                avg_output_seqlen,
                current_queue_depth,
                num_vPods,
            )
        return num_vPods

    def get_recommended_vPod_config(self, *args, **kwargs) -> list[ModelConfig]:
        """Horizontal scaling does not change the vPod config. Just return the initial config."""
        return [deepcopy(self.vPod_config)]

    def get_vPod_create_delay_ns(self, vPod: "LLMInferenceInstanceBase") -> int:  # type: ignore
        """
        Get the delay for creating a new vPod in nanoseconds.
        """
        VM_startup_delay = int(vPod.config.instance_startup_delay_sec * 1e9)
        model_weight_bytes_per_chip = (
            mem_footprint_lib.get_llm_inference_weight_mem_requirement(
                vPod.config.llm_config,
                1 if "deepseek" in vPod.config.model_name else 2,
            )
        )
        model_weight_load_bw_per_chip_GBps = vPod.config.llm_config.dcn_bw_GBps
        model_weight_load_delay = int(
            model_weight_bytes_per_chip
            / (model_weight_load_bw_per_chip_GBps * 1024 * 1024 * 1024)
        ) * int(1e9)
        return VM_startup_delay + model_weight_load_delay

    def get_vPod_reconfig_delay_ns(
        self, old_config: ModelConfig, new_config: ModelConfig
    ) -> int:
        # This function may be called for only once during initial VS event.
        # But the actual reconfig event will not be triggered for HS autoscaler.
        return 0

    def schedule_next_vs_event(self, event: vPodVerticalScalingRecommendationEvent):
        return  # do not perform vertical scaling for this class


class IdealAutoScaler(vPodAutoScaler):
    """
    Ideal auto scaler.
    This class implements the logic for both horizontal and vertical scaling of vPods.
    It always picks the ideal allocation based on the workload.
    """

    def __init__(
        self,
        simulator: "NPUFleetSimulator",
        name: str | None = None,
        metrics_server: MetricsServer | None = None,
        prefill_or_decode: str = "prefill",
    ):
        super().__init__(simulator, name, metrics_server, prefill_or_decode)

        # find average seqlen of all requests
        assert self.simulator.client
        requests = self.simulator.client.requests
        avg_input_seqlen = np.mean([req.input_seqlen for req in requests])
        avg_output_seqlen = np.mean([req.output_seqlen for req in requests])
        self.simulator.config.workload_config.llm_config.input_seqlen = (
            autoscaler_lib.pad_seqlen(
                math.ceil(avg_input_seqlen),
                self.simulator.config.workload_config.input_seqlen_padding_factors,
                self.simulator.config.workload_config.input_seqlen_padding_steps,
            )
        )
        self.simulator.config.workload_config.llm_config.output_seqlen = (
            autoscaler_lib.pad_seqlen(
                math.ceil(avg_output_seqlen),
                self.simulator.config.workload_config.output_seqlen_padding_factors,
                self.simulator.config.workload_config.output_seqlen_padding_steps,
            )
        )

        optimal_cfg_for_avg_seqlen = (
            autoscaler_lib.get_optimal_vPod_config_with_seqlen_fallback(
                self.simulator.config, prefill_or_decode
            )
        )

        self.vPod_config = optimal_cfg_for_avg_seqlen[0]

    def get_initial_allocation(self) -> list[tuple[int, ModelConfig]]:
        return [(32, self.vPod_config)]

    def get_recommended_num_vPods(self, timestamp: int = -1, *args, **kwargs) -> int:
        """
        @timestamp: The current timestamp.
        Scale based on the max request queue length in the last HS interval.
        Try to maintain (very aggressive scaling):
            max request queue length == 0.1 * (max_batch_size * num_vPods)
        """
        assert timestamp != -1
        # compute average request queue length in the last HS interval
        if self.prefill_or_decode == "prefill":
            request_queue_lengths = self.metrics_server.prefill_queue_lengths
        else:
            request_queue_lengths = self.metrics_server.decode_queue_lengths
        interval_request_queue_lengths = [
            x[1] for x in request_queue_lengths if x[0] >= timestamp - self.hs_window_ns
        ]
        max_batch_size = self.vPod_config.microbatch_size_ici

        scale_metric = (
            max(interval_request_queue_lengths)
            if len(interval_request_queue_lengths) > 0
            else 0
        )
        num_vPods = math.ceil(scale_metric / (0.1 * max_batch_size))
        if logging.level_debug():
            # logging.debug("(%s) (timestamp=%d) all request queue lengths: %s", self.prefill_or_decode, timestamp, request_queue_lengths)
            # logging.debug("(%s) (timestamp=%d) interval request queue lengths: %s", self.prefill_or_decode, timestamp, interval_request_queue_lengths)
            logging.debug(
                "(%s) (timestamp=%d) Max request queue length: %s, max_batch_size per pod: %d. Recommended num_vPods: %d",
                self.prefill_or_decode,
                timestamp,
                scale_metric,
                max_batch_size,
                num_vPods,
            )
        return max(1, num_vPods)  # at least 1 vPod instance should be preserved

    def get_recommended_vPod_config(
        self, timestamp: int = -1, *args, **kwargs
    ) -> list[ModelConfig]:
        assert timestamp != -1
        # compute average request seqlen in the last VS interval
        prefill_seqlens = self.metrics_server.prefill_seqlens
        decode_seqlens = self.metrics_server.decode_seqlens
        interval_prefill_seqlens = [
            x[1] for x in prefill_seqlens if x[0] >= timestamp - self.vs_window_ns
        ]
        interval_decode_seqlens = [
            x[1] for x in decode_seqlens if x[0] >= timestamp - self.vs_window_ns
        ]
        if len(interval_prefill_seqlens) == 0:
            return [
                self.vPod_config
            ]  # no requests in the last interval, return the current config
        if len(interval_decode_seqlens) == 0:
            return [
                self.vPod_config
            ]  # no requests in the last interval, return the current config
        avg_prefill_seqlen = max(1, math.ceil(np.mean(interval_prefill_seqlens)))
        avg_decode_seqlen = max(1, math.ceil(np.mean(interval_decode_seqlens)))

        self.vPod_config.input_seqlen = autoscaler_lib.pad_seqlen(
            avg_prefill_seqlen,
            self.simulator.config.workload_config.input_seqlen_padding_factors,
            self.simulator.config.workload_config.input_seqlen_padding_steps,
        )
        self.vPod_config.output_seqlen = autoscaler_lib.pad_seqlen(
            avg_decode_seqlen,
            self.simulator.config.workload_config.output_seqlen_padding_factors,
            self.simulator.config.workload_config.output_seqlen_padding_steps,
        )
        self.simulator.config.workload_config.llm_config.input_seqlen = (
            self.vPod_config.input_seqlen
        )
        self.simulator.config.workload_config.llm_config.output_seqlen = (
            self.vPod_config.output_seqlen
        )

        if logging.level_debug():
            logging.debug(
                "(timestamp=%d) Average prefill seqlen: %d, Average decode seqlen: %d",
                timestamp,
                avg_prefill_seqlen,
                avg_decode_seqlen,
            )
            logging.debug(
                "(timestamp=%d) Padded input seqlen: %d, Padded output seqlen: %d",
                timestamp,
                self.vPod_config.input_seqlen,
                self.vPod_config.output_seqlen,
            )

        old_num_chips = self.vPod_config.num_chips
        candidate_configs = autoscaler_lib.get_optimal_vPod_config_with_seqlen_fallback(
            self.simulator.config, self.prefill_or_decode
        )
        self.vPod_config = candidate_configs[0]
        if logging.level_debug():
            logging.debug(
                "(%s) (timestamp=%d) Vertical scaling: changing num_chips: %d -> %d",
                self.prefill_or_decode,
                timestamp,
                old_num_chips,
                self.vPod_config.num_chips,
            )
            logging.debug(
                "(%s) (timestamp=%d) Recommended vPod config: %s",
                self.prefill_or_decode,
                timestamp,
                self.vPod_config,
            )

        return [self.vPod_config]

    def get_vPod_create_delay_ns(self, vPod: "vPodInstance") -> int:
        """
        Get the delay for creating a new vPod in nanoseconds.
        """
        return 0

    def get_vPod_reconfig_delay_ns(
        self, old_config: ModelConfig, new_config: ModelConfig
    ) -> int:
        """
        Get the delay for reconfiguring a vPod in nanoseconds.
        """
        return 0


class NeuScaleAutoScaler(vPodAutoScaler):
    """
    NeuScale auto scaler.
    This class implements the logic for both horizontal and vertical scaling of vPods.
    """

    def __init__(
        self,
        simulator: "NPUFleetSimulator",
        name: str | None = None,
        metrics_server: MetricsServer | None = None,
        prefill_or_decode: str = "prefill",
        enable_mpa: bool = True,
    ):
        super().__init__(
            simulator, name, metrics_server, prefill_or_decode, enable_mpa=enable_mpa
        )

        # find average seqlen of all requests
        assert self.simulator.client
        requests = self.simulator.client.requests
        interval_requests = [
            # (req.input_seqlen, req.output_seqlen if self.prefill_or_decode == "decode" else 4)  # do not care about decode seqlen for prefill vPods
            req
            for req in requests
            if req.enqueue_timestamp
            <= self.mpa_window_ns  # use the first timewindow as warmup
        ]

        recommendations = self.get_recommended_allocation(
            timestamp=self.mpa_window_ns, current_request_queue=interval_requests
        )
        self.current_recommended_allocation: dict[tuple[LLMConfig, ...], int] = {  # type: ignore (self.get_recommended_allocation returns LLMConfigs)
            tuple(configs): count for configs, count in recommendations
        }
        # avg_input_seqlen, avg_output_seqlen = max(  # use Base-Max as initial allocation
        #     [(req.input_seqlen, req.output_seqlen) for req in requests],
        #     key=lambda x: x[0] if prefill_or_decode == "prefill" else (x[0] + x[1]),
        # )

        # avg_input_seqlen, avg_output_seqlen = autoscaler_lib.recommend_seqlen_by_regression_prediction(
        #     interval_seqlens,
        #     self.simulator.config,
        #     self.prefill_or_decode,
        # )
        # input_seqlen = (
        #     autoscaler_lib.pad_seqlen(
        #         avg_input_seqlen,
        #         self.simulator.config.workload_config.input_seqlen_padding_factors,
        #         self.simulator.config.workload_config.input_seqlen_padding_steps,
        #     )
        # )
        # output_seqlen = autoscaler_lib.pad_seqlen(
        #     avg_output_seqlen if prefill_or_decode == "decode" else 4,  # do not care about decode seqlen for prefill vPods
        #     self.simulator.config.workload_config.output_seqlen_padding_factors,
        #     self.simulator.config.workload_config.output_seqlen_padding_steps,
        # )
        # temp_cfg = deepcopy(self.simulator.config)
        # temp_cfg.workload_config.llm_config.input_seqlen = input_seqlen
        # temp_cfg.workload_config.llm_config.output_seqlen = output_seqlen

        # optimal_cfg_for_avg_seqlen = autoscaler_lib.get_optimal_vPod_config(temp_cfg, prefill_or_decode)

        # self.vPod_config = optimal_cfg_for_avg_seqlen[0]

        # self.current_recommended_allocation: dict[tuple[LLMConfig, ...], int] = {(optimal_cfg_for_avg_seqlen[0],): 32}

    def get_initial_allocation(self) -> list[tuple[int, ModelConfig]]:
        return [
            (count, deepcopy(configs[0]))
            for configs, count in self.current_recommended_allocation.items()
        ]

    def get_initial_allocation_alternatives(
        self,
    ) -> list[tuple[int, list[ModelConfig]]]:
        # configs is the per-seqlen version-ordered alternatives tuple (best-efficiency
        # first, e.g. 6e, 4, 5e); expose all of them so the endpoint can spill onto a
        # less-preferred version when the best one hits its max_chips_per_version cap.
        return [
            (count, [deepcopy(cfg) for cfg in configs])
            for configs, count in self.current_recommended_allocation.items()
        ]

    def get_recommended_allocation(
        self,
        timestamp: int = -1,
        current_request_queue: Sequence[LLMRequest] | None = None,
        *args,
        **kwargs,
    ) -> list[tuple[Sequence[ModelConfig], int]]:
        """
        @timestamp: The current timestamp.

        Implements the MPA algorithm for NeuScale.
        First, we analyze the seqlen distribution in the last MPA window to determine which seqlens
        should goto which vPod configuration (VS).
        Then, we determine the number of vPods needed for each configuration based on the max request queue length
        in the last MPA window (HS).
        """
        assert timestamp != -1
        assert self.enable_mpa, "get_recommended_allocation should only be called when multi-dimensional auto-scaling is enabled."

        self.queue_depth_driven_configs = set()
        current_request_queue = (
            list(current_request_queue) if current_request_queue else []
        )

        # get seqlen distribution from historical requests in the last VS interval
        # if timestamp < self.mpa_window_ns:
        #     # Not enough data in the last VS interval. Only use the current request queue.
        #     finished_requests = set(current_request_queue)
        # else:
        # Evict stale requests before reading the window (lazy eviction)
        self.metrics_server.evict_stale_mpa_requests(timestamp, self.prefill_or_decode)
        if self.prefill_or_decode == "prefill":
            finished_requests = self.metrics_server.prefill_requests_in_mpa_time_window
        else:
            finished_requests = self.metrics_server.decode_requests_in_mpa_time_window
            # [req for req in self.metrics_server.request_trace if req.enqueue_timestamp >= timestamp - self.mpa_window_ns]
        # Use the union of historical + queue for seqlen distribution (VS config selection),
        # but keep them separate: only historical requests feed EWMA rate and avg seqlen
        # computation, while queue depth is handled independently by _compute_num_vpods_from_queue_depth.
        all_requests = set(finished_requests).union(set(current_request_queue))
        interval_seqlens = [
            (
                req.input_seqlen,
                req.output_seqlen if self.prefill_or_decode == "decode" else 4,
            )  # do not care about decode seqlen for prefill vPods
            for req in all_requests
        ]

        if logging.level_debug():
            logging.debug(
                "(timestamp=%d) (%s) MPA seqlens: %s",
                timestamp,
                self.prefill_or_decode,
                set(interval_seqlens),
            )

        if len(interval_seqlens) == 0:
            return [
                (list(deepcopy(configs)), count)
                for configs, count in self.current_recommended_allocation.items()
            ]  # Not enough data in the last VS interval. Return current allocation.

        seqlen_to_cfgs_mapping = autoscaler_lib.get_seqlen_to_configs_mapping(
            interval_seqlens, self.simulator.config, self.prefill_or_decode
        )
        if logging.level_debug():
            logging.debug("Seqlen to config mapping: %s", seqlen_to_cfgs_mapping)
        seqlen_counts: dict[tuple[int, int], int] = {}
        for seqlen in interval_seqlens:
            if seqlen not in seqlen_counts:
                seqlen_counts[seqlen] = 1
            else:
                seqlen_counts[seqlen] += 1

        # Build per-config request lists for EWMA-based HS scaling
        # Use parallelism-tuple-based consolidation to merge configs that differ only in seqlens
        _tuple_to_canonical: dict[tuple, tuple[LLMConfig, ...]] = {}
        cfgs_to_requests_mapping: dict[tuple[LLMConfig, ...], list[LLMRequest]] = {}
        # Track the set of (input, output) seqlens served by each canonical group, used to
        # order groups by sequence length for coalescing ("next larger seqlen" target).
        cfgs_to_seqlens: dict[tuple[LLMConfig, ...], set[tuple[int, int]]] = {}
        for req in finished_requests:
            seqlen = (
                req.input_seqlen,
                req.output_seqlen if self.prefill_or_decode == "decode" else 4,
            )
            if seqlen in seqlen_to_cfgs_mapping:
                cfgs_tuple: tuple[LLMConfig, ...] = tuple(
                    seqlen_to_cfgs_mapping[seqlen]
                )
                cfg_key = cfgs_tuple[0].get_chip_version_and_parallelism_degree_tuple()
                canonical = _tuple_to_canonical.setdefault(cfg_key, cfgs_tuple)
                if canonical not in cfgs_to_requests_mapping:
                    cfgs_to_requests_mapping[canonical] = []
                cfgs_to_requests_mapping[canonical].append(req)
                cfgs_to_seqlens.setdefault(canonical, set()).add(seqlen)

        # Build per-config queue depth from current_request_queue only
        cfgs_to_queue_depth: dict[tuple[LLMConfig, ...], int] = {}
        for req in current_request_queue:
            seqlen = (
                req.input_seqlen,
                req.output_seqlen if self.prefill_or_decode == "decode" else 4,
            )
            if seqlen in seqlen_to_cfgs_mapping:
                cfgs_tuple_q: tuple[LLMConfig, ...] = tuple(
                    seqlen_to_cfgs_mapping[seqlen]
                )
                cfg_key = cfgs_tuple_q[
                    0
                ].get_chip_version_and_parallelism_degree_tuple()
                canonical_q = _tuple_to_canonical.setdefault(cfg_key, cfgs_tuple_q)
                cfgs_to_queue_depth[canonical_q] = (
                    cfgs_to_queue_depth.get(canonical_q, 0) + 1
                )
                cfgs_to_seqlens.setdefault(canonical_q, set()).add(seqlen)

        # Ensure configs that appear only in the queue (no historical requests yet)
        # are still processed via the drain path with an empty request list.
        for cfgs_q in cfgs_to_queue_depth:
            if cfgs_q not in cfgs_to_requests_mapping:
                cfgs_to_requests_mapping[cfgs_q] = []

        # Based on EWMA peak rate + queue drain, determine how many vPods are needed for each configuration
        cfgs_to_num_vpods_mapping: dict[tuple[LLMConfig, ...], int] = {}
        cfgs_to_rpeak: dict[
            tuple[LLMConfig, ...], float
        ] = {}  # R_peak per group (req/s)
        cfgs_to_tpod: dict[
            tuple[LLMConfig, ...], float
        ] = {}  # T_pod: single-vPod throughput (req/s)
        mode = self.prefill_or_decode
        for cfgs, group_requests in cfgs_to_requests_mapping.items():
            cfg0 = cfgs[0]
            ewma_key = f"{cfg0.name}_{get_pstr(cfg0)}"

            ewma_peak_rate = self._compute_ewma_peak_rate(
                group_requests, timestamp, self.mpa_window_ns, ewma_key
            )

            ideal_latency = autoscaler_lib.get_ideal_latency_for_config(
                self.simulator.config, cfg0, mode
            )

            # Token-budget effective batch size
            config_batch_size = cfg0.microbatch_size_ici
            if mode == "prefill":
                config_seqlen = cfg0.input_seqlen
                avg_req_seqlen = (
                    float(np.mean([r.input_seqlen for r in group_requests]))
                    if len(group_requests) > 0
                    else float(config_seqlen)
                )
            else:
                config_seqlen = cfg0.input_seqlen + cfg0.output_seqlen
                avg_req_seqlen = (
                    float(np.mean([r.total_seqlen for r in group_requests]))
                    if len(group_requests) > 0
                    else float(config_seqlen)
                )

            token_budget = config_batch_size * config_seqlen
            batch_size = token_budget / max(1.0, avg_req_seqlen)

            avg_output_seqlen = 1.0
            if mode == "decode" and len(group_requests) > 0:
                avg_output_seqlen = float(
                    max(1.0, np.mean([r.output_seqlen for r in group_requests]))
                )

            num_vPods_ewma = self._compute_num_vpods_from_rate(
                ewma_peak_rate, batch_size, ideal_latency, mode, avg_output_seqlen
            )
            queue_depth = cfgs_to_queue_depth.get(cfgs, 0)
            num_vPods_drain = self._compute_num_vpods_from_queue_depth(
                queue_depth, batch_size, ideal_latency, mode, avg_output_seqlen
            )
            if num_vPods_drain > num_vPods_ewma:
                self.queue_depth_driven_configs.add(cfgs)
            num_vPods = max(num_vPods_ewma, num_vPods_drain)
            cfgs_to_num_vpods_mapping[cfgs] = num_vPods

            # Per-group single-vPod throughput T_pod (req/s) and peak arrival R_peak (req/s),
            # used by the coalescing step below (normalized utilization N_L = R_peak / T_pod).
            if mode == "prefill":
                throughput = (
                    batch_size / ideal_latency if ideal_latency > 0 else float("inf")
                )
            else:
                latency_s = ideal_latency / 1000.0
                throughput = (
                    batch_size / (avg_output_seqlen * latency_s)
                    if latency_s > 0 and avg_output_seqlen > 0
                    else float("inf")
                )
            cfgs_to_rpeak[cfgs] = ewma_peak_rate
            cfgs_to_tpod[cfgs] = throughput

            if self.simulator.enable_profile:
                logging.info(
                    "(%s) [MPA-profile] t=%d key=%s | #reqs=%d queue_depth=%d | "
                    "ewma_rate=%.2f ideal_lat=%.4f cfg_bs=%d cfg_seqlen=%d avg_req_seqlen=%.1f "
                    "token_budget=%d eff_bs=%.2f avg_oseq=%.1f throughput=%.4f | "
                    "vpods_ewma=%d vpods_drain=%d -> %d",
                    mode,
                    timestamp,
                    ewma_key,
                    len(group_requests),
                    queue_depth,
                    ewma_peak_rate,
                    ideal_latency,
                    config_batch_size,
                    config_seqlen,
                    avg_req_seqlen,
                    token_budget,
                    batch_size,
                    avg_output_seqlen,
                    throughput,
                    num_vPods_ewma,
                    num_vPods_drain,
                    num_vPods,
                )

        # ---- Coalesce underutilized vPod groups ----
        # When seqlens are sparse, each maps to its own best-fit vPod group, fragmenting
        # capacity into many lightly loaded groups. Merge a group whose normalized peak
        # throughput N_L = R_peak / T_pod (fractional utilization of a single vPod) is below
        # coalesce_nl_threshold into the group serving the next-larger sequence length, IF
        # that target group has spare capacity to absorb the group's peak load without
        # allocating a new vPod. Merged groups get no vPods of their own; at runtime their
        # requests fall through to the next-larger config via the dispatcher's best-fit
        # routing (find_best_fit_config prefers a larger config that can serve the seqlen).
        # Scan groups in ascending N_L and repeat until no further merges apply.
        coalesce_threshold = self.simulator.config.workload_config.coalesce_nl_threshold
        if coalesce_threshold > 0 and len(cfgs_to_num_vpods_mapping) > 1:

            def _rep_seqlen(cfgs: tuple) -> int:
                # representative sequence length of a group (max it serves), for ordering
                sls = cfgs_to_seqlens.get(cfgs)
                if sls:
                    return max(
                        (s[0] if mode == "prefill" else s[0] + s[1]) for s in sls
                    )
                c0 = cfgs[0]
                return (
                    c0.input_seqlen
                    if mode == "prefill"
                    else c0.input_seqlen + c0.output_seqlen
                )

            # Effective per-group load (req/s) = max(EWMA peak arrival rate, queue-implied
            # arrival rate). A group's vPods may be justified entirely by a live queue
            # backlog (num_vPods_drain) while its EWMA over *finished* requests is ~0; using
            # only the EWMA would make such a backlogged group look idle (N_L=0) and get
            # coalesced away with its backlog unaccounted for. The queue-implied rate
            # queue_depth/drain_target is the same arrival rate the drain-based sizing
            # targets, so folding it in makes both the N_L test and the absorb/capacity check
            # see the real demand. This dict is mutable: a target's load grows as it absorbs
            # merged-in groups, so chained merges stay conservative.
            drain_target = (
                self.simulator.config.workload_config.queue_drain_target_seconds
            )

            def _eff_load(cfgs: tuple) -> float:
                ewma = cfgs_to_rpeak.get(cfgs, 0.0)
                q_rate = (
                    (cfgs_to_queue_depth.get(cfgs, 0) / drain_target)
                    if drain_target > 0
                    else 0.0
                )
                return max(ewma, q_rate)

            load = {c: _eff_load(c) for c in cfgs_to_num_vpods_mapping}
            merged_groups: set = set()

            def _nl(cfgs: tuple) -> float:
                tpod = cfgs_to_tpod.get(cfgs, float("inf"))
                return (
                    load.get(cfgs, 0.0) / tpod if tpod and tpod != float("inf") else 0.0
                )

            made_merge = True
            while made_merge:
                made_merge = False
                active = sorted(
                    (c for c in cfgs_to_num_vpods_mapping if c not in merged_groups),
                    key=_nl,
                )
                for L in active:
                    if L in merged_groups or _nl(L) >= coalesce_threshold:
                        continue
                    sL = _rep_seqlen(L)
                    # target = active group serving the smallest seqlen strictly larger than L
                    targets = [
                        c
                        for c in active
                        if c not in merged_groups and c is not L and _rep_seqlen(c) > sL
                    ]
                    if not targets:
                        continue
                    T = min(targets, key=_rep_seqlen)
                    # spare capacity of T = (its vPods * per-vPod throughput) - its effective
                    # load. Absorbing L's smaller requests costs <= load[L] on T's larger-
                    # seqlen pods, so requiring spare >= load[L] is conservative (no new vPod).
                    capacity_T = cfgs_to_num_vpods_mapping[T] * cfgs_to_tpod.get(T, 0.0)
                    if capacity_T - load.get(T, 0.0) >= load.get(L, 0.0):
                        load[T] = load.get(T, 0.0) + load.get(L, 0.0)
                        merged_groups.add(L)
                        made_merge = True

            for L in merged_groups:
                del cfgs_to_num_vpods_mapping[L]
            if merged_groups:
                logging.info(
                    "(%s) (%s) (t=%d) Coalesced %d underutilized vPod group(s) into "
                    "larger-seqlen groups (N_L < %.2f); %d groups remain.",
                    self.simulator.name,
                    mode,
                    timestamp,
                    len(merged_groups),
                    coalesce_threshold,
                    len(cfgs_to_num_vpods_mapping),
                )

        # ---- Aggressive decode pooling (optional) ----
        # Decode is memory-bandwidth-bound: per-token cost = iteration_cost / achieved_batch, so
        # concentrating the WHOLE decode arrival stream onto one config (a single deep queue)
        # amortizes batch far better than splitting it across per-seqlen groups, which each sit
        # at batch~1 and get no amortization. Coalescing alone stops short (groups become rated-
        # "busy" and the capacity check blocks further merges as load rises), leaving ~3 decode
        # groups at high rate. When enabled, collapse ALL remaining decode groups into the single
        # largest-seqlen umbrella config (memory-safe -- it can serve every shorter sequence) and
        # re-size that umbrella from the TOTAL pooled decode load, mirroring the single-config
        # Base-Max baseline that wins decode $/token at high rate. Prefill is deliberately left
        # fragmented (compute-bound: more groups = more burst capacity at ~no per-token cost).
        if (
            mode == "decode"
            and self.simulator.config.workload_config.decode_pool_single_config
            and len(cfgs_to_num_vpods_mapping) > 1
        ):

            def _grp_max_seqlen(cfgs: tuple) -> int:
                sls = cfgs_to_seqlens.get(cfgs)
                return (
                    max((s[0] + s[1]) for s in sls)
                    if sls
                    else (cfgs[0].input_seqlen + cfgs[0].output_seqlen)
                )

            # Cache the umbrella config so it is STABLE across ticks: re-selecting it every tick
            # (the largest-seqlen group can change as the distribution shifts) churns the decode
            # pool -- vPods torn down/rebuilt never serve, the backlog balloons, and the 1s queue-
            # drain term then demands a huge (physically impossible) vPod count. A persistent
            # umbrella lets the endpoint keep one stable decode pool that actually drains.
            cached = getattr(self, "_decode_pool_umbrella", None)
            if cached is not None and cached in cfgs_to_num_vpods_mapping:
                umbrella = cached
            else:
                umbrella = max(cfgs_to_num_vpods_mapping, key=_grp_max_seqlen)
                self._decode_pool_umbrella = umbrella
            pooled_requests: list[LLMRequest] = []
            pooled_queue = 0
            for cfgs in list(cfgs_to_num_vpods_mapping):
                pooled_requests += cfgs_to_requests_mapping.get(cfgs, [])
                pooled_queue += cfgs_to_queue_depth.get(cfgs, 0)
                if cfgs is not umbrella:
                    cfgs_to_seqlens.setdefault(umbrella, set()).update(
                        cfgs_to_seqlens.get(cfgs, set())
                    )
                    del cfgs_to_num_vpods_mapping[cfgs]
            # Re-size the umbrella from the TOTAL pooled decode load (same token-budget effective
            # batch + EWMA/drain sizing used per-group above, now applied to the merged stream).
            cfg0 = umbrella[0]
            ideal_latency = autoscaler_lib.get_ideal_latency_for_config(
                self.simulator.config, cfg0, mode
            )
            config_seqlen = cfg0.input_seqlen + cfg0.output_seqlen
            # Seqlen stats for throughput sizing: prefer finished requests, but fall back to the
            # live QUEUE (not a constant) so the warm-up phase -- when nothing has finished yet and
            # avg_output_seqlen would otherwise default to 1.0, ~26x over-estimating throughput --
            # provisions enough vPods to drain the initial backlog instead of letting it pile up.
            avg_src = (
                pooled_requests if pooled_requests else list(current_request_queue)
            )
            avg_req_seqlen = (
                float(np.mean([r.total_seqlen for r in avg_src]))
                if avg_src
                else float(config_seqlen)
            )
            batch_size = (cfg0.microbatch_size_ici * config_seqlen) / max(
                1.0, avg_req_seqlen
            )
            avg_output_seqlen = (
                float(max(1.0, np.mean([r.output_seqlen for r in avg_src])))
                if avg_src
                else 1.0
            )
            # Size by ARRIVAL rate over the window: (finished + still-queued) decode requests all
            # arrived within the window, so this is the demand the single pooled config must meet
            # -- exactly how the single-config Base-Max baseline sizes (a stable count that keeps
            # up and drains any transient backlog over the umbrella's real capacity). The per-tick
            # 1s queue-drain term is deliberately NOT used here: on the concentrated pooled queue
            # it demands a physically impossible drain (tens of vPods for a few-thousand backlog),
            # oscillating the count and churning the pool so it never actually catches up.
            window_s = max(1e-9, self.mpa_window_ns / 1e9)
            arrival_rate = (len(pooled_requests) + pooled_queue) / window_s
            cfgs_to_num_vpods_mapping[umbrella] = max(
                1,
                self._compute_num_vpods_from_rate(
                    arrival_rate, batch_size, ideal_latency, mode, avg_output_seqlen
                ),
            )
            logging.info(
                "(%s) (decode) (t=%d) Pooled all decode groups into single umbrella config "
                "%s (serves seqlen<=%d) -> %d vPods (arrival_rate=%.2f req/s, queue=%d).",
                self.simulator.name,
                timestamp,
                get_pstr(cfg0),
                config_seqlen,
                cfgs_to_num_vpods_mapping[umbrella],
                arrival_rate,
                pooled_queue,
            )

        # Expose each remaining group's served-seqlen range (keyed by version+parallelism,
        # which is seqlen-independent and matches the dispatcher's grouping key). The runtime
        # dispatcher uses this as a fallback when a deployed config has no exact-match request
        # in the current tick: without it the config's per-tick range is (-1,-1) and
        # find_best_fit_config skips it, stranding requests -- notably requests whose own
        # group was just coalesced away into this (possibly momentarily idle) larger group.
        self.config_to_seqlen_range_fallback: dict[tuple, tuple[int, int]] = {}
        for cfgs in cfgs_to_num_vpods_mapping:
            sls = cfgs_to_seqlens.get(cfgs)
            if not sls:
                continue
            vals = [s[0] if mode == "prefill" else s[0] + s[1] for s in sls]
            key = cfgs[0].get_chip_version_and_parallelism_degree_tuple()
            self.config_to_seqlen_range_fallback[key] = (min(vals), max(vals))

        if logging.level_debug():
            logging.debug(
                "(timestamp=%d) Recommended allocation: %s",
                timestamp,
                [
                    ([(cfg.name, cfg.num_chips) for cfg in configs], count)
                    for configs, count in cfgs_to_num_vpods_mapping.items()
                ],
            )

        self.current_recommended_allocation = cfgs_to_num_vpods_mapping
        return [
            (list(deepcopy(configs)), count)
            for configs, count in cfgs_to_num_vpods_mapping.items()
        ]

    def get_recommended_num_vPods(
        self, timestamp: int = -1, max_batch_size: int = -1, *args, **kwargs
    ) -> int:
        """
        @timestamp: The current timestamp.

        Implements separated HS+VS algorithm for NeuScale (can serve as a baseline).
        Scale based on the max request queue length in the last HS interval.
        Try to maintain:
            max request queue length == 0.6 * (max_batch_size * num_vPods)
        """
        raise NotImplementedError(
            "NeuScaleAutoScaler.get_recommended_num_vPods is not implemented yet. Use get_recommended_allocation instead."
        )
        # assert timestamp != -1
        # assert not self.enable_mpa or max_batch_size > 0, \
        #     "get_recommended_num_vPods should not be called without a specified max_batch_size when multi-dimensional auto-scaling is enabled."
        # # compute average request queue length in the last HS interval
        # if self.prefill_or_decode == "prefill":
        #     request_queue_lengths = self.metrics_server.prefill_queue_lengths
        # else:
        #     request_queue_lengths = self.metrics_server.decode_queue_lengths
        # interval_request_queue_lengths = [x[1] for x in request_queue_lengths if x[0] >= timestamp - self.hs_window_ns]
        # max_batch_size = max_batch_size if max_batch_size > 0 else self.vPod_config.microbatch_size_ici

        # scale_metric = max(interval_request_queue_lengths) if len(interval_request_queue_lengths) > 0 else 0
        # num_vPods = math.ceil(scale_metric / (0.6 * max_batch_size))
        # if logging.level_debug():
        #     # logging.debug("(%s) (timestamp=%d) all request queue lengths: %s", self.prefill_or_decode, timestamp, request_queue_lengths)
        #     # logging.debug("(%s) (timestamp=%d) interval request queue lengths: %s", self.prefill_or_decode, timestamp, interval_request_queue_lengths)
        #     logging.debug("(%s) (timestamp=%d) Max request queue length: %s, max_batch_size per pod: %d. Recommended num_vPods: %d", self.prefill_or_decode, timestamp, scale_metric, max_batch_size, num_vPods)
        # return max(1, num_vPods)  # at least 1 vPod instance should be preserved

    def get_recommended_vPod_config(
        self, timestamp: int = -1, *args, **kwargs
    ) -> list[LLMConfig]:  # type: ignore
        """
        @timestamp: The current timestamp.

        Implements separated HS+VS algorithm for NeuScale (can serve as a baseline).
        Scale based on the seqlen distribution in the last VS interval.
        Use linear regression to predict the target seqlen.
        Assumes all instances have the same configuration.
        """
        raise NotImplementedError(
            "NeuScaleAutoScaler.get_recommended_vPod_config is not implemented yet. Use get_recommended_allocation instead."
        )
        # assert timestamp != -1
        # assert not self.enable_mpa, "get_recommended_vPod_config should not be called when multi-dimensional auto-scaling is enabled."
        # if timestamp < self.vs_window_ns:
        #     return [self.vPod_config]  # Not enough data in the last VS interval. Return current config.

        # finished_requests = [req for req in self.metrics_server.request_trace if req.enqueue_timestamp >= timestamp - self.vs_window_ns]
        # interval_seqlens = [
        #     (req.input_seqlen, req.output_seqlen if self.prefill_or_decode == "decode" else 4)  # do not care about decode seqlen for prefill vPods
        #     for req in finished_requests
        # ]
        # # else:
        #     # raise ValueError(f"Unknown prefill_or_decode mode: {self.prefill_or_decode}")
        # if len(interval_seqlens) == 0:
        #     return [self.vPod_config]
        # # assert all([s[0] != 0 for s in interval_seqlens]), "All input seqlens must be non-zero"
        # prefill_seqlen, decode_seqlen = autoscaler_lib.recommend_seqlen_by_regression_prediction(
        #     interval_seqlens,
        #     self.simulator.config,
        #     self.prefill_or_decode,
        # )

        # self.vPod_config.input_seqlen = autoscaler_lib.pad_seqlen(
        #     prefill_seqlen,
        #     self.simulator.config.workload_config.input_seqlen_padding_factors,
        #     self.simulator.config.workload_config.input_seqlen_padding_steps,
        # )
        # self.vPod_config.output_seqlen = autoscaler_lib.pad_seqlen(
        #     decode_seqlen if self.prefill_or_decode == "decode" else 4,
        #     self.simulator.config.workload_config.output_seqlen_padding_factors,
        #     self.simulator.config.workload_config.output_seqlen_padding_steps,
        # )
        # temp_cfg = deepcopy(self.simulator.config)
        # temp_cfg.workload_config.llm_config.input_seqlen = self.vPod_config.input_seqlen
        # temp_cfg.workload_config.llm_config.output_seqlen = self.vPod_config.output_seqlen

        # if logging.level_debug():
        #     logging.debug("(timestamp=%d) Target prefill seqlen: %d, Target decode seqlen: %d", timestamp, prefill_seqlen, decode_seqlen)
        #     logging.debug("(timestamp=%d) Padded input seqlen: %d, Padded output seqlen: %d", timestamp, self.vPod_config.input_seqlen, self.vPod_config.output_seqlen)

        # old_num_chips = self.vPod_config.num_chips
        # candidate_cfgs = autoscaler_lib.get_optimal_vPod_config(temp_cfg, self.prefill_or_decode)
        # self.vPod_config = candidate_cfgs[0]

        # if logging.level_debug():
        #     logging.debug("(%s) (timestamp=%d) Vertical scaling: changing num_chips: %d -> %d", self.prefill_or_decode, timestamp, old_num_chips, self.vPod_config.num_chips)
        #     logging.debug("(%s) (timestamp=%d) Recommended vPod config: %s", self.prefill_or_decode, timestamp, self.vPod_config)

        # return candidate_cfgs

    def get_vPod_create_delay_ns(self, vPod: "LLMInferenceInstanceBase") -> int:  # type: ignore
        """
        Get the delay for creating a new vPod in nanoseconds.
        """
        VM_startup_delay = int(vPod.config.instance_startup_delay_sec * 1e9)
        model_weight_bytes_per_chip = (
            mem_footprint_lib.get_llm_inference_weight_mem_requirement(
                vPod.config.llm_config,
                1 if "deepseek" in vPod.config.model_name else 2,
            )
        )
        model_weight_load_bw_per_chip_GBps = vPod.config.llm_config.dcn_bw_GBps
        model_weight_load_delay = int(
            model_weight_bytes_per_chip
            / (model_weight_load_bw_per_chip_GBps * 1024 * 1024 * 1024)
        ) * int(1e9)
        return VM_startup_delay + model_weight_load_delay
        # return int(1e9)  # for debugging only

    def get_vPod_reconfig_delay_ns(
        self, old_config: ModelConfig, new_config: ModelConfig
    ) -> int:
        """
        Get the delay for reconfiguring a vPod in nanoseconds.
        """
        # VM_startup_delay = int(1e9)
        # old_bytes_per_chip = mem_footprint_lib.get_llm_inference_weight_mem_requirement(
        #     old_config,  # type: ignore # For now, assuming this is always LLMConfig
        #     1 if "deepseek" in self.simulator.config.workload_config.model_name else 2,
        # )
        # new_bytes_per_chip = mem_footprint_lib.get_llm_inference_weight_mem_requirement(
        #     new_config,  # type: ignore # For now, assuming this is always LLMConfig
        #     1 if "deepseek" in self.simulator.config.workload_config.model_name else 2,
        # )
        # model_weight_load_bw_per_chip_GBps = min(old_config.dcn_bw_GBps, new_config.dcn_bw_GBps)
        # reconfig_network_delay = int(abs(new_bytes_per_chip - old_bytes_per_chip) / (model_weight_load_bw_per_chip_GBps * 1024 * 1024 * 1024)) * int(1e9)
        # return VM_startup_delay + reconfig_network_delay
        return int(1e9)  # for debugging only


class VerticalAutoScaler(vPodAutoScaler):
    """
    Vertical auto scaler.
    This class implements the logic for vertical scaling of vPods.
    It does not implement horizontal scaling.
    Mainly for debugging purposes only.
    """

    def __init__(
        self,
        simulator: "NPUFleetSimulator",
        name: str | None = None,
        metrics_server: MetricsServer | None = None,
        prefill_or_decode: str = "prefill",
        num_vPods: int = 1,
    ):
        super().__init__(simulator, name, metrics_server, prefill_or_decode)
        self.vPod_config: ModelConfig = (
            autoscaler_lib.get_optimal_vPod_config_with_seqlen_fallback(
                self.simulator.config, prefill_or_decode
            )[0]
        )
        self.num_vPods: int = num_vPods

    def get_initial_allocation(self) -> list[tuple[int, ModelConfig]]:
        return [(self.num_vPods, self.vPod_config)]

    def get_recommended_vPod_config(
        self, timestamp: int = -1, *args, **kwargs
    ) -> list[ModelConfig]:
        assert timestamp != -1
        # TODO: For debugging only now.
        num_chips = self.vPod_config.num_chips
        if num_chips < 2048:
            # self.vPod_config = autoscaler_lib.get_optimal_vPod_config(self.simulator.config.workload_config.llm_config)
            self.vPod_config.num_chips = num_chips * 2
            self.vPod_config.tensor_parallelism_degree *= 2
            if self.vPod_config.num_tensor_parallel_axes == 0:
                self.vPod_config.num_tensor_parallel_axes = 1
            logging.debug(
                "Vertical scaling: changing num_chips: %d -> %d",
                num_chips,
                self.vPod_config.num_chips,
            )
        return [self.vPod_config]

    def get_recommended_num_vPods(self, *args, **kwargs) -> int:
        # Vertical scaling does not change the number of vPods.
        return self.num_vPods

    def get_vPod_reconfig_delay_ns(
        self, old_config: ModelConfig, new_config: ModelConfig
    ) -> int:
        """
        Get the delay for reconfiguring a vPod in nanoseconds.
        """
        return int(1e9)  # for debugging purposes

    def schedule_next_hs_event(self, event: vPodHorizontalScalingRecommendationEvent):
        return  # do not perform horizontal scaling for this class


class MultiPoolAutoScaler(vPodAutoScaler):
    """
    Multi-pool auto scaler.
    Creates a fixed number of pools, each with a static vPod config optimized for a
    percentile of the sequence length distribution. Only pool sizes (num vPods) change
    at runtime via MPA. Conceptually, HorizontalAutoScaler is a special case with 1 pool.
    """

    def __init__(
        self,
        simulator: "NPUFleetSimulator",
        name: str | None = None,
        metrics_server: MetricsServer | None = None,
        prefill_or_decode: str = "prefill",
        enable_mpa: bool = True,
    ):
        super().__init__(
            simulator, name, metrics_server, prefill_or_decode, enable_mpa=enable_mpa
        )

        num_pools = self.simulator.config.workload_config.num_pools
        assert num_pools >= 1, f"num_pools must be >= 1, got {num_pools}"

        # Collect all requests from the trace
        assert self.simulator.client
        requests = self.simulator.client.requests

        # Compute effective seqlen per request
        if prefill_or_decode == "prefill":
            effective_seqlens = [req.input_seqlen for req in requests]
        else:
            effective_seqlens = [
                req.input_seqlen + req.output_seqlen for req in requests
            ]

        # Compute percentile boundaries
        percentiles = [100.0 * (i + 1) / num_pools for i in range(num_pools)]
        self.boundaries: list[float] = [
            float(np.percentile(effective_seqlens, p)) for p in percentiles
        ]

        logging.info(
            "(%s) (%s) MultiPoolAutoScaler: %d pools, percentiles=%s, boundaries=%s",
            self.simulator.name,
            self.prefill_or_decode,
            num_pools,
            percentiles,
            self.boundaries,
        )

        # For each pool, determine representative (input_seqlen, output_seqlen) and look up optimal config
        self.pool_configs: list[list[LLMConfig]] = []
        for i, boundary in enumerate(self.boundaries):
            if prefill_or_decode == "prefill":
                rep_input = math.ceil(boundary)
                rep_output = 4
            else:  # decode
                # Find request whose total_seqlen is closest to boundary
                closest_req = min(
                    requests,
                    key=lambda r: abs((r.input_seqlen + r.output_seqlen) - boundary),
                )
                rep_input = closest_req.input_seqlen
                rep_output = closest_req.output_seqlen

            # Pad seqlens
            padded_input = autoscaler_lib.pad_seqlen(
                rep_input,
                self.simulator.config.workload_config.input_seqlen_padding_factors,
                self.simulator.config.workload_config.input_seqlen_padding_steps,
            )
            padded_output = autoscaler_lib.pad_seqlen(
                rep_output if prefill_or_decode == "decode" else 4,
                self.simulator.config.workload_config.output_seqlen_padding_factors,
                self.simulator.config.workload_config.output_seqlen_padding_steps,
            )

            temp_cfg = deepcopy(self.simulator.config)
            temp_cfg.workload_config.llm_config.input_seqlen = padded_input
            temp_cfg.workload_config.llm_config.output_seqlen = padded_output

            optimal_cfgs = autoscaler_lib.get_optimal_vPod_config_with_seqlen_fallback(
                temp_cfg, prefill_or_decode
            )
            self.pool_configs.append(optimal_cfgs)

            logging.info(
                "(%s) (%s) Pool %d: boundary=%.1f, rep_seqlen=(%d,%d), padded=(%d,%d), "
                "optimal_config: type=%s, num_chips=%s, microbatch_size_ici=%s",
                self.simulator.name,
                self.prefill_or_decode,
                i,
                boundary,
                rep_input,
                rep_output,
                padded_input,
                padded_output,
                optimal_cfgs[0].name,
                optimal_cfgs[0].num_chips,
                optimal_cfgs[0].microbatch_size_ici,
            )

        # Build mapping from config identity to pool index for dispatch routing
        self._config_to_pool_index: dict[int, int] = {}
        for i, cfgs in enumerate(self.pool_configs):
            self._config_to_pool_index[id(cfgs[0])] = i

        # Compute initial allocation using first MPA window of requests
        interval_requests = [
            req for req in requests if req.enqueue_timestamp <= self.mpa_window_ns
        ]
        recommendations = self.get_recommended_allocation(
            timestamp=self.mpa_window_ns, current_request_queue=interval_requests
        )
        self.current_recommended_allocation: dict[tuple[ModelConfig, ...], int] = {
            tuple(configs): count for configs, count in recommendations
        }

    def get_initial_allocation(self) -> list[tuple[int, ModelConfig]]:
        return [
            (count, deepcopy(configs[0]))
            for configs, count in self.current_recommended_allocation.items()
        ]

    def get_recommended_allocation(
        self,
        timestamp: int = -1,
        current_request_queue: Sequence[LLMRequest] | None = None,
        *args,
        **kwargs,
    ) -> list[tuple[Sequence[ModelConfig], int]]:
        assert timestamp != -1
        assert self.enable_mpa

        self.queue_depth_driven_configs = set()
        current_request_queue = (
            list(current_request_queue) if current_request_queue else []
        )

        # Evict stale requests before reading the window (lazy eviction)
        self.metrics_server.evict_stale_mpa_requests(timestamp, self.prefill_or_decode)

        # Gather requests from metrics server + current queue
        if self.prefill_or_decode == "prefill":
            finished_requests = self.metrics_server.prefill_requests_in_mpa_time_window
        else:
            finished_requests = self.metrics_server.decode_requests_in_mpa_time_window
        # Use the union of historical + queue for pool classification,
        # but keep them separate: only finished requests feed EWMA rate computation.
        all_requests = set(finished_requests).union(set(current_request_queue))

        if len(all_requests) == 0:
            return [
                (list(deepcopy(configs)), count)
                for configs, count in self.current_recommended_allocation.items()
            ]

        # Classify requests into pools by effective seqlen range
        pool_all_requests: list[list[LLMRequest]] = [
            [] for _ in range(len(self.pool_configs))
        ]
        pool_finished_requests: list[list[LLMRequest]] = [
            [] for _ in range(len(self.pool_configs))
        ]
        finished_requests_set = set(finished_requests)
        for req in all_requests:
            if self.prefill_or_decode == "prefill":
                eff_seqlen = req.input_seqlen
            else:
                eff_seqlen = req.input_seqlen + req.output_seqlen
            pool_idx = self._get_pool_index_for_seqlen(eff_seqlen)
            pool_all_requests[pool_idx].append(req)
            if req in finished_requests_set:
                pool_finished_requests[pool_idx].append(req)

        # Build per-pool queue depth from current_request_queue only
        pool_queue_depths: list[int] = [0] * len(self.pool_configs)
        for req in current_request_queue:
            if self.prefill_or_decode == "prefill":
                eff_seqlen = req.input_seqlen
            else:
                eff_seqlen = req.input_seqlen + req.output_seqlen
            pool_idx = self._get_pool_index_for_seqlen(eff_seqlen)
            pool_queue_depths[pool_idx] += 1

        if logging.level_debug():
            logging.debug(
                "(timestamp=%d) (%s) MultiPool request counts per pool: %s",
                timestamp,
                self.prefill_or_decode,
                [len(pr) for pr in pool_all_requests],
            )

        # For each pool with requests, compute num_vPods using EWMA peak rate
        mode = self.prefill_or_decode
        cfgs_to_num_vpods: dict[tuple[ModelConfig, ...], int] = {}
        for i, group_all_requests in enumerate(pool_all_requests):
            if len(group_all_requests) == 0:
                continue
            group_finished = pool_finished_requests[i]
            cfgs_tuple = tuple(self.pool_configs[i])
            cfg0 = self.pool_configs[i][0]
            ewma_key = f"pool_{i}"

            ewma_peak_rate = self._compute_ewma_peak_rate(
                group_finished, timestamp, self.mpa_window_ns, ewma_key
            )

            ideal_latency = autoscaler_lib.get_ideal_latency_for_config(
                self.simulator.config, cfg0, mode
            )

            # Token-budget effective batch size (use finished requests for avg seqlen)
            config_batch_size = cfg0.microbatch_size_ici
            if mode == "prefill":
                config_seqlen = cfg0.input_seqlen
                avg_req_seqlen = (
                    float(np.mean([r.input_seqlen for r in group_finished]))
                    if len(group_finished) > 0
                    else float(config_seqlen)
                )
            else:
                config_seqlen = cfg0.input_seqlen + cfg0.output_seqlen
                avg_req_seqlen = (
                    float(np.mean([r.total_seqlen for r in group_finished]))
                    if len(group_finished) > 0
                    else float(config_seqlen)
                )

            token_budget = config_batch_size * config_seqlen
            batch_size = token_budget / max(1.0, avg_req_seqlen)

            avg_output_seqlen = 1.0
            if mode == "decode" and len(group_finished) > 0:
                avg_output_seqlen = float(
                    max(1.0, np.mean([r.output_seqlen for r in group_finished]))
                )

            num_vPods_ewma = self._compute_num_vpods_from_rate(
                ewma_peak_rate, batch_size, ideal_latency, mode, avg_output_seqlen
            )
            num_vPods_drain = self._compute_num_vpods_from_queue_depth(
                pool_queue_depths[i], batch_size, ideal_latency, mode, avg_output_seqlen
            )
            if num_vPods_drain > num_vPods_ewma:
                self.queue_depth_driven_configs.add(cfgs_tuple)
            num_vPods = max(num_vPods_ewma, num_vPods_drain)

            # Merge pools that share the same config tuple
            if cfgs_tuple in cfgs_to_num_vpods:
                cfgs_to_num_vpods[cfgs_tuple] += num_vPods
            else:
                cfgs_to_num_vpods[cfgs_tuple] = num_vPods

        if logging.level_debug():
            logging.debug(
                "(timestamp=%d) MultiPool recommended allocation: %s",
                timestamp,
                [
                    ([(cfg.name, cfg.num_chips) for cfg in configs], count)
                    for configs, count in cfgs_to_num_vpods.items()
                ],
            )

        self.current_recommended_allocation = cfgs_to_num_vpods
        return [
            (list(deepcopy(configs)), count)
            for configs, count in cfgs_to_num_vpods.items()
        ]

    def _get_pool_index_for_seqlen(self, eff_seqlen: float) -> int:
        """Return the pool index for a given effective sequence length."""
        for i, boundary in enumerate(self.boundaries):
            if eff_seqlen <= boundary:
                return i
        return len(self.boundaries) - 1  # last pool catches everything above

    def get_seqlens_for_config(
        self, seqlen_pairs: Sequence[tuple[int, int]], config: LLMConfig
    ) -> set[tuple[int, int]]:
        """Pool-boundary-based matching: return seqlen pairs that fall in the pool(s) for config."""
        # Find which pool(s) this config belongs to
        pool_indices: list[int] = []
        deployed_key = config.get_chip_version_and_parallelism_degree_tuple()
        for i, cfgs in enumerate(self.pool_configs):
            candidate = cfgs[0]
            # The endpoint deliberately replaces deployed configs' sequence lengths
            # with dummy values because those lengths describe requests, not vPod
            # identity. Match on the same stable hardware/parallelism identity used
            # elsewhere in FleetSim so that this mutation cannot strand a pool.
            if (
                candidate.get_chip_version_and_parallelism_degree_tuple()
                == deployed_key
            ):
                pool_indices.append(i)

        if not pool_indices:
            return set()

        relevant_seqlens: set[tuple[int, int]] = set()
        for input_sl, output_sl in seqlen_pairs:
            if self.prefill_or_decode == "prefill":
                eff_seqlen = input_sl
            else:
                eff_seqlen = input_sl + output_sl
            pool_idx = self._get_pool_index_for_seqlen(eff_seqlen)
            if pool_idx in pool_indices:
                relevant_seqlens.add((input_sl, output_sl))

        return relevant_seqlens

    def get_recommended_num_vPods(self, *args, **kwargs) -> int:
        raise NotImplementedError(
            "MultiPoolAutoScaler.get_recommended_num_vPods is not implemented. Use get_recommended_allocation instead."
        )

    def get_recommended_vPod_config(self, *args, **kwargs) -> list[ModelConfig]:
        raise NotImplementedError(
            "MultiPoolAutoScaler.get_recommended_vPod_config is not implemented. Use get_recommended_allocation instead."
        )

    def get_vPod_create_delay_ns(self, vPod: "LLMInferenceInstanceBase") -> int:  # type: ignore
        VM_startup_delay = int(vPod.config.instance_startup_delay_sec * 1e9)
        model_weight_bytes_per_chip = (
            mem_footprint_lib.get_llm_inference_weight_mem_requirement(
                vPod.config.llm_config,
                1 if "deepseek" in vPod.config.model_name else 2,
            )
        )
        model_weight_load_bw_per_chip_GBps = vPod.config.llm_config.dcn_bw_GBps
        model_weight_load_delay = int(
            model_weight_bytes_per_chip
            / (model_weight_load_bw_per_chip_GBps * 1024 * 1024 * 1024)
        ) * int(1e9)
        return VM_startup_delay + model_weight_load_delay

    def get_vPod_reconfig_delay_ns(
        self, old_config: ModelConfig, new_config: ModelConfig
    ) -> int:
        return int(1e9)


class StaticAutoScaler(vPodAutoScaler):
    """
    Static auto scaler. Does not perform any auto-scaling.
    Provisions a fixed set of homogeneous vPods at startup based on a user-provided config.
    Uses non-MPA dispatch (same as HorizontalAutoScaler).
    Overrides schedule_next_hs_event and schedule_next_vs_event to prevent recursive scaling.
    """

    def __init__(
        self,
        simulator: "NPUFleetSimulator",
        name: str | None = None,
        metrics_server: MetricsServer | None = None,
        prefill_or_decode: str = "prefill",
    ):
        super().__init__(
            simulator,
            name,
            metrics_server,
            prefill_or_decode,
            enable_mpa=False,
            enable_autoscaling_events=False,
        )

        # Build static allocation from config
        alloc_config = simulator.config.workload_config.static_vpod_allocation
        assert (
            alloc_config is not None
        ), "static_vpod_allocation must be set in workload config for StaticAutoScaler"
        entry = (
            alloc_config.prefill
            if prefill_or_decode == "prefill"
            else alloc_config.decode
        )
        self._vpod_config = self._build_llm_config(entry)
        self._num_vpods = entry.count

    def _build_llm_config(self, entry) -> LLMConfig:
        """Build a full LLMConfig from a StaticVPodEntry by merging
        base model config + target chip config + user parallelism params."""
        from neusim.configs.models.LLMConfig import DeepSeekConfig

        base_model = self.simulator.config.workload_config.llm_config
        chip_config = self.simulator.cluster_manager.chip_configs[entry.npu_type]  # type: ignore

        # Start with base model (has all fields), override chip-specific fields
        config_dict = base_model.model_dump()
        config_dict.update(chip_config.model_dump())

        # Override parallelism and batch params
        config_dict["num_chips"] = entry.num_chips
        config_dict["data_parallelism_degree"] = entry.dp
        config_dict["tensor_parallelism_degree"] = entry.tp
        config_dict["pipeline_parallelism_degree"] = entry.pp
        config_dict["microbatch_size_ici"] = entry.batch_size
        config_dict["global_batch_size"] = entry.batch_size * entry.pp
        config_dict["microbatch_size_dcn"] = entry.batch_size * entry.pp
        config_dict["name"] = entry.npu_type
        # Set seqlens to dummy values (not used by dispatch)
        config_dict["input_seqlen"] = 32
        config_dict["output_seqlen"] = 32

        if entry.ep > 1:
            config_dict["expert_parallelism_degree"] = entry.ep

        model_name = self.simulator.config.workload_config.model_name
        if "deepseek" in model_name.lower():
            return DeepSeekConfig.model_validate(config_dict)
        return LLMConfig.model_validate(config_dict)

    def get_initial_allocation(self) -> list[tuple[int, ModelConfig]]:
        return [(self._num_vpods, self._vpod_config)]

    def get_recommended_allocation(self, *args, **kwargs):
        return [([self._vpod_config], self._num_vpods)]

    def get_recommended_num_vPods(self, *args, **kwargs) -> int:
        return self._num_vpods

    def get_recommended_vPod_config(self, *args, **kwargs) -> list[ModelConfig]:
        return [self._vpod_config]

    def schedule_next_hs_event(self, event):
        return  # Do not schedule recursive horizontal scaling events

    def schedule_next_vs_event(self, event):
        return  # Do not schedule recursive vertical scaling events

    def get_vPod_create_delay_ns(self, vPod) -> int:
        return 0  # All vPods created at init, no runtime creation

    def get_vPod_reconfig_delay_ns(self, old_config, new_config) -> int:
        return 0  # No reconfiguration


def get_autoscaler_by_name(name: str) -> type[vPodAutoScaler]:
    """
    Get the auto scaler class by name.
    """
    if name == "HorizontalAutoScaler":
        return HorizontalAutoScaler
    elif name == "IdealAutoScaler":
        return IdealAutoScaler
    elif name == "NeuScaleAutoScaler":
        return NeuScaleAutoScaler
    elif name == "VerticalAutoScaler":
        return VerticalAutoScaler
    elif name == "MultiPoolAutoScaler":
        return MultiPoolAutoScaler
    elif name == "StaticAutoScaler":
        return StaticAutoScaler
    else:
        raise ValueError(f"Unknown auto scaler name: {name}")
