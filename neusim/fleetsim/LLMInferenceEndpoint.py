import gzip
import hashlib
import json
import math
import os
import pickle
import typing
import uuid
from collections import deque
from collections.abc import Sequence
from copy import copy, deepcopy

from absl import logging

from neusim.configs.models.LLMConfig import (
    DeepSeekConfig,
    LLMConfig,
)
from neusim.configs.systems.NPUFleetConfig import (
    PhysicalCubeConfig,
)
from neusim.eventsim.Event import EventListener
from neusim.fleetsim.NPUClusterManager import NPUPodAllocationRequest

if typing.TYPE_CHECKING:
    from neusim.fleetsim.NPUFleetSimulator import NPUFleetSimulator
# from neusim.fleetsim.util import ListMap
import neusim.fleetsim.npusim_backend_interface as sim_backend
import neusim.fleetsim.util as util
import neusim.fleetsim.vPodAutoScaler as vPodAutoScaler
import neusim.fleetsim.vPodAutoScaler_lib as autoscaler_lib
from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
)
from neusim.fleetsim import cost_model
from neusim.fleetsim.LLMInferenceEvents import (
    LLMInferenceDecodeIterationEndEvent,
    LLMInferenceDecodeIterationStartEvent,
    LLMInferenceEngineReadyEvent,
    LLMInferencePrefillEndEvent,
    LLMInferencePrefillStartEvent,
    LLMInferenceRequestEnqueueEvent,
    vPodCreateEvent,
    vPodDestroyEvent,
    vPodHorizontalScalingEvent,
    vPodHorizontalScalingRecommendationEvent,
    vPodMultiDimensionalScalingEvent,
    vPodMultiDimensionalScalingRecommendationEvent,
    vPodReconfigurationEndEvent,
    vPodReconfigurationStartEvent,
    vPodVerticalScalingEvent,
    vPodVerticalScalingRecommendationEvent,
)
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.fleetsim.SimObject import SimObject


def get_decode_schedule_num_iterations(
    requests: Sequence[LLMRequest],
    min_num_iterations: int,
    max_num_iterations: int,
) -> int:
    """Choose a decode chunk without running any request past its final token."""
    if not requests:
        raise ValueError("Cannot schedule an empty decode request batch.")

    output_seqlen = max(request.output_seqlen for request in requests)
    scheduled_iterations = (
        min_num_iterations
        if output_seqlen <= min_num_iterations * 64
        else min(max_num_iterations, math.ceil(output_seqlen / 32))
    )

    remaining_iterations = min(
        request.output_seqlen - request.current_decode_step for request in requests
    )
    if remaining_iterations <= 0:
        raise ValueError("Cannot schedule a request whose decode phase is complete.")
    return min(scheduled_iterations, remaining_iterations)


def _stable_prediction_integer(seed: int, request_id: str, stream: str) -> int:
    """Return a run-independent pseudo-random 64-bit integer."""
    payload = f"{seed}\0{request_id}\0{stream}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _decode_config_sort_key(
    config: LLMConfig, seqlen_range: tuple[int, int]
) -> tuple[object, ...]:
    """Canonicalize a deployed decode group for deterministic sampling."""
    return (
        seqlen_range[1],
        seqlen_range[0],
        config.name,
        config.num_chips,
        *config.get_chip_version_and_parallelism_degree_tuple(),
    )


def select_decode_config_for_output_prediction(
    request: LLMRequest,
    correct_config: LLMConfig,
    config_to_seqlen_range: dict[LLMConfig, tuple[int, int]],
    workload_config: LLMInferenceWorkloadConfig,
) -> LLMConfig:
    """Select a real decode-routing group under output prediction error.

    A correct prediction selects the request's best-fit group. A miss selects
    uniformly from other deployed groups that can hold the request's full
    sequence. The stable request ID and seed make both decisions reproducible
    and keep one routing decision fixed for the request. This deliberately
    models only prediction accuracy; periodic re-prediction and migration are
    outside its scope. If no distinct legal group exists, use the correct group.
    """
    accuracy = workload_config.output_prediction_accuracy
    if accuracy >= 1.0:
        return correct_config

    outcome = _stable_prediction_integer(
        workload_config.output_prediction_seed,
        request.id,
        "prediction-outcome",
    )
    if outcome / (1 << 64) < accuracy:
        return correct_config

    required_total_seqlen = request.input_seqlen + request.output_seqlen
    correct_key = correct_config.get_chip_version_and_parallelism_degree_tuple()
    wrong_configs = [
        (config, seqlen_range)
        for config, seqlen_range in config_to_seqlen_range.items()
        if config.get_chip_version_and_parallelism_degree_tuple() != correct_key
        and seqlen_range[1] >= required_total_seqlen
    ]
    if not wrong_configs:
        return correct_config

    wrong_configs.sort(key=lambda item: _decode_config_sort_key(*item))
    choice = _stable_prediction_integer(
        workload_config.output_prediction_seed,
        request.id,
        "wrong-group-choice",
    )
    return wrong_configs[choice % len(wrong_configs)][0]


class vPodInstance(SimObject):
    """
    Base class for vPod instances.
    This class is used to represent a single vPod in the fleet.
    It can be extended to implement specific vPod types (e.g., prefill, decode).
    """

    def __init__(
        self, simulator: "NPUFleetSimulator", name: str, pod_config: PhysicalCubeConfig
    ):
        super().__init__(name, simulator)
        self.id: str = uuid.uuid4().hex
        """Unique identifier for the vPod instance."""

        self.pod_config: PhysicalCubeConfig = pod_config

        # lifecycle tracking
        self.create_timestamp: int = self.simulator.timestamp
        self.destroy_timestamp: int = -1

    @property
    def destroyed(self) -> bool:
        return self.destroy_timestamp != -1

    def num_chips(self) -> int:
        return self.pod_config.num_chips

    def save_to_checkpoint(self, checkpoint_path: str):
        # remove the simulator attribute since it cannot be pickled
        # and restore it back after we pickled other attributes
        simulator = self.simulator
        self.simulator = None  # type: ignore
        with gzip.open(checkpoint_path, "wb") as f:
            pickle.dump(self, f)
        self.simulator = simulator

    def load_from_checkpoint(self, checkpoint_path: str):
        with gzip.open(checkpoint_path, "rb") as f:
            obj = pickle.load(f)
            simulator = self.simulator
            self.__dict__.update(obj.__dict__)  # will load a None to self.simulator,
            self.simulator = simulator  # so we need to save-and-restore it


class LLMInferenceInstanceBase(vPodInstance):
    """
    Base class for prefill/decode instances in LLM inference.

    Handles the following events:
        LLMInferenceEngineReadyEvent: mark the instance as ready to process more requests.
    """

    def __init__(
        self,
        simulator: "NPUFleetSimulator",
        name: str,
        config: LLMInferenceWorkloadConfig,
        pod_config: LLMConfig | None = None,
    ):
        super().__init__(
            simulator, name, PhysicalCubeConfig.from_ModelConfig(config.llm_config)
        )
        self.config: LLMInferenceWorkloadConfig = deepcopy(config)
        # Initialize BEFORE update_pod_config: that call sets _config_input/output_seqlen
        # from pod_config, and these defaults must not clobber it afterwards. (Previously
        # they ran after update_pod_config, zeroing the real seqlens -> get_token_budget hit
        # its "request count" fallback -> the dispatch batching collapsed every batch to 1.)
        self._config_input_seqlen: int = 0
        self._config_output_seqlen: int = 0
        if pod_config:
            self.update_pod_config(pod_config)

        self.busy: bool = False
        """Whether the instance is busy. If busy == False, it can accept new requests. (We should probably rename this to `ready`.)"""
        self.head_of_line_request_end_timestamp: int | None = None
        """
        The ongoing event that may be blocking other later requests from being finished.
        Mainly used for tracking pipeline stage executions and simulating head-of-line blocking of a long request
        at the front of the pipeline blocking other later requests.
        When we put a new batch into the pipeline, we compute the finish time of this batch as follows:

            if self.head_of_line_request_end_timestamp is not None:
                batch_end_time = the regular batch end time with no HOL blocking
                finish_time = max(self.head_of_line_request_end_timestamp, batch_end_time)
            else:
                batch_end_time = just the regular batch end time
                self.head_of_line_request_end_timestamp = this batch's end event timestamp
        """
        self.under_creation: bool = False
        """Indicates whether this vPod instance is still initializing. Used for horizontal scaling."""
        self.pending_destroy: bool = False
        """Whether this vPod is pending destruction. If True, should not accept new requests."""
        self.pending_or_under_reconfiguration: bool = False
        """Whether this vPod is pending or currently under reconfiguration. If True, should not accept new requests."""

        self.reconfiguration_trace: list[tuple[int, int, LLMConfig]] = []
        """
        Trace of reconfigurations. Records the start, end timestamp and the new config for each reconfiguration event.
        [(start, end, config), ...]
        """
        self.num_pending_requests: int = 0
        """Number of requests currently being processed by this instance."""

    def initialize(self):
        super().initialize()

        # self.simulator.add_event_listener(
        #     LLMInferenceEngineReadyEvent.get_worker_id_listener(
        #         self.on_instance_ready,
        #         self.id,
        #         priority=10,
        #         metadata={"type": "LLMInferenceBase_worker_id_listener"},
        #     )
        # )
        # self.simulator.add_event_listener(
        #     vPodCreateEvent.get_worker_id_listener(
        #         self.on_instance_create,
        #         self.id,
        #         priority=10,
        #         metadata={"type": "LLMInferenceBase_worker_id_listener"},
        #     )
        # )
        # self.simulator.add_event_listener(
        #     vPodDestroyEvent.get_worker_id_listener(
        #         self.on_instance_destroy,
        #         self.id,
        #         priority=10,
        #         metadata={"type": "LLMInferenceBase_worker_id_listener"},
        #     )
        # )
        # self.simulator.add_event_listener(
        #     vPodReconfigurationStartEvent.get_worker_id_listener(
        #         self.on_instance_reconfiguration_start,
        #         self.id,
        #         priority=20,
        #         metadata={"type": "LLMInferenceBase_worker_id_listener"},
        #     )
        # )
        # self.simulator.add_event_listener(
        #     vPodReconfigurationEndEvent.get_worker_id_listener(
        #         self.on_instance_reconfiguration_end,
        #         self.id,
        #         priority=20,
        #         metadata={"type": "LLMInferenceBase_worker_id_listener"},
        #     )
        # )

    @property
    def ready(self):
        """Indicates whether this vPod instance is ready to accept new requests."""
        return (
            not self.busy
            and not self.under_creation
            and not self.pending_destroy
            and not self.pending_or_under_reconfiguration
        )

    def num_chips(self) -> int:
        return self.config.llm_config.num_chips

    def get_max_batch_size(self) -> int:
        """
        Get the maximum batch size that this instance can handle (w/o violating SLO).
        """
        return self.config.llm_config.microbatch_size_ici

    def get_token_budget(self, prefill_or_decode: str) -> int:
        """
        Get the token budget for this vPod.
        token_budget = microbatch_size_ici * config_seqlen
        For prefill: based on input_seqlen
        For decode: based on input_seqlen + output_seqlen (total_seqlen)
        Falls back to microbatch_size_ici if config seqlens were never set.
        """
        batch_size = self.config.llm_config.microbatch_size_ici
        if self._config_input_seqlen == 0:
            return batch_size  # fallback: treat as request count
        if prefill_or_decode == "prefill":
            return batch_size * self._config_input_seqlen
        else:
            return batch_size * (self._config_input_seqlen + self._config_output_seqlen)

    def num_pipeline_stages(self) -> int:
        """
        Number of pipeline parallelism stages.
        """
        return self.config.llm_config.pipeline_parallelism_degree

    def on_instance_ready(self, event: LLMInferenceEngineReadyEvent):
        """
        This method is called when the instance is ready to process more requests.
        It simply sets the instance as not busy.
        """
        self.busy = False
        logging.debug(
            "Instance %s is ready to process more requests at %d",
            self.id,
            event.timestamp,
        )

    def on_instance_create(self, event: vPodCreateEvent):
        """
        This method is called when a new instance is created.
        It simply sets the instance as ready for serving requests.
        """
        self.under_creation = False
        self.busy = False
        self.create_timestamp = event.timestamp
        logging.debug(
            "Instance %s is created and ready at %d", self.id, event.timestamp
        )

    def on_instance_destroy(self, event: vPodDestroyEvent):
        """
        This method is called when an instance is destroyed.
        """
        # Add any bookkeeping or wind down operations here
        self.destroy_timestamp = event.timestamp

        # def criteria(listener: EventListener) -> bool:
        #     return (
        #         listener.metadata.get("type", "")
        #         == "LLMInferenceBase_worker_id_listener"
        #         and listener.metadata.get("worker_id") == self.id
        #     )

        # # remove event listeners related to this instance
        # self.simulator.remove_event_listeners_by_criteria(criteria)

        logging.debug("Instance %s is destroyed at %d", self.id, event.timestamp)

    def update_pod_config(self, pod_config: LLMConfig):
        """
        Update the LLM engine config for this instance,
        including the num chips, parallelism config, max batch size.
        """
        self.config.llm_config = deepcopy(pod_config)
        self._config_input_seqlen = pod_config.input_seqlen
        self._config_output_seqlen = pod_config.output_seqlen
        self.config.llm_config.input_seqlen = 32  # dummy value
        self.config.llm_config.output_seqlen = 32  # dummy value

    def on_instance_reconfiguration_start(self, event: vPodReconfigurationStartEvent):
        """
        This method is called when a vPod reconfiguration starts.
        It will mark the current instance as under reconfiguration and stops serving more requests.
        Also updates the pod config.
        It will schedule the vPodReconfigurationEndEvent.
        """
        self.pending_or_under_reconfiguration = True
        target_config = event.target_config
        assert isinstance(target_config, LLMConfig)

        # Track chip changes: deallocate old chips, and for same-type scale-down
        # also allocate new chips (since VS didn't track them at decision time)
        old_npu_type = self.config.llm_config.name
        old_num_chips = self.config.llm_config.num_chips
        is_same_type_scale_down = (
            target_config.name == old_npu_type
            and target_config.num_chips <= old_num_chips
        )
        self.simulator.cluster_manager.track_deallocate(old_npu_type, old_num_chips)  # type: ignore
        if is_same_type_scale_down:
            self.simulator.cluster_manager.track_allocate(  # type: ignore
                target_config.name, target_config.num_chips
            )

        self.update_pod_config(target_config)
        self.reconfiguration_trace.append(
            (event.timestamp, -1, deepcopy(target_config))
        )
        end_timestamp = self.simulator.timestamp + event.delay_ns
        reconfiguration_end_event = vPodReconfigurationEndEvent(end_timestamp, self.id)
        self.simulator.put(reconfiguration_end_event)

        if logging.level_debug():
            logging.debug(
                "Instance %s is reconfiguring from (num_chips=%d, max_batch_size=%d) to (num_chips=%d, max_batch_size=%d). Reconfiguration delay = %d ns",
                self.id,
                self.config.llm_config.num_chips,
                self.config.llm_config.microbatch_size_ici,
                target_config.num_chips,
                target_config.microbatch_size_ici,
                event.delay_ns,
            )

    def on_instance_reconfiguration_end(self, event: vPodReconfigurationEndEvent):
        """
        This method is called when a vPod reconfiguration ends.
        It will mark the current instance as not under reconfiguration and resumes serving requests.
        """
        self.pending_or_under_reconfiguration = False
        self.reconfiguration_trace[-1] = (
            self.reconfiguration_trace[-1][0],
            event.timestamp,
            self.reconfiguration_trace[-1][2],
        )
        logging.debug(
            "Instance %s has finished reconfiguring to %s",
            self.id,
            self.config.llm_config,
        )


class LLMInferencePrefillInstance(LLMInferenceInstanceBase):
    """
    LLM Inference Prefill Engine.

    Handles the following events:
        LLMInferencePrefillStartEvent: process the prefill requests in a batch; mark the vPod as busy.
        LLMInferencePrefillEndEvent: update request stats.
        vPodReconfigurationStartEvent: trigger a vPod reconfiguration event.
        vPodReconfigurationEndEvent: process the vPod reconfiguration event.
    Fires the following events:
        LLMInferencePrefillEndEvent: when the prefill requests are finished.
        vPodReconfigurationEndEvent: when the vPod reconfiguration is finished.
    """

    def __init__(
        self,
        simulator: "NPUFleetSimulator",
        name: str,
        config: LLMInferenceWorkloadConfig,
        pod_config: LLMConfig | None = None,
    ):
        super().__init__(simulator, name, config, pod_config)

        ## stats
        self.num_processed_tokens: int = 0
        """Total number of tokens processed by this instance."""

        logging.debug(
            "Created LLMInferencePrefillInstance %s with config: %s",
            self.id,
            self.config,
        )

    def initialize(self):
        super().initialize()

        # register event listeners
        # self.simulator.add_event_listener(
        #     LLMInferencePrefillStartEvent.get_worker_id_listener(
        #         self.execute_batch_requests,
        #         self.id,
        #         priority=10,
        #         metadata={"type": "LLMInferencePrefill_worker_id_listener"},
        #     )
        # )
        # self.simulator.add_event_listener(
        #     LLMInferencePrefillEndEvent.get_worker_id_listener(
        #         self.on_prefill_end,
        #         self.id,
        #         priority=10,
        #         metadata={"type": "LLMInferencePrefill_worker_id_listener"},
        #     )
        # )

    def execute_batch_requests(self, event: LLMInferencePrefillStartEvent):
        requests = event.requests
        temp_requests = list(requests)
        # if self.config.use_ideal_batch_size:
        #     # extend temp_requests to the max batch size with copies of the last request
        #     if len(temp_requests) < self.config.llm_config.microbatch_size_ici:
        #         temp_requests.extend(
        #             [copy(temp_requests[-1])] * (
        #                 self.config.llm_config.microbatch_size_ici
        #                 - len(temp_requests)
        #             )
        #         )
        batch_size = len(temp_requests)
        prefill_ops, decode_ops = sim_backend.run_inference_request_batch(
            self.config.llm_config,
            temp_requests,
            self.simulator.config.workload_config.input_seqlen_padding_factors,
            self.simulator.config.workload_config.input_seqlen_padding_steps,
            self.simulator.config.workload_config.output_seqlen_padding_factors,
            self.simulator.config.workload_config.output_seqlen_padding_steps,
        )
        prefill_time_ns_per_stage = sum(
            op.stats.execution_time_ns * op.stats.count for op in prefill_ops
        )
        prefill_time_ns = prefill_time_ns_per_stage * self.num_pipeline_stages()
        prefill_energy_J = (
            sum(op.stats.total_energy_J * op.stats.count for op in prefill_ops)
            * self.config.llm_config.num_chips
        )
        prefill_cost_dollars = cost_model.pipeline_batch_monetary_cost_dollars(
            prefill_ops, self.config.llm_config
        )
        logging.debug(
            "Prefill batch requests (batch size = %d) will take %d ns",
            len(requests),
            prefill_time_ns,
        )
        self.busy = True

        prefill_start_time = event.timestamp
        if self.num_pipeline_stages() > 1 and self.head_of_line_request_end_timestamp:
            end_event = LLMInferencePrefillEndEvent(
                requests,
                max(
                    event.timestamp + prefill_time_ns,
                    self.head_of_line_request_end_timestamp + prefill_time_ns_per_stage,
                ),
                self.id,
            )
            if (
                end_event.timestamp
                == self.head_of_line_request_end_timestamp + prefill_time_ns_per_stage
            ):
                logging.debug(
                    "event %s is HOL blocked by event at timestamp %d. Setting end time to %d",
                    end_event,
                    self.head_of_line_request_end_timestamp,
                    end_event.timestamp,
                )
            if end_event.timestamp >= self.head_of_line_request_end_timestamp:
                # if the end event is after the head of line request end event, we set the new HOL request end event
                self.head_of_line_request_end_timestamp = end_event.timestamp
                logging.debug(
                    "Setting new head of line request end event to %s at %d",
                    end_event,
                    end_event.timestamp,
                )
        else:
            end_event = LLMInferencePrefillEndEvent(
                requests,
                event.timestamp + prefill_time_ns,
                self.id,
            )
            self.head_of_line_request_end_timestamp = end_event.timestamp
            logging.debug(
                "Setting new head of line request end event to %s at %d",
                end_event,
                end_event.timestamp,
            )

        for request in requests:
            request.mark_prefill_started(prefill_start_time)
            logging.debug(
                "Prefill request %s will start at %d", request.id, prefill_start_time
            )
            request.ideal_TTFT_ns = prefill_time_ns
            request.prefill_energy_J = prefill_energy_J / batch_size
            request.prefill_cost_dollars = prefill_cost_dollars / batch_size
            request.config_prefill_npu_type = self.config.llm_config.name
            # request.config_prefill_input_seqlen = self.config.llm_config.input_seqlen  # unused for now
            request.config_prefill_num_chips = self.config.llm_config.num_chips
            request.config_prefill_batch_size = batch_size
            request.config_prefill_pcfg = util.get_pstr(self.config.llm_config)

        self.simulator.put(end_event)

        # if self.num_pipeline_stages() > 1:
        # if we have pipeline parallelism, we need to add LLMInferenceEngineReadyEvent
        ready_event = LLMInferenceEngineReadyEvent(
            event.timestamp + prefill_time_ns_per_stage,
            self.id,
        )
        self.simulator.put(ready_event)

    def on_prefill_end(self, event: LLMInferencePrefillEndEvent):
        # self.busy = False
        # mark the requests as finished
        for request in event.requests:
            request.mark_prefill_finished(event.timestamp)
            self.num_processed_tokens += request.input_seqlen
            logging.debug(
                "Prefill request %s finished at %d. Prefill time = %d ns",
                request.id,
                event.timestamp,
                request.prefill_latency_ns(),
            )
        self.num_pending_requests -= len(event.requests)

        # check if this instance is already destroyed
        # if so, clean up the event listeners in simulator
        if self.destroyed and self.num_pending_requests == 0:

            def criteria(listener: EventListener) -> bool:
                return (
                    listener.metadata.get("type", "")
                    == "LLMInferencePrefill_worker_id_listener"
                    and listener.metadata.get("worker_id") == self.id
                )

            # remove event listeners related to this instance
            self.simulator.remove_event_listeners_by_criteria(criteria)
            logging.debug(
                "Instance %s has been destroyed and has no ongoing requests. Cleaning up event listeners.",
                self.id,
            )


class LLMInferenceDecodeInstance(LLMInferenceInstanceBase):
    """
    LLM Inference Decode Engine.

    Handles the following events:
        LLMInferenceDecodeIterationStartEvent: process the decode requests in a batch; mark the vPod as busy.
        LLMInferenceDecodeIterationEndEvent: mark the vPod as idle.
        vPodReconfigurationStartEvent: trigger a vPod reconfiguration event.
        vPodReconfigurationEndEvent: process the vPod reconfiguration event.
    Fires the following events:
        LLMInferenceDecodeIterationEndEvent: when the decode requests are finished.
        vPodReconfigurationEndEvent: when the vPod reconfiguration is finished.
    """

    def __init__(
        self,
        simulator: "NPUFleetSimulator",
        name: str,
        config: LLMInferenceWorkloadConfig,
        pod_config: LLMConfig | None = None,
    ):
        super().__init__(simulator, name, config, pod_config)

        ## stats
        self.num_processed_tokens: int = 0
        """Total number of tokens generated by this instance."""

        logging.debug(
            "Created LLMInferenceDecodeInstance %s with config: %s",
            self.id,
            self.config,
        )

    def initialize(self):
        super().initialize()

        # register event listeners
        # self.simulator.add_event_listener(
        #     LLMInferenceDecodeIterationStartEvent.get_worker_id_listener(
        #         self.execute_batch_requests,
        #         self.id,
        #         priority=10,
        #         metadata={"type": "LLMInferenceDecode_worker_id_listener"},
        #     )
        # )
        # self.simulator.add_event_listener(
        #     LLMInferenceDecodeIterationEndEvent.get_worker_id_listener(
        #         self.on_decode_iteration_end,
        #         self.id,
        #         priority=10,
        #         metadata={"type": "LLMInferenceDecode_worker_id_listener"},
        #     )
        # )

    def execute_batch_requests(self, event: LLMInferenceDecodeIterationStartEvent):
        requests = event.requests
        temp_requests = list(requests)
        # if self.config.use_ideal_batch_size:
        #     # extend temp_requests to the max batch size with copies of the last request
        #     if len(temp_requests) < self.config.llm_config.microbatch_size_ici:
        #         temp_requests.extend(
        #             [copy(temp_requests[-1])] * (
        #                 self.config.llm_config.microbatch_size_ici
        #                 - len(temp_requests)
        #             )
        #         )
        batch_size = len(temp_requests)
        prefill_ops, decode_ops = sim_backend.run_inference_request_batch(
            self.config.llm_config,
            temp_requests,
            self.simulator.config.workload_config.input_seqlen_padding_factors,
            self.simulator.config.workload_config.input_seqlen_padding_steps,
            self.simulator.config.workload_config.output_seqlen_padding_factors,
            self.simulator.config.workload_config.output_seqlen_padding_steps,
        )
        decode_time_ns_per_stage = sum(
            op.stats.execution_time_ns * op.stats.count for op in decode_ops
        )
        decode_time_ns = decode_time_ns_per_stage * self.num_pipeline_stages()
        decode_energy_per_token_J = (
            sum(op.stats.total_energy_J * op.stats.count for op in decode_ops)
            * self.config.llm_config.num_chips
        )
        decode_cost_dollars = cost_model.pipeline_batch_monetary_cost_dollars(
            decode_ops, self.config.llm_config
        )
        logging.debug(
            "Decode batch requests (batch size = %d) will take %d ns",
            len(requests),
            decode_time_ns,
        )
        self.busy = True

        decode_iteration_start_time = event.timestamp
        if self.num_pipeline_stages() > 1 and self.head_of_line_request_end_timestamp:
            end_event = LLMInferenceDecodeIterationEndEvent(
                requests,
                max(
                    event.timestamp + decode_time_ns * event.num_iterations,
                    self.head_of_line_request_end_timestamp + decode_time_ns_per_stage,
                ),
                self.id,
                event.num_iterations,
            )
            if (
                end_event.timestamp
                == self.head_of_line_request_end_timestamp + decode_time_ns_per_stage
            ):
                logging.debug(
                    "event %s is HOL blocked by event at timestamp %d. Setting end time to %d",
                    end_event,
                    self.head_of_line_request_end_timestamp,
                    end_event.timestamp,
                )
            if end_event.timestamp >= self.head_of_line_request_end_timestamp:
                # if the end event is after the head of line request end event, we set the new HOL request end event
                self.head_of_line_request_end_timestamp = end_event.timestamp
                logging.debug(
                    "Setting new head of line request end event to %s at %d",
                    end_event,
                    end_event.timestamp,
                )
        else:
            end_event = LLMInferenceDecodeIterationEndEvent(
                requests,
                event.timestamp + decode_time_ns * event.num_iterations,
                self.id,
                event.num_iterations,
            )
            self.head_of_line_request_end_timestamp = end_event.timestamp
            logging.debug(
                "Setting new head of line request end event to %s at %d",
                end_event,
                end_event.timestamp,
            )

        self.simulator.put(end_event)

        for request in requests:
            request.mark_decode_iteration_started(decode_iteration_start_time)
            logging.debug(
                "Decode iteration %s will start at %d",
                request.id,
                decode_iteration_start_time,
            )
            request.ideal_TPOT_ns = decode_time_ns
            request.decode_energy_per_token_J += [
                decode_energy_per_token_J / batch_size
            ] * event.num_iterations
            request.decode_cost_per_token_dollars += [
                decode_cost_dollars / batch_size
            ] * event.num_iterations
            # request.config_decode_input_seqlens += [
            #     self.config.llm_config.input_seqlen
            # ] * event.num_iterations  # unused for now
            # request.config_decode_output_seqlens += [
            #     self.config.llm_config.output_seqlen
            # ] * event.num_iterations
            request.config_decode_batch_sizes += [batch_size] * event.num_iterations
            request.config_decode_npu_types += [
                self.config.llm_config.name
            ] * event.num_iterations
            request.config_decode_num_chips += [
                self.config.llm_config.num_chips
            ] * event.num_iterations
            request.config_decode_pcfgs += [
                util.get_pstr(self.config.llm_config)
            ] * event.num_iterations

        # if self.num_pipeline_stages() > 1:
        # if we have pipeline parallelism, we need to add LLMInferenceEngineReadyEvent
        ready_event = LLMInferenceEngineReadyEvent(
            event.timestamp + decode_time_ns_per_stage,
            self.id,
        )
        self.simulator.put(ready_event)

    def on_decode_iteration_end(self, event: LLMInferenceDecodeIterationEndEvent):
        # self.busy = False
        # mark the requests as finished
        for request in event.requests:
            request.mark_decode_iteration_finished(
                event.timestamp, event.num_iterations
            )
            logging.debug(
                "Decode iteration %s finished at %d. Decode iteration = %d. Num iter per step = %d",
                request.id,
                event.timestamp,
                request.current_decode_step,
                event.num_iterations,
            )
        self.num_processed_tokens += len(event.requests) * event.num_iterations
        self.num_pending_requests -= len(event.requests)

        # check if this instance is already destroyed
        # if so, clean up the event listeners in simulator
        if self.destroyed and self.num_pending_requests == 0:

            def criteria(listener: EventListener) -> bool:
                return (
                    listener.metadata.get("type", "")
                    == "LLMInferenceDecode_worker_id_listener"
                    and listener.metadata.get("worker_id") == self.id
                )

            # remove event listeners related to this instance
            self.simulator.remove_event_listeners_by_criteria(criteria)
            logging.debug(
                "Instance %s has been destroyed and has no ongoing requests. Cleaning up event listeners.",
                self.id,
            )


class LLMInferenceEndpoint(SimObject):
    """
    LLM Inference Endpoint.

    Handles the following events:
        LLMInferenceRequestEnqueueEvent: enqueue the request to the prefill queue; try to dispatch prefill requests to idle prefill vPods.
        LLMInferencePrefillEndEvent: mark the prefill vPod as idle; try to dispatch more prefill requests; enqueue decode requests to the decode queue; try to dispatch decode requests to idle decode vPods.
        LLMInferenceDecodeIterationEndEvent: mark the decode vPod as idle; if the request is not finished, enqueue the next decode iteration to the decode queue; try to dispatch decode requests to idle decode vPods.
    Fires the following events:
        LLMInferencePrefillStartEvent: when a prefill request is dispatched to a prefill vPod.
        LLMInferenceDecodeIterationStartEvent: when a decode iteration is dispatched to a decode vPod.
    """

    def __init__(self, name: str, simulator: "NPUFleetSimulator"):
        super().__init__(name, simulator)

        self.prefill_vpods: dict[str, LLMInferencePrefillInstance] = {}
        self.decode_vpods: dict[str, LLMInferenceDecodeInstance] = {}

        # used to track destroyed vPods
        self.destroyed_prefill_vpods: dict[str, LLMInferencePrefillInstance] = {}
        self.destroyed_decode_vpods: dict[str, LLMInferenceDecodeInstance] = {}

        # config → {vpod_id → vpod} for O(1) config-based lookups
        # Maintained alongside prefill_vpods/decode_vpods on add/remove.
        # Safe because vpod configs never change in-place (reconfiguration is unused).
        self._prefill_vpods_by_config: dict[
            LLMConfig, dict[str, LLMInferencePrefillInstance]
        ] = {}
        self._decode_vpods_by_config: dict[
            LLMConfig, dict[str, LLMInferenceDecodeInstance]
        ] = {}

        AutoScalerType = vPodAutoScaler.get_autoscaler_by_name(
            self.simulator.config.workload_config.autoscaler_type
        )
        self.prefill_autoscaler: vPodAutoScaler.vPodAutoScaler = AutoScalerType(
            simulator,
            f"{name}_prefill_autoscaler",
            prefill_or_decode="prefill",
            metrics_server=self.simulator.metrics_server,
        )
        self.decode_autoscaler: vPodAutoScaler.vPodAutoScaler = AutoScalerType(
            simulator,
            f"{name}_decode_autoscaler",
            prefill_or_decode="decode",
            metrics_server=self.simulator.metrics_server,
        )

        self.prefill_request_queue: deque[LLMRequest] = deque()
        """Prefill request queue. append() to enqueue, popleft() to dequeue."""
        self.decode_request_queue: deque[LLMRequest] = deque()
        """Decode request queue. append() to enqueue, popleft() to dequeue."""

        # TODO: parametrize these in configs
        self.prefill_schedule_delay_ns = 0
        self.decode_schedule_delay_ns = 0

    def _register_vpod_config(self, vpod: LLMInferenceInstanceBase):
        """Add vpod to the config-grouped secondary index."""
        by_config = (
            self._prefill_vpods_by_config
            if isinstance(vpod, LLMInferencePrefillInstance)
            else self._decode_vpods_by_config
        )
        cfg = vpod.config.llm_config
        if cfg not in by_config:
            by_config[cfg] = {}
        by_config[cfg][vpod.id] = vpod  # type: ignore

    def _unregister_vpod_config(self, vpod: LLMInferenceInstanceBase):
        """Remove vpod from the config-grouped secondary index."""
        by_config = (
            self._prefill_vpods_by_config
            if isinstance(vpod, LLMInferencePrefillInstance)
            else self._decode_vpods_by_config
        )
        cfg = vpod.config.llm_config
        if cfg in by_config:
            by_config[cfg].pop(vpod.id, None)
            if not by_config[cfg]:
                del by_config[cfg]

    def _select_initial_config_with_capacity(
        self, candidate_configs: "list[LLMConfig]"
    ) -> "LLMConfig | None":
        """
        Pick the most-preferred candidate config (candidates are ordered best-efficiency
        first) whose NPU version still has free capacity under max_chips_per_version.
        Capacity is checked against the cluster manager's running _allocated_chips, which
        track_allocate updates as each initial vPod is placed -- so this fills the best
        version first and spills onto the next-best version once it is full.

        Returns None if no candidate version has capacity, i.e. the physical per-version
        caps are exhausted. The caller then skips this vPod, matching the dynamic-scaling
        path where allocate() returns False once a version is full (the caps are hard).
        """
        cluster_manager = self.simulator.cluster_manager
        for cfg in candidate_configs:
            assert isinstance(cfg, LLMConfig)
            if cluster_manager._has_capacity(cfg.name, cfg.num_chips):  # type: ignore
                return cfg
        return None

    def initialize(self):
        super().initialize()

        # Use the version-ordered alternatives so the initial fleet is capacity-aware:
        # each vPod is placed on its most-preferred (most-efficient) version that still has
        # room under max_chips_per_version, spilling onto the next-best version once the
        # best one is full. Prefill is placed before decode and both share the cluster
        # manager's _allocated_chips, so the two phases coordinate against the same caps.
        prefill_allocations = (
            self.prefill_autoscaler.get_initial_allocation_alternatives()
        )
        decode_allocations = (
            self.decode_autoscaler.get_initial_allocation_alternatives()
        )
        prefill_skipped = 0
        for num_prefill_vpods, prefill_vpod_configs in prefill_allocations:
            for i in range(num_prefill_vpods):
                prefill_vpod_config = self._select_initial_config_with_capacity(
                    prefill_vpod_configs
                )
                if prefill_vpod_config is None:
                    prefill_skipped += 1
                    continue
                prefill_vpod = LLMInferencePrefillInstance(
                    self.simulator,
                    f"{self.name}_prefill_{i}",
                    self.simulator.config.workload_config,
                    prefill_vpod_config,
                )
                prefill_vpod.initialize()
                self.prefill_vpods[prefill_vpod.id] = prefill_vpod
                self._register_vpod_config(prefill_vpod)
                self.simulator.cluster_manager.track_allocate(  # type: ignore
                    prefill_vpod_config.name, prefill_vpod_config.num_chips
                )
        decode_skipped = 0
        for num_decode_vpods, decode_vpod_configs in decode_allocations:
            for i in range(num_decode_vpods):
                decode_vpod_config = self._select_initial_config_with_capacity(
                    decode_vpod_configs
                )
                if decode_vpod_config is None:
                    decode_skipped += 1
                    continue
                decode_vpod = LLMInferenceDecodeInstance(
                    self.simulator,
                    f"{self.name}_decode_{i}",
                    self.simulator.config.workload_config,
                    decode_vpod_config,
                )
                decode_vpod.initialize()
                self.decode_vpods[decode_vpod.id] = decode_vpod
                self._register_vpod_config(decode_vpod)
                self.simulator.cluster_manager.track_allocate(  # type: ignore
                    decode_vpod_config.name, decode_vpod_config.num_chips
                )
        if prefill_skipped or decode_skipped:
            logging.warning(
                "(%s) Initial allocation hit per-version caps: skipped %d prefill + %d "
                "decode vPod(s) (no NPU version with free capacity). allocated=%s caps=%s",
                self.name,
                prefill_skipped,
                decode_skipped,
                self.simulator.cluster_manager.get_allocated_chips(),  # type: ignore
                self.simulator.cluster_manager.scheduler_config.max_chips_per_version,  # type: ignore
            )

        ### BEGIN: Central event listener dispatchers for vPod instances.
        # We use central dispatchers to avoid registering too many event listeners in the simulator
        # which can slow down the event processing.
        # base LLM inference vpod events
        self.simulator.add_event_listener(
            LLMInferenceEngineReadyEvent.get_type_listener(
                self.on_instance_ready,
                priority=10,
                # metadata={"type": "LLMInferenceBase_worker_id_listener"},
            )
        )
        self.simulator.add_event_listener(
            vPodCreateEvent.get_type_listener(
                self.on_instance_create,
                priority=10,
                # metadata={"type": "LLMInferenceBase_worker_id_listener"},
            )
        )
        # already added below
        # self.simulator.add_event_listener(
        #     vPodDestroyEvent.get_type_listener(
        #         self.on_instance_destroy,
        #         priority=10,
        #         # metadata={"type": "LLMInferenceBase_worker_id_listener"},
        #     )
        # )
        self.simulator.add_event_listener(
            vPodReconfigurationStartEvent.get_type_listener(
                self.on_instance_reconfiguration_start,
                priority=20,
                # metadata={"type": "LLMInferenceBase_worker_id_listener"},
            )
        )
        self.simulator.add_event_listener(
            vPodReconfigurationEndEvent.get_type_listener(
                self.on_instance_reconfiguration_end,
                priority=20,
                # metadata={"type": "LLMInferenceBase_worker_id_listener"},
            )
        )
        # prefill vpod events
        self.simulator.add_event_listener(
            LLMInferencePrefillStartEvent.get_type_listener(
                self.on_prefill_start,
                priority=10,
                # metadata={"type": "LLMInferencePrefill_worker_id_listener"},
            )
        )
        self.simulator.add_event_listener(
            LLMInferencePrefillEndEvent.get_type_listener(
                self.on_prefill_end,
                priority=10,
                # metadata={"type": "LLMInferencePrefill_worker_id_listener"},
            )
        )
        # decode vpod events
        self.simulator.add_event_listener(
            LLMInferenceDecodeIterationStartEvent.get_type_listener(
                self.on_decode_iteration_start,
                # self.id,
                priority=10,
                # metadata={"type": "LLMInferenceDecode_worker_id_listener"},
            )
        )
        self.simulator.add_event_listener(
            LLMInferenceDecodeIterationEndEvent.get_type_listener(
                self.on_decode_iteration_end,
                # self.id,
                priority=10,
                # metadata={"type": "LLMInferenceDecode_worker_id_listener"},
            )
        )
        ### END: Central event listener dispatchers for vPod instances.

        # register event listener for scaling events
        if self.prefill_autoscaler.enable_mpa:
            # use MPA for NeuScale / MultiPool
            self.simulator.add_event_listener(
                vPodMultiDimensionalScalingEvent.get_type_listener(
                    self.on_vpod_mpa_scaling
                )
            )
        else:
            self.simulator.add_event_listener(
                vPodHorizontalScalingEvent.get_type_listener(
                    self.on_vpod_horizontal_scaling
                )
            )
            self.simulator.add_event_listener(
                vPodVerticalScalingEvent.get_type_listener(
                    self.on_vpod_vertical_scaling
                )
            )

        # register event listener for request enqueue events
        self.simulator.add_event_listener(
            LLMInferenceRequestEnqueueEvent.get_type_listener(
                self.enqueue_prefill_request,
                priority=40,
            )
        )

        ### The following event listeners must be registered after the vPods have added their own event listeners.
        # inference engine ready event listener
        self.simulator.add_event_listener(
            LLMInferenceEngineReadyEvent.get_type_listener(self.on_engine_ready)
        )
        # instance destroy event listener
        self.simulator.add_event_listener(
            vPodDestroyEvent.get_type_listener(self.on_instance_destroy)
        )
        # instance create and ready event listener
        self.simulator.add_event_listener(
            vPodCreateEvent.get_type_listener(self.on_engine_ready)
        )
        self.simulator.add_event_listener(
            vPodReconfigurationEndEvent.get_type_listener(self.on_engine_ready)
        )
        # register event listener for prefill end events
        # so that we can dispatch decode requests
        self.simulator.add_event_listener(
            LLMInferencePrefillEndEvent.get_type_listener(
                self.prefill_end_and_enqueue_decode_request,
                priority=40,
            )
        )
        # register event listener for decode iteration end events
        # so that we can dispatch next decode iteration or finish the request
        self.simulator.add_event_listener(
            LLMInferenceDecodeIterationEndEvent.get_type_listener(
                self.decode_iteration_end_and_enqueue_next_iteration,
                priority=40,
            )
        )
        # periodic auto scaling event
        if self.prefill_autoscaler.enable_mpa:
            # use MPA for NeuScale / MultiPool
            self.simulator.add_event_listener(
                vPodMultiDimensionalScalingRecommendationEvent.get_type_listener(
                    self.on_mpa_scaling_recommendation
                )
            )
        else:
            self.simulator.add_event_listener(
                vPodHorizontalScalingRecommendationEvent.get_type_listener(
                    self.on_horizontal_scaling_recommendation
                )
            )
            self.simulator.add_event_listener(
                vPodVerticalScalingRecommendationEvent.get_type_listener(
                    self.on_vertical_scaling_recommendation
                )
            )

    @property
    def num_prefill_replicas(self) -> int:
        return len(self.prefill_vpods)

    @property
    def num_decode_replicas(self) -> int:
        return len(self.decode_vpods)

    def lookup_vpod_by_id(
        self, vpod_id: str, only_live: bool = False
    ) -> LLMInferenceInstanceBase | None:
        """
        Lookup vPod instance by its id.
        If only_live is True, only look up in live vPods.
        """

        if vpod_id in self.prefill_vpods:
            return self.prefill_vpods[vpod_id]
        elif vpod_id in self.decode_vpods:
            return self.decode_vpods[vpod_id]

        if only_live:
            return None

        if vpod_id in self.destroyed_prefill_vpods:
            return self.destroyed_prefill_vpods[vpod_id]
        elif vpod_id in self.destroyed_decode_vpods:
            return self.destroyed_decode_vpods[vpod_id]
        else:
            return None

    def on_instance_ready(self, event: LLMInferenceEngineReadyEvent):
        """
        Central dispatcher for vPod.on_instance_ready.
        """
        vpod_id = event.worker_id
        vpod = self.lookup_vpod_by_id(vpod_id)
        if vpod:
            vpod.on_instance_ready(event)
        else:
            raise ValueError(f"Unknown vPod id {vpod_id} in on_instance_ready event.")

    def on_instance_create(self, event: vPodCreateEvent):
        """
        Central dispatcher for vPod.on_instance_create.
        """
        vpod_id = event.worker_id
        vpod = self.lookup_vpod_by_id(vpod_id)
        if vpod:
            vpod.on_instance_create(event)
        else:
            raise ValueError(f"Unknown vPod id {vpod_id} in on_instance_create event.")

    def on_instance_reconfiguration_start(self, event: vPodReconfigurationStartEvent):
        """
        Central dispatcher for vPod.on_instance_reconfiguration_start.
        """
        vpod_id = event.worker_id
        vpod = self.lookup_vpod_by_id(vpod_id)
        if vpod:
            vpod.on_instance_reconfiguration_start(event)
        else:
            raise ValueError(
                f"Unknown vPod id {vpod_id} in on_instance_reconfiguration_start event."
            )

    def on_instance_reconfiguration_end(self, event: vPodReconfigurationEndEvent):
        """
        Central dispatcher for vPod.on_instance_reconfiguration_end.
        """
        vpod_id = event.worker_id
        vpod = self.lookup_vpod_by_id(vpod_id)
        if vpod:
            vpod.on_instance_reconfiguration_end(event)
        else:
            raise ValueError(
                f"Unknown vPod id {vpod_id} in on_instance_reconfiguration_end event."
            )

    def on_prefill_start(self, event: LLMInferencePrefillStartEvent):
        """
        Central dispatcher for vPod.execute_batch_requests for prefill.
        """
        vpod_id = event.worker_id
        vpod = self.lookup_vpod_by_id(vpod_id)
        if vpod and isinstance(vpod, LLMInferencePrefillInstance):
            vpod.execute_batch_requests(event)
        else:
            raise ValueError(
                f"Unknown prefill vPod id {vpod_id} in on_prefill_start event."
            )

    def on_prefill_end(self, event: LLMInferencePrefillEndEvent):
        """
        Central dispatcher for vPod.on_prefill_end.
        """
        vpod_id = event.worker_id
        vpod = self.lookup_vpod_by_id(vpod_id)
        if vpod and isinstance(vpod, LLMInferencePrefillInstance):
            vpod.on_prefill_end(event)
        else:
            raise ValueError(
                f"Unknown prefill vPod id {vpod_id} in on_prefill_end event."
            )

    def on_decode_iteration_start(self, event: LLMInferenceDecodeIterationStartEvent):
        """
        Central dispatcher for vPod.execute_batch_requests for decode.
        """
        vpod_id = event.worker_id
        vpod = self.lookup_vpod_by_id(vpod_id)
        if vpod and isinstance(vpod, LLMInferenceDecodeInstance):
            vpod.execute_batch_requests(event)
        else:
            raise ValueError(
                f"Unknown decode vPod id {vpod_id} in on_decode_iteration_start event."
            )

    def on_decode_iteration_end(self, event: LLMInferenceDecodeIterationEndEvent):
        """
        Central dispatcher for vPod.on_decode_iteration_end.
        """
        vpod_id = event.worker_id
        vpod = self.lookup_vpod_by_id(vpod_id)
        if vpod and isinstance(vpod, LLMInferenceDecodeInstance):
            vpod.on_decode_iteration_end(event)
        else:
            raise ValueError(
                f"Unknown decode vPod id {vpod_id} in on_decode_iteration_end event."
            )

    def enqueue_prefill_request(self, event: LLMInferenceRequestEnqueueEvent):
        request = event.request
        self.prefill_request_queue.append(request)
        logging.debug(
            "Enqueued prefill request: %s. Prefill queue length: %d",
            request,
            len(self.prefill_request_queue),
        )
        self.try_dispatch_prefill_requests(event.timestamp)

    def try_dispatch_prefill_requests(self, timestamp: int):
        """
        Try to dispatch prefill requests from the queue to available prefill vPods.
        This method will add LLMInferencePrefillStartEvent for each dispatched request.
        """
        if len(self.prefill_request_queue) == 0:
            logging.debug("No prefill requests in the queue at %d.", timestamp)
            return

        # if there is an idle prefill vPod, schedule the request immediately
        # sort all vPods by (1) if they are busy, and (2) the earliest head of line request end event timestamp
        # vpods_sorted_by_util = sorted(
        #     [v for v in self.prefill_vpods if v.ready],
        #     key=lambda vpod: vpod.head_of_line_request_end_timestamp or 0,
        # )
        vpods_sorted_by_util = [v for v in self.prefill_vpods.values() if v.ready]
        if len(vpods_sorted_by_util) == 0:
            logging.debug(
                "All prefill vPods are busy. No requests dispatched at %d.", timestamp
            )
            return

        if (
            not self.prefill_autoscaler.enable_mpa
        ):  # if not using MPA, just evenly route requests with batching
            # reorder requests to have similar input_seqlen in the same batch
            # prioritize short requests
            self.prefill_request_queue = deque(
                sorted(
                    self.prefill_request_queue,
                    key=lambda r: r.input_seqlen,
                    # reverse=True,
                )
            )

            self.dispatch_requests_to_vpods(
                timestamp, vpods_sorted_by_util, self.prefill_request_queue, "prefill"
            )
        else:  # if using MPA, route requests based on their seqlens
            self.dispatch_requests_to_vpods_mpa(
                timestamp, vpods_sorted_by_util, "prefill"
            )

    def try_dispatch_decode_requests(self, timestamp: int):
        """
        Try to dispatch decode requests from the queue to available decode vPods.
        This method will add LLMInferenceDecodeIterationStartEvent for each dispatched request.
        """
        if len(self.decode_request_queue) == 0:
            logging.debug("No decode requests in the queue at %d.", timestamp)
            return

        # if there is an idle decode vPod, schedule the request immediately
        # sort all vPods by (1) if they are busy, and (2) the earliest head of line request end event timestamp
        # vpods_sorted_by_util = sorted(
        #     [v for v in self.decode_vpods if v.ready],
        #     key=lambda vpod: vpod.head_of_line_request_end_timestamp or 0,
        # )
        vpods_sorted_by_util = [v for v in self.decode_vpods.values() if v.ready]
        if len(vpods_sorted_by_util) == 0:
            logging.debug(
                "All decode vPods are busy. No requests dispatched at %d.", timestamp
            )
            return

        if (
            not self.prefill_autoscaler.enable_mpa
        ):  # if not using MPA, just evenly route requests with batching
            # reorder requests to have similar total seqlen in the same batch
            # prioritize short requests
            self.decode_request_queue = deque(
                sorted(
                    self.decode_request_queue,
                    key=lambda r: r.total_seqlen,
                    # reverse=True,
                )
            )

            self.dispatch_requests_to_vpods(
                timestamp, vpods_sorted_by_util, self.decode_request_queue, "decode"
            )
        else:  # if using MPA, route requests based on their seqlens
            self.dispatch_requests_to_vpods_mpa(
                timestamp, vpods_sorted_by_util, "decode"
            )

    def dispatch_requests_to_vpods_mpa(
        self,
        timestamp: int,
        vpods_sorted_by_util: Sequence[
            LLMInferencePrefillInstance | LLMInferenceDecodeInstance
        ],
        prefill_or_decode: str,
    ):
        """
        Dispatch prefill requests to prefill vPods based on MPA routing algorithm.
        """
        request_queue = copy(
            self.prefill_request_queue
            if prefill_or_decode == "prefill"
            else self.decode_request_queue
        )
        initial_queue_length = len(request_queue)
        if initial_queue_length == 0:
            logging.debug(
                "No %s requests in the queue at %d. Skip dispatching.",
                prefill_or_decode,
                timestamp,
            )
            return

        # group vpods by configs
        config_to_vpods: dict[
            LLMConfig, list[LLMInferencePrefillInstance | LLMInferenceDecodeInstance]
        ] = {}
        for vpod in vpods_sorted_by_util:
            cfg = vpod.config.llm_config
            # ignore seqlen in the config object
            cfg.input_seqlen = 32
            cfg.output_seqlen = 32
            config_to_vpods.setdefault(cfg, []).append(vpod)

        if logging.level_debug():
            logging.debug(
                "%s request queue seqlens: %s",
                prefill_or_decode,
                [
                    (
                        r.input_seqlen
                        if prefill_or_decode == "prefill"
                        else (r.input_seqlen + r.output_seqlen)
                    )
                    for r in request_queue
                ],
            )
            logging.debug(
                "Routing %s requests to vPod configs: %s",
                prefill_or_decode,
                [
                    (cfg.num_chips, cfg.name, len(vpods))
                    for cfg, vpods in config_to_vpods.items()
                ],
            )

        seqlen_pairs = [
            (
                (r.input_seqlen, 4)
                if prefill_or_decode == "prefill"
                else (r.input_seqlen, r.output_seqlen)
            )
            for r in request_queue
        ]

        autoscaler = (
            self.prefill_autoscaler
            if prefill_or_decode == "prefill"
            else self.decode_autoscaler
        )
        config_to_seqlens_mapping = {}
        for cfg in config_to_vpods.keys():
            seqlens = autoscaler.get_seqlens_for_config(seqlen_pairs, cfg)
            if len(seqlens) == 0:
                if logging.level_debug():
                    logging.debug(
                        "No requests can be routed to config %s. Skipping.",
                        cfg,
                    )
                continue
            config_to_seqlens_mapping[cfg] = seqlens

        if logging.level_debug():
            logging.debug(
                "(%s) (%d) %s: Routing %d requests to %d vPod configs. Config to seqlens mapping: %s",
                self.simulator.name,
                timestamp,
                prefill_or_decode,
                initial_queue_length,
                len(config_to_vpods),
                [
                    ((cfg.name, cfg.num_chips), reqs)
                    for cfg, reqs in config_to_seqlens_mapping.items()
                ],
            )
            logging.debug(
                "(%s) (%d) %s: seq lens in the request queue: %s",
                self.simulator.name,
                timestamp,
                prefill_or_decode,
                seqlen_pairs,
            )

        # Precompute seqlen ranges per config for best-fit fallback
        config_to_seqlen_range: dict[LLMConfig, tuple[int, int]] = {}
        for cfg in config_to_vpods.keys():
            config_to_seqlen_range[cfg] = autoscaler_lib.get_seqlen_range_for_config(
                seqlen_pairs, cfg, self.simulator.config, prefill_or_decode
            )
        # A deployed config with no exact-match request this tick gets range (-1,-1), which
        # find_best_fit_config skips -- stranding orphaned requests (e.g. those whose group
        # was coalesced away into this larger, momentarily-idle group). Fall back to the
        # autoscaler's last-known served-seqlen range (keyed by version+parallelism) so the
        # config stays a valid best-fit target.
        range_fallback = getattr(autoscaler, "config_to_seqlen_range_fallback", {})
        if range_fallback:
            for cfg in config_to_vpods.keys():
                if config_to_seqlen_range.get(cfg, (-1, -1)) == (-1, -1):
                    key = cfg.get_chip_version_and_parallelism_degree_tuple()
                    if key in range_fallback:
                        config_to_seqlen_range[cfg] = range_fallback[key]

        requests_grouped_by_cfg: dict[LLMConfig, deque[LLMRequest]] = {
            cfg: deque() for cfg in config_to_vpods
        }
        remaining_requests: deque[LLMRequest] = deque()
        best_fit_count = 0
        for req in request_queue:
            dispatched = False
            seqlen = (
                (req.input_seqlen, 4)
                if prefill_or_decode == "prefill"
                else (req.input_seqlen, req.output_seqlen)
            )
            if (
                prefill_or_decode == "decode"
                and self.simulator.config.workload_config.output_prediction_accuracy
                < 1.0
            ):
                correct_cfg = next(
                    (
                        cfg
                        for cfg, seqlens in config_to_seqlens_mapping.items()
                        if seqlen in seqlens
                    ),
                    None,
                )
                if correct_cfg is None:
                    correct_cfg = autoscaler_lib.find_best_fit_config(
                        req.input_seqlen + req.output_seqlen,
                        config_to_seqlen_range,
                    )
                if correct_cfg is not None:
                    selected_cfg = select_decode_config_for_output_prediction(
                        req,
                        correct_cfg,
                        config_to_seqlen_range,
                        self.simulator.config.workload_config,
                    )
                    requests_grouped_by_cfg[selected_cfg].append(req)
                    dispatched = True
            if not dispatched:
                for cfg, seqlens in config_to_seqlens_mapping.items():
                    if seqlen in seqlens:
                        requests_grouped_by_cfg[cfg].append(req)
                        dispatched = True
                        break  # Request is routed, move to the next one
            if not dispatched:
                # Best-fit fallback: find the closest-matching vPod config
                eff_seqlen = (
                    req.input_seqlen
                    if prefill_or_decode == "prefill"
                    else (req.input_seqlen + req.output_seqlen)
                )
                best_cfg = autoscaler_lib.find_best_fit_config(
                    eff_seqlen, config_to_seqlen_range
                )
                if best_cfg is not None:
                    requests_grouped_by_cfg[best_cfg].append(req)
                    best_fit_count += 1
                    dispatched = True
            if not dispatched:
                remaining_requests.append(req)

        if best_fit_count > 0 and logging.level_debug():
            logging.debug(
                "(%s) (%d) %s: %d requests routed via best-fit fallback (no exact config match).",
                self.simulator.name,
                timestamp,
                prefill_or_decode,
                best_fit_count,
            )

        if logging.level_debug():
            logging.debug(
                "(%s) (%d) %s: Routing %d requests to %d vPod configs. Requests grouped by config: %s. Remaining requests not routed: %d",
                self.simulator.name,
                timestamp,
                prefill_or_decode,
                initial_queue_length,
                len(config_to_vpods),
                [
                    ((cfg.name, cfg.num_chips), len(reqs))
                    for cfg, reqs in requests_grouped_by_cfg.items()
                ],
                len(remaining_requests),
            )

        # route requests
        for config, vpods in config_to_vpods.items():
            requests_to_dispatch = requests_grouped_by_cfg[config]
            if len(requests_to_dispatch) > 0:
                # Sort requests within this small bucket before dispatching
                key_func = (
                    (lambda r: r.input_seqlen)
                    if prefill_or_decode == "prefill"
                    else (lambda r: r.total_seqlen)
                )
                sorted_requests = deque(sorted(requests_to_dispatch, key=key_func))
                if logging.level_debug():
                    logging.debug(
                        "Routing %s requests to %d vPods with v=%s, num_chips=%d",
                        prefill_or_decode,
                        len(vpods),
                        config.name,
                        config.num_chips,
                    )

                self.dispatch_requests_to_vpods(
                    timestamp, vpods, sorted_requests, prefill_or_decode
                )

                # Add any requests not taken by dispatch_requests_to_vpods back to the remaining queue
                remaining_requests.extend(sorted_requests)

        # update the main queue
        if prefill_or_decode == "prefill":
            self.prefill_request_queue = remaining_requests
        else:
            self.decode_request_queue = remaining_requests

        # If a request has been waiting for more than 10s, immediately allocate a vpod to it
        # target_queue = self.prefill_request_queue if prefill_or_decode == "prefill" else self.decode_request_queue
        # request_queue = [
        #     req for req in target_queue if timestamp - req.enqueue_timestamp >= 10 * 1e9
        # ]
        # if len(request_queue) > 0:
        #     logging.debug(
        #         "(%s) (%d) %d out of %d %s requests have been waiting for more than 10s. Immediately trigger scaling event.",
        #         self.simulator.name,
        #         timestamp,
        #         len(request_queue),
        #         initial_queue_length,
        #         prefill_or_decode,
        #     )
        #     # trigger scaling event
        #     self.simulator.put(
        #         vPodMultiDimensionalScalingEvent(
        #             timestamp,
        #             None,
        #             prefill_or_decode,
        #             True,
        #             False,
        #         )
        #     )

        # dispatch any remaining requests that have been waiting for a long time to avoid starvation
        target_queue = remaining_requests
        request_queue = deque()
        remaining_requests = deque()
        for req in target_queue:
            if timestamp - req.enqueue_timestamp >= 60 * 1e9:  # 1min
                request_queue.append(req)
            else:
                remaining_requests.append(req)
        if len(request_queue) > 0:
            if logging.level_debug():
                logging.debug(
                    "(%s) (%d) %s: %d out of %d requests are waiting for longer than 1min. Promptly dispatching them to any available vPods.",
                    self.simulator.name,
                    timestamp,
                    prefill_or_decode,
                    len(request_queue),
                    initial_queue_length,
                )
            # get all available vpods
            available_vpods = [vpod for vpod in vpods_sorted_by_util if vpod.ready]
            if len(available_vpods) > 0:
                # if prefill_or_decode == "prefill":
                #     self.prefill_request_queue = deque(
                #         sorted(
                #             self.prefill_request_queue,
                #             key=lambda r: r.input_seqlen,
                #             # reverse=True,
                #         )
                #     )
                #     request_queue = self.prefill_request_queue
                # else:
                #     self.decode_request_queue = deque(
                #         sorted(
                #             self.decode_request_queue,
                #             key=lambda r: r.total_seqlen,
                #             # reverse=True,
                #         )
                #     )
                #     request_queue = self.decode_request_queue
                request_queue = deque(
                    sorted(
                        request_queue,
                        key=lambda r: (
                            r.input_seqlen
                            if prefill_or_decode == "prefill"
                            else r.total_seqlen
                        ),
                        # reverse=True,
                    )
                )
                queue_size_before_dispatch = len(request_queue)
                self.dispatch_requests_to_vpods(
                    timestamp, available_vpods, request_queue, prefill_or_decode
                )
                if logging.level_debug():
                    logging.debug(
                        "(%s) (%d) %s: Dispatched %d requests. Remaining requests in the queue: %d",
                        self.simulator.name,
                        timestamp,
                        prefill_or_decode,
                        queue_size_before_dispatch - len(request_queue),
                        len(remaining_requests),
                    )
                request_queue.extend(remaining_requests)
                if prefill_or_decode == "prefill":
                    self.prefill_request_queue = request_queue
                else:
                    self.decode_request_queue = request_queue

    def dispatch_requests_to_vpods(
        self,
        timestamp: int,
        vpods_sorted_by_util: Sequence[
            LLMInferencePrefillInstance | LLMInferenceDecodeInstance
        ],
        request_queue: deque[LLMRequest],
        prefill_or_decode: str,
    ):
        """
        Dispatch requests to vPods with simple heuristics:
        - prioritize more idle vPods
        - try to balance the number of requests across all idle vPods
        - prioritize short requests
        - try to group requests with similar seqlen in the same batch

        Args:
            timestamp: current simulation timestamp
            vpods_sorted_by_util: list of idle vPods sorted by their head_of_line_request_end_timestamp
            request_queue: deque of requests to be dispatched
            prefill_or_decode: "prefill" or "decode"
        """
        # Token-budget-based batching: distribute tokens evenly across idle vPods
        num_idle_pods = len(vpods_sorted_by_util)
        total_queue_tokens = sum(
            r.input_seqlen if prefill_or_decode == "prefill" else r.total_seqlen
            for r in request_queue
        )
        target_token_budget = math.ceil(total_queue_tokens / num_idle_pods)
        if logging.level_debug():
            logging.debug(
                "vPod ready status: %s", [v.ready for v in vpods_sorted_by_util]
            )
            logging.debug(
                "Number of idle %s vPods: %d. Total queue tokens: %d. Target token budget: %d",
                prefill_or_decode,
                num_idle_pods,
                total_queue_tokens,
                target_token_budget,
            )

        for vpod in vpods_sorted_by_util:
            token_budget = vpod.get_token_budget(prefill_or_decode)
            remaining_budget = min(token_budget, target_token_budget)
            logging.debug(
                "%s vPod %s is idle, token budget: %d (capped to %d)",
                prefill_or_decode,
                vpod.id,
                token_budget,
                remaining_budget,
            )
            requests: list[LLMRequest] = []
            while len(request_queue) > 0:
                next_request = request_queue[0]
                req_tokens = (
                    next_request.input_seqlen
                    if prefill_or_decode == "prefill"
                    else next_request.total_seqlen
                )
                if len(requests) > 0:
                    # Token budget check
                    if remaining_budget <= 0 or req_tokens > remaining_budget:
                        break
                    # Seqlen coherence checks
                    if prefill_or_decode == "prefill":
                        seqlen_diff = (
                            next_request.input_seqlen / requests[0].input_seqlen
                            if next_request.input_seqlen >= requests[0].input_seqlen
                            else requests[0].input_seqlen / next_request.input_seqlen
                        )
                        if (
                            # For requests <= 32K tokens, we use a 4x threshold.
                            (requests[0].input_seqlen <= 32 * 1024 and seqlen_diff > 4)
                            # For requests between 32K and 64K tokens, we use a 1.6x threshold.
                            or (
                                32 * 1024 < requests[0].input_seqlen <= 64 * 1024
                                and seqlen_diff > 1.6
                            )
                            # For requests between 128K and 256K tokens, we use a 1.4x threshold.
                            or (
                                64 * 1024 < requests[0].input_seqlen <= 256 * 1024
                                and seqlen_diff > 1.4
                            )
                            # For requests > 256K tokens, we use a 1.2x threshold.
                            or (
                                requests[0].input_seqlen > 256 * 1024
                                and seqlen_diff > 1.2
                            )
                        ):  # if the next request is too different, stop
                            break
                    elif prefill_or_decode == "decode":
                        # Decode batching coherence. Below the length floor both sequences are
                        # short -> batch regardless of ratio (padding is cheap). At or above
                        # the floor, only batch when the total_seqlen ratio is within bound.
                        wcfg = self.simulator.config.workload_config
                        larger = max(
                            next_request.total_seqlen, requests[0].total_seqlen
                        )
                        if larger >= wcfg.decode_batch_seqlen_min_threshold:
                            smaller = min(
                                next_request.total_seqlen, requests[0].total_seqlen
                            )
                            seqlen_diff = larger / max(1, smaller)
                            if seqlen_diff > wcfg.decode_batch_seqlen_ratio_threshold:
                                # next request is too different in length; stop the batch
                                break
                    else:
                        raise ValueError(
                            f"Unknown prefill_or_decode: {prefill_or_decode}"
                        )
                request_queue.popleft()
                requests.append(next_request)
                remaining_budget -= req_tokens

            # Enqueue request start event
            if len(requests) > 0:
                if prefill_or_decode == "prefill":
                    start_event = LLMInferencePrefillStartEvent(
                        requests,
                        timestamp + self.prefill_schedule_delay_ns,
                        vpod.id,
                    )
                elif prefill_or_decode == "decode":
                    # Determine num_iterations based on output seqlen of the request.
                    # (In reality, we do not know the output length in advance, so there
                    # might be a sophisticated algorithm to predict this. But for our simulation purposes,
                    # we can just pick a value based on the true output seqlen.)
                    min_num_iter = self.simulator.config.workload_config.min_decode_schedule_num_iterations
                    max_num_iter = self.simulator.config.workload_config.max_decode_schedule_num_iterations
                    num_iterations = get_decode_schedule_num_iterations(
                        requests, min_num_iter, max_num_iter
                    )

                    start_event = LLMInferenceDecodeIterationStartEvent(
                        requests,
                        timestamp + self.decode_schedule_delay_ns,
                        vpod.id,
                        num_iterations,
                    )
                else:
                    raise ValueError(f"Unknown prefill_or_decode: {prefill_or_decode}")
                self.simulator.put(start_event)
                vpod.num_pending_requests += len(requests)
                vpod.busy = True
                logging.debug(
                    "Dispatched %s requests: %s to vPod %s. Queue length: %d",
                    prefill_or_decode,
                    requests,
                    vpod.id,
                    len(request_queue),
                )

    def prefill_end_and_enqueue_decode_request(
        self, event: LLMInferencePrefillEndEvent
    ):
        # try to dispatch more prefill requests
        self.try_dispatch_prefill_requests(event.timestamp)
        # enqueue decode requests
        self.decode_request_queue += event.requests
        logging.debug(
            "Enqueued decode requests: %s. Decode queue length: %d",
            event.requests,
            len(self.decode_request_queue),
        )
        # try to dispatch decode requests to idle decode vPods
        self.try_dispatch_decode_requests(event.timestamp)

    def decode_iteration_end_and_enqueue_next_iteration(
        self, event: LLMInferenceDecodeIterationEndEvent
    ):
        # try to dispatch more decode requests
        # self.try_dispatch_decode_requests(event.timestamp)
        # if the request is not finished, enqueue the next decode iteration
        new_requests = []
        for request in event.requests:
            if not request.is_decode_finished():
                new_requests.append(request)
            else:
                logging.debug(
                    "Decode request %s finished decoding at %d",
                    request.id,
                    event.timestamp,
                )
        if len(new_requests) > 0:
            # prioritize the requests that are not yet finished decoding
            self.decode_request_queue.extendleft(new_requests)
            logging.debug(
                "Enqueued next decode iteration requests: %s. Decode queue length: %d",
                new_requests,
                len(self.decode_request_queue),
            )
        # try to dispatch decode requests to idle decode vPods
        self.try_dispatch_decode_requests(event.timestamp)

    def on_engine_ready(
        self,
        event: (
            LLMInferenceEngineReadyEvent | vPodCreateEvent | vPodReconfigurationEndEvent
        ),
    ):
        """
        This method is called when the inference engine is ready to process more requests.
        It will try to dispatch prefill and decode requests to idle vPods.
        """
        logging.debug(
            "Try to dispatch requests for ready worker %s at timestamp %d",
            event.worker_id,
            event.timestamp,
        )
        self.try_dispatch_prefill_requests(event.timestamp)
        self.try_dispatch_decode_requests(event.timestamp)

    def on_instance_destroy(self, event: vPodDestroyEvent):
        """
        This method is called when an instance is destroyed.
        It removes the instance from the prefill or decode vpod list.
        """
        vpod_id = event.worker_id
        # find vpod_id in prefill/decode vpods
        # for vpod in self.prefill_vpods.copy().values():
        #     if vpod.id == vpod_id:
        vpod = self.lookup_vpod_by_id(vpod_id, only_live=True)
        if not vpod:
            return
        self.simulator.cluster_manager.track_deallocate(  # type: ignore
            vpod.config.llm_config.name, vpod.config.llm_config.num_chips
        )
        vpod.on_instance_destroy(event)
        self._unregister_vpod_config(vpod)
        if isinstance(vpod, LLMInferencePrefillInstance):
            del self.prefill_vpods[vpod.id]
            self.destroyed_prefill_vpods[vpod.id] = vpod
            logging.debug(
                "Prefill vPod %s is removed from the endpoint at %d",
                vpod.id,
                event.timestamp,
            )
        elif isinstance(vpod, LLMInferenceDecodeInstance):
            del self.decode_vpods[vpod.id]
            self.destroyed_decode_vpods[vpod.id] = vpod
            logging.debug(
                "Decode vPod %s is removed from the endpoint at %d",
                vpod.id,
                event.timestamp,
            )

    def on_horizontal_scaling_recommendation(
        self, event: vPodHorizontalScalingRecommendationEvent
    ):
        """
        This method is called when a horizontal scaling recommendation is received.
        It will trigger the scaling of the vPods.
        """
        autoscaler = (
            self.prefill_autoscaler
            if event.prefill_or_decode == "prefill"
            else self.decode_autoscaler
        )
        recommended_replica_count = autoscaler.get_recommended_num_vPods(
            timestamp=event.timestamp
        )
        current_replica_count = (
            len(self.prefill_vpods)
            if event.prefill_or_decode == "prefill"
            else len(self.decode_vpods)
        )
        if recommended_replica_count != current_replica_count:
            if (
                event.do_not_scale_down
                and recommended_replica_count < current_replica_count
            ):
                logging.debug(
                    "Not scaling down vPods (recommendation: %d -> %d).",
                    current_replica_count,
                    recommended_replica_count,
                )
                return
            # add some safe guards to prevent too aggressive scaling
            if getattr(autoscaler, "last_recommendation_queue_driven", False):
                # queue-driven: allow up to 4x
                if recommended_replica_count > 4 * current_replica_count:
                    recommended_replica_count = 4 * current_replica_count
            elif recommended_replica_count > 2 * current_replica_count:
                recommended_replica_count = 2 * current_replica_count
            if recommended_replica_count < current_replica_count / 2:
                recommended_replica_count = round(current_replica_count / 2)
            if logging.level_debug():
                logging.debug(
                    "Horizontal scaling recommendation: num_vPods %d -> %d",
                    current_replica_count,
                    recommended_replica_count,
                )
            # invoke cluster manager to request for new allocation
            self.simulator.put(
                vPodHorizontalScalingEvent(
                    event.timestamp,
                    recommended_replica_count,
                    event.prefill_or_decode,
                )
            )

    def on_vertical_scaling_recommendation(
        self, event: vPodVerticalScalingRecommendationEvent
    ):
        """
        This method is called when a vertical scaling recommendation is received.
        It will trigger the scaling of the vPods.
        """
        autoscaler = (
            self.prefill_autoscaler
            if event.prefill_or_decode == "prefill"
            else self.decode_autoscaler
        )
        recommended_configs = autoscaler.get_recommended_vPod_config(
            timestamp=event.timestamp
        )
        all_vPod_configs = (
            [vpod.config.llm_config for vpod in self.prefill_vpods.values()]
            if event.prefill_or_decode == "prefill"
            else [vpod.config.llm_config for vpod in self.decode_vpods.values()]
        )
        if logging.level_debug():
            logging.debug(
                "current vPod num_chips: %s", [c.num_chips for c in all_vPod_configs]
            )
            logging.debug(
                "recommended vPod num_chips: %s",
                [cfg.num_chips for cfg in recommended_configs],
            )  # type: ignore
        if any(
            curr_config.num_chips != recommended_configs[0].num_chips
            for curr_config in all_vPod_configs
        ):
            if logging.level_debug():
                logging.debug(
                    "Vertical scaling recommendation: num_chips %s -> %s",
                    [c.num_chips for c in all_vPod_configs],
                    recommended_configs[0].num_chips,
                )
            self.simulator.put(
                vPodVerticalScalingEvent(
                    event.timestamp,
                    recommended_configs[0],
                    event.prefill_or_decode,
                )
            )

    def on_mpa_scaling_recommendation(
        self, event: vPodMultiDimensionalScalingRecommendationEvent
    ):
        """
        This function is just a wrapper to fire the MPA scaling event.
        It does not do anything except for getting the recommendations from the autoscaler.
        It is just there for consistency with the HS/VS events.
        """
        # autoscaler = (
        #     self.prefill_autoscaler
        #     if event.prefill_or_decode == "prefill"
        #     else self.decode_autoscaler
        # )
        # recommended_allocations = autoscaler.get_recommended_allocation(
        #     timestamp=event.timestamp,
        #     current_request_queue=(
        #         self.prefill_request_queue
        #         if event.prefill_or_decode == "prefill"
        #         else self.decode_request_queue
        #     ),
        # )
        recommended_allocations = (
            None  # get the recommendation later in the actual MPA scaling event
        )
        if logging.level_debug():
            logging.debug(
                "current vPod configs: %s",
                [
                    vpod.config.llm_config
                    for vpod in (
                        self.prefill_vpods.values()
                        if event.prefill_or_decode == "prefill"
                        else self.decode_vpods.values()
                    )
                ],
            )
            logging.debug("recommended vPod configs: %s", recommended_allocations)
        # Just fire the MPA event. The actual scaling decision will be handled in that event.
        self.simulator.put(
            vPodMultiDimensionalScalingEvent(
                event.timestamp,
                recommended_allocations,
                event.prefill_or_decode,
                event.do_not_scale_down,
                event.is_routine,
            )
        )

    def on_vpod_horizontal_scaling(self, event: vPodHorizontalScalingEvent):
        """
        This method is called when a vPod is scaled horizontally (i.e., adding/removing replicas).
        It will fire vPodCreateEvent and vPodDestroyEvent.
        """
        if event.prefill_or_decode == "prefill":
            diff_num_replicas = event.replica_count - len(self.prefill_vpods)
            num_current_replicas = len(self.prefill_vpods)
            vpod_dict = self.prefill_vpods
            autoscaler = self.prefill_autoscaler
            vPodType = LLMInferencePrefillInstance
        elif event.prefill_or_decode == "decode":
            diff_num_replicas = event.replica_count - len(self.decode_vpods)
            num_current_replicas = len(self.decode_vpods)
            vpod_dict = self.decode_vpods
            autoscaler = self.decode_autoscaler
            vPodType = LLMInferenceDecodeInstance
        else:
            raise ValueError("Invalid prefill_or_decode value")

        if diff_num_replicas > 0:
            # create new replicas
            # assert isinstance(recommended_config, LLMConfig)
            for i in range(diff_num_replicas):
                recommended_configs = autoscaler.get_recommended_vPod_config(
                    timestamp=event.timestamp
                )
                allocation_success = False
                recommended_config = None
                for config in recommended_configs:
                    assert isinstance(config, LLMConfig)
                    allocation_success = self.simulator.cluster_manager.allocate(  # type: ignore
                        NPUPodAllocationRequest(
                            event.timestamp,
                            npu_type=config.name,
                            num_chips=config.num_chips,
                        )
                    )
                    if allocation_success:
                        recommended_config = config
                        break
                    if logging.level_debug():
                        logging.debug(
                            "Failed to allocate new %s vPod (type=%s, num_chips=%d) at %d due to insufficient resources.",
                            event.prefill_or_decode,
                            config.name,
                            config.num_chips,
                            event.timestamp,
                        )

                if allocation_success:
                    assert recommended_config is not None
                    self.simulator.cluster_manager.track_allocate(  # type: ignore
                        recommended_config.name, recommended_config.num_chips
                    )
                    if logging.level_debug():
                        logging.debug(
                            "Scaling up %s: %d vPods -> %d vPods.",
                            event.prefill_or_decode,
                            num_current_replicas,
                            len(vpod_dict),
                        )
                    new_vpod = vPodType(
                        simulator=self.simulator,
                        name=f"{self.name}_{event.prefill_or_decode}_replica_{num_current_replicas + i}",
                        config=self.simulator.config.workload_config,
                        pod_config=recommended_config,
                    )
                    new_vpod.initialize()
                    new_vpod.busy = True  # cannot take requests when creating
                    new_vpod.under_creation = True  # mark as under creation
                    vpod_dict[new_vpod.id] = new_vpod  # type: ignore
                    self._register_vpod_config(new_vpod)
                    self.simulator.put(
                        vPodCreateEvent(
                            event.timestamp
                            + autoscaler.get_vPod_create_delay_ns(new_vpod),
                            new_vpod.id,
                        )
                    )
        elif diff_num_replicas < 0:
            # destroy the least busy replicas after they drained their current requests
            num_replicas_to_destroy = -diff_num_replicas
            vpods_sorted_by_util = [v for v in vpod_dict.values() if not v.busy][
                :num_replicas_to_destroy
            ]
            while len(vpods_sorted_by_util) > 0:
                vpod = vpods_sorted_by_util.pop()
                vpod.pending_destroy = True
                self.simulator.put(
                    vPodDestroyEvent(
                        max(
                            event.timestamp,
                            vpod.head_of_line_request_end_timestamp or 0,
                        ),
                        vpod.id,
                    )
                )
        else:
            # no change in replicas
            pass

    def on_vpod_vertical_scaling(self, event: vPodVerticalScalingEvent):
        """
        This method is called when a vertical scaling event is received.
        It will trigger the scaling of the vPods.
        """
        if event.prefill_or_decode == "prefill":
            vpod_dict = self.prefill_vpods
            autoscaler = self.prefill_autoscaler
        elif event.prefill_or_decode == "decode":
            vpod_dict = self.decode_vpods
            autoscaler = self.decode_autoscaler
        else:
            raise ValueError("Invalid prefill_or_decode value")

        recommended_configs = autoscaler.get_recommended_vPod_config(
            timestamp=event.timestamp
        )
        vpods_to_update = [
            vpod
            for vpod in vpod_dict.values()
            # all vPods that is not in optimal config may be potential reconfiguration candidates
            if vpod.config.llm_config != recommended_configs[0]
            and not vpod.pending_destroy
            and not vpod.pending_or_under_reconfiguration
        ]
        logging.debug("vpods to update: %s", [vpod.id for vpod in vpods_to_update])
        for vpod in vpods_to_update:
            # There are two cases:
            #    1. The vPod config is alreaady in the recommended config list but not the optimal one.
            #       We need to promote this vPod to a better config, so we will not attempt to allocate
            #       configs that are less optimal than it.
            #    2. The vPod config is not in the recommended config list. We try to allocate until
            #       successful or the list is exhausted.
            recommended_config = None
            allocation_success = False
            for config in recommended_configs:
                assert isinstance(config, LLMConfig)
                if vpod.config.llm_config == config:
                    if logging.level_debug():
                        logging.debug(
                            "Failed to reconfigure vPod %s at %d since all better configs cannot be allocated due to insufficient resources.",
                            vpod.id,
                            event.timestamp,
                        )
                    return  # no need to try more sub-optimal configs
                if (
                    config.num_chips <= vpod.config.llm_config.num_chips
                    and config.name == vpod.config.llm_config.name
                ):
                    allocation_success = True
                else:  # only need to request new allocation for scale-up or change-npu-type events
                    allocation_success = self.simulator.cluster_manager.allocate(  # type: ignore
                        NPUPodAllocationRequest(
                            event.timestamp,
                            npu_type=config.name,
                            num_chips=config.num_chips,
                        )
                    )
                if allocation_success:
                    recommended_config = config
                    # For scale-up or type-change, track the new allocation now
                    if not (
                        config.num_chips <= vpod.config.llm_config.num_chips
                        and config.name == vpod.config.llm_config.name
                    ):
                        self.simulator.cluster_manager.track_allocate(  # type: ignore
                            config.name, config.num_chips
                        )
                    break
                if logging.level_debug():
                    logging.debug(
                        "Failed to reconfigure vPod%s (num_chips: %s -> %s, npu_type: %s -> %s) at %d due to insufficient resources.",
                        vpod.id,
                        vpod.config.llm_config.num_chips,
                        config.num_chips,
                        vpod.config.llm_config.name,
                        config.name,
                        event.timestamp,
                    )
            # schedule vPod reconfiguration start events
            if allocation_success:
                assert recommended_config
                reconfig_delay_ns = autoscaler.get_vPod_reconfig_delay_ns(
                    vpod.config.llm_config, recommended_config
                )
                vpod.pending_or_under_reconfiguration = True
                self.simulator.put(
                    vPodReconfigurationStartEvent(
                        max(
                            event.timestamp,
                            vpod.head_of_line_request_end_timestamp or 0,
                        ),
                        vpod.id,
                        recommended_config,
                        reconfig_delay_ns,
                    )
                )
                if logging.level_debug():
                    logging.debug(
                        "Vertical scaling for vPod %s: num_chips: %s -> %s, npu_type: %s -> %s",
                        vpod.id,
                        vpod.config.llm_config.num_chips,
                        recommended_config.num_chips,
                        vpod.config.llm_config.name,
                        recommended_config.name,
                    )

    def on_vpod_mpa_scaling(self, event: vPodMultiDimensionalScalingEvent):
        if event.prefill_or_decode == "prefill":
            vpod_dict = self.prefill_vpods
            autoscaler = self.prefill_autoscaler
            vPodType = LLMInferencePrefillInstance
        elif event.prefill_or_decode == "decode":
            vpod_dict = self.decode_vpods
            autoscaler = self.decode_autoscaler
            vPodType = LLMInferenceDecodeInstance
        else:
            raise ValueError("Invalid prefill_or_decode value")

        vpods_by_config = (
            self._prefill_vpods_by_config
            if event.prefill_or_decode == "prefill"
            else self._decode_vpods_by_config
        )

        if self.simulator.enable_profile:
            logging.info(
                "(%s) MPA scaling at %d: %d prefill vpods (%d configs), %d decode vpods (%d configs)",
                event.prefill_or_decode,
                event.timestamp,
                len(self.prefill_vpods),
                len(self._prefill_vpods_by_config),
                len(self.decode_vpods),
                len(self._decode_vpods_by_config),
            )
            logging.info(
                "(%s) MPA scaling at %d: Prefill configs: %s",
                event.prefill_or_decode,
                event.timestamp,
                [
                    (cfg.name, util.get_pstr(cfg), len(vpods))
                    for cfg, vpods in self._prefill_vpods_by_config.items()
                ],
            )
            logging.info(
                "(%s) MPA scaling at %d: Decode configs: %s",
                event.prefill_or_decode,
                event.timestamp,
                [
                    (cfg.name, util.get_pstr(cfg), len(vpods))
                    for cfg, vpods in self._decode_vpods_by_config.items()
                ],
            )

        # recommended_allocations = event.vPod_allocations
        recommended_allocations = autoscaler.get_recommended_allocation(
            timestamp=event.timestamp,
            current_request_queue=(
                self.prefill_request_queue
                if event.prefill_or_decode == "prefill"
                else self.decode_request_queue
            ),
        )
        for cfgs, _count in recommended_allocations:
            for cfg in cfgs:
                assert isinstance(cfg, LLMConfig)
                cfg.input_seqlen = 32  # dummy value
                cfg.output_seqlen = 32  # dummy value

        ### diff the current allocation and recommended allocation
        ###    Case (1): config exists, need to change number of replicas (add/remove replicas)
        ###    Case (2): config does not exist, need to add new vPods
        ### TODO: For now, we just model add/remove replicas.
        ###       In reality, our complete design should include an algorithm
        ###       to do reconfiguration as well. For example, we can transform some existing
        ###       config that are not needed anymore into new config that are requested.
        ###       Currently, we just remove the old instance and start a new instance with the new config.
        ###       We should leverage reconfiguration in the future as it may reduce some overheads.
        ### This part also needs to handle allocation failures,
        ### as we derive the two diff lists below.

        # config -> how many new vPods we need to add
        allocations_to_add: dict[LLMConfig, int] = {}

        # config -> how many vPods we need to remove
        allocations_to_remove: dict[LLMConfig, int] = {}

        # config -> how many vPods we should not change
        allocations_to_preserve: dict[LLMConfig, int] = {}

        ### 1. figure out allocations_to_add
        ### try allocate for each list of candidate configs in the recommendation
        # skip the allocation if one pod with the recommended config already exists
        for cfgs, count in recommended_allocations:
            for _ in range(count):
                allocation_success = False
                for cfg in cfgs:
                    assert isinstance(cfg, LLMConfig)
                    # check if we already have a pod with this config (O(1) lookup)
                    # Only match if we haven't already consumed all existing vPods
                    existing_count = len(vpods_by_config.get(cfg, {}))
                    preserved_count = allocations_to_preserve.get(cfg, 0)
                    if cfg in vpods_by_config and preserved_count < existing_count:
                        allocation_success = True
                        allocations_to_preserve[cfg] = preserved_count + 1
                    if allocation_success:
                        if logging.level_debug():
                            logging.debug(
                                "Skipping allocation for MPA scaling at %d since we already have a vPod with config %s.",
                                event.timestamp,
                                cfg,
                            )
                        break
                    allocation_success = self.simulator.cluster_manager.allocate(  # type: ignore
                        NPUPodAllocationRequest(
                            event.timestamp,
                            npu_type=cfg.name,
                            num_chips=cfg.num_chips,
                        )
                    )
                    if allocation_success:
                        self.simulator.cluster_manager.track_allocate(  # type: ignore
                            cfg.name, cfg.num_chips
                        )
                        if cfg not in allocations_to_add:
                            allocations_to_add[cfg] = 1
                        else:
                            allocations_to_add[cfg] += 1
                        break
                    if logging.level_debug():
                        logging.debug(
                            "Failed to allocate new %s vPod (type=%s, num_chips=%d) at %d due to insufficient resources.",
                            event.prefill_or_decode,
                            cfg.name,
                            cfg.num_chips,
                            event.timestamp,
                        )
                if not allocation_success:
                    logging.debug(
                        "Failed to allocate new vPod for MPA scaling at %d due to insufficient resources. Skipping this config.",
                        event.timestamp,
                    )
                    break  # no need to try more if one fails since this means we do not have enough resources anyway

        if logging.level_debug():
            logging.debug(
                "(%s) MPA scaling recommendation at %d: to_add=%s, to_remove=%s, to_preserve=%s",
                event.prefill_or_decode,
                event.timestamp,
                [
                    (cfg.name, cfg.num_chips, cnt)
                    for cfg, cnt in allocations_to_add.items()
                ],
                [
                    (cfg.name, cfg.num_chips, cnt)
                    for cfg, cnt in allocations_to_remove.items()
                ],
                [
                    (cfg.name, cfg.num_chips, cnt)
                    for cfg, cnt in allocations_to_preserve.items()
                ],
            )
            logging.debug(
                "(%s) Current vPod configs: %s",
                event.prefill_or_decode,
                [
                    (vpod.config.llm_config.name, vpod.config.llm_config.num_chips)
                    for vpod in vpod_dict.values()
                ],
            )

        #### gather current allocations into [(config, count)]
        # O(K) where K = number of unique configs, via secondary index
        current_allocations: dict[LLMConfig, int] = {
            cfg: len(vpods) for cfg, vpods in vpods_by_config.items()
        }

        # add safe guards to prevent too aggressive scaling
        # Use relaxed caps for configs where queue-depth drives the recommendation
        queue_driven_cfgs = getattr(autoscaler, "queue_depth_driven_configs", set())
        # Build a set of config objects that are queue-driven (match by equality)
        queue_driven_config_set: set[LLMConfig] = set()
        for cfgs_tuple in queue_driven_cfgs:
            for cfg in cfgs_tuple:
                queue_driven_config_set.add(cfg)

        # When queue backlog is large, relax scaling caps to drain faster
        current_queue = (
            self.prefill_request_queue
            if event.prefill_or_decode == "prefill"
            else self.decode_request_queue
        )
        queue_backlog = len(current_queue)
        # Threshold: if queue has more than 2x total current vpod count, consider it a burst
        total_current_vpods = sum(current_allocations.values())
        burst_mode = queue_backlog > 2 * max(total_current_vpods, 1)

        for config, add_count in allocations_to_add.items():
            is_queue_driven = config in queue_driven_config_set
            if config in current_allocations:
                current_count = current_allocations[config]
                if burst_mode:
                    # In burst mode, allow up to 4x for rate-driven, uncapped for queue-driven
                    if is_queue_driven:
                        pass  # no cap — let the autoscaler recommendation drive
                    else:
                        if add_count > current_count * 3:
                            allocations_to_add[config] = current_count * 3
                else:
                    if is_queue_driven:
                        # queue-driven: cap at current_count * 3 (4x total)
                        if add_count > current_count * 3:
                            allocations_to_add[config] = current_count * 3
                    else:
                        # ensure at max we double the # of vPods for this config
                        if add_count > current_count:
                            allocations_to_add[config] = current_count
            else:
                if burst_mode:
                    # In burst mode, allow more new configs
                    if is_queue_driven:
                        allocations_to_add[config] = min(add_count, 64)
                    else:
                        allocations_to_add[config] = min(add_count, 32)
                else:
                    if is_queue_driven:
                        # queue-driven new config: cap at 32
                        allocations_to_add[config] = min(add_count, 32)
                    else:
                        # if new config, at max add 8 vPods as the starting point
                        allocations_to_add[config] = min(add_count, 8)

        ### 2. figure out allocations_to_remove
        if event.do_not_scale_down:
            # if we are not allowed to scale down, do nothing
            pass
        else:
            allocations_to_remove = current_allocations
            for config, preserve_count in allocations_to_preserve.items():
                if config in allocations_to_remove:
                    current_count = allocations_to_remove[config]
                    # add some safe guards to prevent too aggressive scaling
                    if preserve_count < current_count / 2:
                        preserve_count = round(current_count / 2)
                    allocations_to_remove[config] -= preserve_count
                    if allocations_to_remove[config] <= 0:
                        del allocations_to_remove[config]

        ### END diff allocations ###

        ### fire scaling events based on the diff results
        # handle removals to free up resources
        for config, count in allocations_to_remove.items():
            # O(V_per_config) via secondary index instead of O(V)
            vpods_for_config = vpods_by_config.get(config, {})
            free_vpods = [vpod for vpod in vpods_for_config.values() if vpod.ready]

            # scale in the vPods
            for vpod in free_vpods[:count]:
                vpod.pending_destroy = True
                self.simulator.put(
                    vPodDestroyEvent(
                        max(
                            event.timestamp,
                            vpod.head_of_line_request_end_timestamp or 0,
                        ),
                        vpod.id,
                    )
                )
                if logging.level_debug():
                    logging.debug(
                        "(%s) MPA scaling: Destroying vPod %s (num_chips: %s, npu_type: %s)",
                        event.prefill_or_decode,
                        vpod.id,
                        vpod.config.llm_config.num_chips,
                        vpod.config.llm_config.name,
                    )

        # handle adding new vPods
        # since we already handle allocation failures when deriving the two alloc diff lists,
        # we just need to fire vPod creation events here
        for config, count in allocations_to_add.items():
            for _ in range(count):
                new_vpod = vPodType(
                    simulator=self.simulator,
                    name=f"{self.name}_{event.prefill_or_decode}_replica_{config.name}-{config.num_chips}",
                    config=self.simulator.config.workload_config,
                    pod_config=config,
                )
                new_vpod.initialize()
                new_vpod.busy = True  # cannot take requests when creating
                new_vpod.under_creation = True  # mark as under creation
                vpod_dict[new_vpod.id] = new_vpod  # type: ignore
                self._register_vpod_config(new_vpod)
                self.simulator.put(
                    vPodCreateEvent(
                        event.timestamp + autoscaler.get_vPod_create_delay_ns(new_vpod),
                        new_vpod.id,
                    )
                )
                if logging.level_debug():
                    logging.debug(
                        "(%s) MPA scaling: Creating vPod %s (num_chips: %s, npu_type: %s)",
                        event.prefill_or_decode,
                        new_vpod.id,
                        config.num_chips,
                        config.name,
                    )

    def dump_simulation_stats(self):
        """
        Dump/print the simulation stats for the endpoint.
        """
        logging.info("%s simulation stats:", self.name)
        vpod_lifecycle_traces = {
            "prefill": [],
            "decode": [],
        }
        for vpod in (self.prefill_vpods | self.destroyed_prefill_vpods).values():
            logging.debug(
                "  Prefill vPod %s processed %d tokens (Created=%d, Destroyed=%d)",
                vpod.id,
                vpod.num_processed_tokens,
                vpod.create_timestamp,
                vpod.destroy_timestamp,
            )
            vpod_lifecycle_traces["prefill"].append(
                {
                    "id": vpod.id,
                    "npu_type": vpod.config.llm_config.name,
                    "num_chips": vpod.config.llm_config.num_chips,
                    "pcfg": util.get_pstr(vpod.config.llm_config),
                    "num_processed_tokens": vpod.num_processed_tokens,
                    "create_timestamp": vpod.create_timestamp,
                    "destroy_timestamp": (
                        vpod.destroy_timestamp
                        if vpod.destroy_timestamp > 0
                        else self.simulator.timestamp
                    ),
                }
            )
        for vpod in (self.decode_vpods | self.destroyed_decode_vpods).values():
            logging.debug(
                "  Decode vPod %s processed %d tokens (Created=%d, Destroyed=%d)",
                vpod.id,
                vpod.num_processed_tokens,
                vpod.create_timestamp,
                vpod.destroy_timestamp,
            )
            vpod_lifecycle_traces["decode"].append(
                {
                    "id": vpod.id,
                    "npu_type": vpod.config.llm_config.name,
                    "num_chips": vpod.config.llm_config.num_chips,
                    "pcfg": util.get_pstr(vpod.config.llm_config),
                    "num_processed_tokens": vpod.num_processed_tokens,
                    "create_timestamp": vpod.create_timestamp,
                    "destroy_timestamp": (
                        vpod.destroy_timestamp
                        if vpod.destroy_timestamp > 0
                        else self.simulator.timestamp
                    ),
                }
            )

        # dump lifecycle traces to file
        with open(
            os.path.join(
                self.simulator.config.output_dir, "vpod_lifecycle_traces.json"
            ),
            "w",
        ) as f:
            json.dump(vpod_lifecycle_traces, f, indent=4)
            logging.info("Dumped vPod lifecycle traces to %s", f.name)

        # do not print these; too long
        # logging.info("  Prefill vPod Reconfiguration Events:")
        # for vpod in self.prefill_vpods + self.destroyed_prefill_vpods:
        #     if len(vpod.reconfiguration_trace) > 0:
        #         logging.info("    vPod %s reconfiguration events:", vpod.id)
        #         for reconfig_event in vpod.reconfiguration_trace:
        #             logging.info("      %s", reconfig_event)
        # logging.info("  Decode vPod Reconfiguration Events:")
        # for vpod in self.decode_vpods + self.destroyed_decode_vpods:
        #     if len(vpod.reconfiguration_trace) > 0:
        #         logging.info("    vPod %s reconfiguration events:", vpod.id)
        #         for reconfig_event in vpod.reconfiguration_trace:
        #             logging.info("      %s", reconfig_event)

        def get_parallelism_str(cfg: LLMConfig) -> str:
            s = f"bs{cfg.microbatch_size_ici}-dp{cfg.data_parallelism_degree}-tp{cfg.tensor_parallelism_degree}-pp{cfg.pipeline_parallelism_degree}"
            if isinstance(cfg, DeepSeekConfig):
                s += f"-ep{cfg.expert_parallelism_degree}"
            s += f"-inputseqlen{cfg.input_seqlen}-outputseqlen{cfg.output_seqlen}"
            return s

        # dump reconfiguration traces to file
        with open(
            os.path.join(
                self.simulator.config.output_dir, "vpod_reconfiguration_traces.json"
            ),
            "w",
        ) as f:
            json.dump(
                {
                    "prefill": {
                        vpod.id: [
                            (
                                start,
                                end,
                                cfg.name,
                                cfg.num_chips,
                                get_parallelism_str(cfg),
                            )
                            for (start, end, cfg) in vpod.reconfiguration_trace
                        ]
                        for vpod in (
                            self.prefill_vpods | self.destroyed_prefill_vpods
                        ).values()
                    },
                    "decode": {
                        vpod.id: [
                            (
                                start,
                                end,
                                cfg.name,
                                cfg.num_chips,
                                get_parallelism_str(cfg),
                            )
                            for (start, end, cfg) in vpod.reconfiguration_trace
                        ]
                        for vpod in (
                            self.prefill_vpods | self.destroyed_prefill_vpods
                        ).values()
                    },
                },
                f,
                indent=4,
            )
            logging.info("Dumped vPod reconfiguration traces to %s", f.name)

    def save_to_checkpoint(self, checkpoint_path: str):
        # remove the simulator attribute since it cannot be pickled
        # and restore it back after we pickled other attributes
        simulator = self.simulator
        prefill_autoscaler = self.prefill_autoscaler
        decode_autoscaler = self.decode_autoscaler
        prefill_vpods = self.prefill_vpods
        decode_vpods = self.decode_vpods
        destroyed_prefill_vpods = self.destroyed_prefill_vpods
        destroyed_decode_vpods = self.destroyed_decode_vpods

        self.prefill_autoscaler.save_to_checkpoint(
            checkpoint_path.replace(".pkl.gz", "_prefill_autoscaler.pkl.gz")
        )
        self.decode_autoscaler.save_to_checkpoint(
            checkpoint_path.replace(".pkl.gz", "_decode_autoscaler.pkl.gz")
        )
        for vpod in (
            prefill_vpods
            | decode_vpods
            | destroyed_prefill_vpods
            | destroyed_decode_vpods
        ).values():
            vpod.save_to_checkpoint(
                checkpoint_path.replace(".pkl.gz", f"_{vpod.id}.pkl.gz")
            )

        _prefill_vpods_by_config = self._prefill_vpods_by_config
        _decode_vpods_by_config = self._decode_vpods_by_config

        self.prefill_autoscaler = None  # type: ignore
        self.decode_autoscaler = None  # type: ignore
        self.simulator = None  # type: ignore
        self.prefill_vpods = None  # type: ignore
        self.decode_vpods = None  # type: ignore
        self.destroyed_prefill_vpods = None  # type: ignore
        self.destroyed_decode_vpods = None  # type: ignore
        self._prefill_vpods_by_config = None  # type: ignore
        self._decode_vpods_by_config = None  # type: ignore

        with gzip.open(checkpoint_path, "wb") as f:
            pickle.dump(self, f)

        # save vpod ids and types into a json file
        with open(checkpoint_path.replace(".pkl.gz", "_vpod_ids.json"), "w") as f:
            json.dump(
                {
                    "prefill": [vpod.id for vpod in prefill_vpods.values()],
                    "decode": [vpod.id for vpod in decode_vpods.values()],
                    "destroyed_prefill": [
                        vpod.id for vpod in destroyed_prefill_vpods.values()
                    ],
                    "destroyed_decode": [
                        vpod.id for vpod in destroyed_decode_vpods.values()
                    ],
                },
                f,
                indent=4,
            )

        self.simulator = simulator
        self.prefill_autoscaler = prefill_autoscaler
        self.decode_autoscaler = decode_autoscaler
        self.prefill_vpods = prefill_vpods
        self.decode_vpods = decode_vpods
        self.destroyed_prefill_vpods = destroyed_prefill_vpods
        self.destroyed_decode_vpods = destroyed_decode_vpods
        self._prefill_vpods_by_config = _prefill_vpods_by_config
        self._decode_vpods_by_config = _decode_vpods_by_config

    def load_from_checkpoint(self, checkpoint_path: str):
        with gzip.open(checkpoint_path, "rb") as f:
            obj = pickle.load(f)
            simulator = self.simulator
            self.__dict__.update(obj.__dict__)  # will load a None to self.simulator,
            self.simulator = simulator  # so we need to save-and-restore it
        self.prefill_autoscaler.load_from_checkpoint(
            checkpoint_path.replace(".pkl.gz", "_prefill_autoscaler.pkl.gz")
        )
        self.decode_autoscaler.load_from_checkpoint(
            checkpoint_path.replace(".pkl.gz", "_decode_autoscaler.pkl.gz")
        )
        with open(checkpoint_path.replace(".pkl.gz", "_vpod_ids.json")) as f:
            vpod_ids = json.load(f)
            for vpod_id, vpod in zip(
                vpod_ids["prefill"], self.prefill_vpods.values(), strict=False
            ):
                vpod.load_from_checkpoint(
                    checkpoint_path.replace(".pkl.gz", f"_{vpod_id}.pkl.gz")
                )
            for vpod_id, vpod in zip(
                vpod_ids["decode"], self.decode_vpods.values(), strict=False
            ):
                vpod.load_from_checkpoint(
                    checkpoint_path.replace(".pkl.gz", f"_{vpod_id}.pkl.gz")
                )
            for vpod_id, vpod in zip(
                vpod_ids["destroyed_prefill"],
                self.destroyed_prefill_vpods.values(),
                strict=False,
            ):
                vpod.load_from_checkpoint(
                    checkpoint_path.replace(".pkl.gz", f"_{vpod_id}.pkl.gz")
                )
            for vpod_id, vpod in zip(
                vpod_ids["destroyed_decode"],
                self.destroyed_decode_vpods.values(),
                strict=False,
            ):
                vpod.load_from_checkpoint(
                    checkpoint_path.replace(".pkl.gz", f"_{vpod_id}.pkl.gz")
                )
