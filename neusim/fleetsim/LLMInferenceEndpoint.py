import json
import math
import os
import typing
from collections import deque
from collections.abc import Sequence
from copy import deepcopy

from absl import logging

import neusim.fleetsim.npusim_backend_interface as sim_backend
import neusim.fleetsim.util as util
from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig
from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
    StaticVPodEntry,
)
from neusim.fleetsim import cost_model
from neusim.fleetsim.LLMInferenceEvents import (
    LLMInferenceDecodeIterationEndEvent,
    LLMInferenceDecodeIterationStartEvent,
    LLMInferenceEngineReadyEvent,
    LLMInferencePrefillEndEvent,
    LLMInferencePrefillStartEvent,
    LLMInferenceRequestEnqueueEvent,
)
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.fleetsim.SimObject import SimObject

if typing.TYPE_CHECKING:
    from neusim.fleetsim.NPUFleetSimulator import NPUFleetSimulator


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


class StaticVPodInstance(SimObject):
    """A statically provisioned virtual NPU pod."""

    def __init__(self, simulator: "NPUFleetSimulator", name: str):
        super().__init__(name, simulator)
        self.id = name


class LLMInferenceInstanceBase(StaticVPodInstance):
    """Common state for a statically provisioned prefill or decode engine."""

    def __init__(
        self,
        simulator: "NPUFleetSimulator",
        name: str,
        config: LLMInferenceWorkloadConfig,
        pod_config: LLMConfig | None = None,
    ):
        super().__init__(simulator, name)
        self.config: LLMInferenceWorkloadConfig = deepcopy(config)
        self._config_input_seqlen: int = 0
        self._config_output_seqlen: int = 0
        if pod_config:
            self.update_pod_config(pod_config)

        self.busy: bool = False
        self.head_of_line_request_end_timestamp: int | None = None
        self.num_pending_requests: int = 0

    @property
    def ready(self) -> bool:
        return not self.busy

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


class LLMInferencePrefillInstance(LLMInferenceInstanceBase):
    """A statically provisioned prefill engine."""

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
        batch_metrics = sim_backend.run_inference_request_batch(
            self.config.llm_config,
            temp_requests,
            self.simulator.config.workload_config.input_seqlen_padding_factors,
            self.simulator.config.workload_config.input_seqlen_padding_steps,
            self.simulator.config.workload_config.output_seqlen_padding_factors,
            self.simulator.config.workload_config.output_seqlen_padding_steps,
        )
        prefill_time_ns_per_stage = batch_metrics.prefill.stage_time_ns
        prefill_time_ns = prefill_time_ns_per_stage * self.num_pipeline_stages()
        prefill_energy_J = (
            batch_metrics.prefill.energy_per_chip_J * self.config.llm_config.num_chips
        )
        workload_config = self.simulator.config.workload_config
        if workload_config.enable_dvfs:
            from neusim.fleetsim.dvfs_scheduler import compute_dvfs_plan_for_batch

            metrics_server = self.simulator.metrics_server
            endpoint = self.simulator.llm_inference_endpoint
            if metrics_server is None or endpoint is None:
                raise RuntimeError("FleetSim DVFS requires initialized service objects")
            frozen_config = sim_backend.make_frozen_config_for_batch(
                self.config.llm_config,
                temp_requests,
                workload_config.input_seqlen_padding_factors,
                workload_config.input_seqlen_padding_steps,
                workload_config.output_seqlen_padding_factors,
                workload_config.output_seqlen_padding_steps,
            )
            dvfs_time_ns, dvfs_energy_J, _ = compute_dvfs_plan_for_batch(
                phase_metrics=batch_metrics.prefill,
                frozen_config=frozen_config,
                requests=requests,
                event_timestamp=event.timestamp,
                baseline_time_ns=prefill_time_ns,
                num_pipeline_stages=self.num_pipeline_stages(),
                num_chips=self.config.llm_config.num_chips,
                prefill_or_decode="prefill",
                num_iterations=1,
                metrics_server=metrics_server,
                request_queue=endpoint.prefill_request_queue,
                workload_config=workload_config,
            )
            if dvfs_energy_J < prefill_energy_J:
                prefill_time_ns = dvfs_time_ns
                prefill_time_ns_per_stage = dvfs_time_ns // self.num_pipeline_stages()
                prefill_energy_J = dvfs_energy_J
        prefill_cost_dollars = (
            cost_model.pipeline_batch_monetary_cost_from_stage_time_dollars(
                prefill_time_ns_per_stage, self.config.llm_config
            )
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


class LLMInferenceDecodeInstance(LLMInferenceInstanceBase):
    """A statically provisioned decode engine."""

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
        batch_metrics = sim_backend.run_inference_request_batch(
            self.config.llm_config,
            temp_requests,
            self.simulator.config.workload_config.input_seqlen_padding_factors,
            self.simulator.config.workload_config.input_seqlen_padding_steps,
            self.simulator.config.workload_config.output_seqlen_padding_factors,
            self.simulator.config.workload_config.output_seqlen_padding_steps,
        )
        decode_time_ns_per_stage = batch_metrics.decode.stage_time_ns
        decode_time_ns = decode_time_ns_per_stage * self.num_pipeline_stages()
        decode_energy_per_token_J = (
            batch_metrics.decode.energy_per_chip_J * self.config.llm_config.num_chips
        )
        workload_config = self.simulator.config.workload_config
        if workload_config.enable_dvfs:
            from neusim.fleetsim.dvfs_scheduler import compute_dvfs_plan_for_batch

            metrics_server = self.simulator.metrics_server
            endpoint = self.simulator.llm_inference_endpoint
            if metrics_server is None or endpoint is None:
                raise RuntimeError("FleetSim DVFS requires initialized service objects")
            frozen_config = sim_backend.make_frozen_config_for_batch(
                self.config.llm_config,
                temp_requests,
                workload_config.input_seqlen_padding_factors,
                workload_config.input_seqlen_padding_steps,
                workload_config.output_seqlen_padding_factors,
                workload_config.output_seqlen_padding_steps,
            )
            dvfs_time_ns, dvfs_energy_J, _ = compute_dvfs_plan_for_batch(
                phase_metrics=batch_metrics.decode,
                frozen_config=frozen_config,
                requests=requests,
                event_timestamp=event.timestamp,
                baseline_time_ns=decode_time_ns,
                num_pipeline_stages=self.num_pipeline_stages(),
                num_chips=self.config.llm_config.num_chips,
                prefill_or_decode="decode",
                num_iterations=event.num_iterations,
                metrics_server=metrics_server,
                request_queue=endpoint.decode_request_queue,
                workload_config=workload_config,
            )
            if dvfs_energy_J < decode_energy_per_token_J:
                decode_time_ns = dvfs_time_ns
                decode_time_ns_per_stage = dvfs_time_ns // self.num_pipeline_stages()
                decode_energy_per_token_J = dvfs_energy_J
        decode_cost_dollars = (
            cost_model.pipeline_batch_monetary_cost_from_stage_time_dollars(
                decode_time_ns_per_stage, self.config.llm_config
            )
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
            per_request_energy_J = decode_energy_per_token_J / batch_size
            per_request_cost_dollars = decode_cost_dollars / batch_size
            # Preserve the original per-token summation order exactly while
            # retaining only totals. A full Azure day otherwise materializes
            # hundreds of millions of Python list entries.
            for _ in range(event.num_iterations):
                request.decode_energy_J += per_request_energy_J
                request.decode_cost_dollars += per_request_cost_dollars

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


class LLMInferenceEndpoint(SimObject):
    """LLM inference endpoint with an immutable, statically configured vPod fleet."""

    def __init__(self, name: str, simulator: "NPUFleetSimulator"):
        super().__init__(name, simulator)
        self.prefill_vpods: dict[str, LLMInferencePrefillInstance] = {}
        self.decode_vpods: dict[str, LLMInferenceDecodeInstance] = {}
        self.prefill_request_queue: deque[LLMRequest] = deque()
        self.decode_request_queue: deque[LLMRequest] = deque()
        self.prefill_schedule_delay_ns = 0
        self.decode_schedule_delay_ns = 0

    def _build_static_llm_config(self, entry: StaticVPodEntry) -> LLMConfig:
        """Merge the model, chip, and fixed-vPod settings into one engine config."""
        base_model = self.simulator.config.workload_config.llm_config
        cluster_manager = self.simulator.cluster_manager
        assert cluster_manager is not None
        chip_config = cluster_manager.chip_configs[entry.npu_type]

        config_dict = base_model.model_dump()
        # This separate workload flag enables component-level V/f power tables
        # without also enabling the service-level DVFS scheduler.
        detailed_dvfs_power = (
            self.simulator.config.workload_config.enable_dvfs_power_model
        )
        config_dict.update(chip_config.model_dump())
        config_dict.update(
            {
                "name": entry.npu_type,
                "num_chips": entry.num_chips,
                "data_parallelism_degree": entry.dp,
                "tensor_parallelism_degree": entry.tp,
                "pipeline_parallelism_degree": entry.pp,
                "microbatch_size_ici": entry.batch_size,
                "global_batch_size": entry.batch_size * entry.pp,
                "microbatch_size_dcn": entry.batch_size * entry.pp,
                "input_seqlen": 32,
                "output_seqlen": 32,
                "enable_dvfs": detailed_dvfs_power,
            }
        )
        if isinstance(base_model, DeepSeekConfig):
            config_dict["expert_parallelism_degree"] = entry.ep
            return DeepSeekConfig.model_validate(config_dict)
        return LLMConfig.model_validate(config_dict)

    def initialize(self):
        super().initialize()
        allocation = self.simulator.config.workload_config.static_vpod_allocation
        if allocation is None:
            raise ValueError("static_vpod_allocation is required by FleetSim")

        prefill_config = self._build_static_llm_config(allocation.prefill)
        decode_config = self._build_static_llm_config(allocation.decode)

        for index in range(allocation.prefill.count):
            vpod = LLMInferencePrefillInstance(
                self.simulator,
                f"{self.name}_prefill_{index}",
                self.simulator.config.workload_config,
                prefill_config,
            )
            vpod.initialize()
            self.prefill_vpods[vpod.id] = vpod

        for index in range(allocation.decode.count):
            vpod = LLMInferenceDecodeInstance(
                self.simulator,
                f"{self.name}_decode_{index}",
                self.simulator.config.workload_config,
                decode_config,
            )
            vpod.initialize()
            self.decode_vpods[vpod.id] = vpod

        self.simulator.add_event_listener(
            LLMInferenceEngineReadyEvent.get_type_listener(
                self.on_instance_ready, priority=10
            )
        )
        self.simulator.add_event_listener(
            LLMInferencePrefillStartEvent.get_type_listener(
                self.on_prefill_start, priority=10
            )
        )
        self.simulator.add_event_listener(
            LLMInferencePrefillEndEvent.get_type_listener(
                self.on_prefill_end, priority=10
            )
        )
        self.simulator.add_event_listener(
            LLMInferenceDecodeIterationStartEvent.get_type_listener(
                self.on_decode_iteration_start, priority=10
            )
        )
        self.simulator.add_event_listener(
            LLMInferenceDecodeIterationEndEvent.get_type_listener(
                self.on_decode_iteration_end, priority=10
            )
        )
        self.simulator.add_event_listener(
            LLMInferenceRequestEnqueueEvent.get_type_listener(
                self.enqueue_prefill_request, priority=40
            )
        )
        self.simulator.add_event_listener(
            LLMInferenceEngineReadyEvent.get_type_listener(self.on_engine_ready)
        )
        self.simulator.add_event_listener(
            LLMInferencePrefillEndEvent.get_type_listener(
                self.prefill_end_and_enqueue_decode_request, priority=40
            )
        )
        self.simulator.add_event_listener(
            LLMInferenceDecodeIterationEndEvent.get_type_listener(
                self.decode_iteration_end_and_enqueue_next_iteration, priority=40
            )
        )

    @property
    def num_prefill_replicas(self) -> int:
        return len(self.prefill_vpods)

    @property
    def num_decode_replicas(self) -> int:
        return len(self.decode_vpods)

    def lookup_vpod_by_id(self, vpod_id: str) -> LLMInferenceInstanceBase | None:
        return self.prefill_vpods.get(vpod_id) or self.decode_vpods.get(vpod_id)

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
        if not self.prefill_request_queue:
            logging.debug("No prefill requests in the queue at %d.", timestamp)
            return
        available = [vpod for vpod in self.prefill_vpods.values() if vpod.ready]
        if not available:
            logging.debug("All prefill vPods are busy at %d.", timestamp)
            return
        self.prefill_request_queue = deque(
            sorted(self.prefill_request_queue, key=lambda request: request.input_seqlen)
        )
        self.dispatch_requests_to_vpods(
            timestamp, available, self.prefill_request_queue, "prefill"
        )

    def try_dispatch_decode_requests(self, timestamp: int):
        if not self.decode_request_queue:
            logging.debug("No decode requests in the queue at %d.", timestamp)
            return
        available = [vpod for vpod in self.decode_vpods.values() if vpod.ready]
        if not available:
            logging.debug("All decode vPods are busy at %d.", timestamp)
            return
        self.decode_request_queue = deque(
            sorted(self.decode_request_queue, key=lambda request: request.total_seqlen)
        )
        self.dispatch_requests_to_vpods(
            timestamp, available, self.decode_request_queue, "decode"
        )

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

    def on_engine_ready(self, event: LLMInferenceEngineReadyEvent):
        logging.debug(
            "Try to dispatch requests for ready worker %s at timestamp %d",
            event.worker_id,
            event.timestamp,
        )
        self.try_dispatch_prefill_requests(event.timestamp)
        self.try_dispatch_decode_requests(event.timestamp)

    def dump_simulation_stats(self):
        """Write per-vPod counters for the fixed allocation."""
        logging.info("%s simulation stats:", self.name)

        def serialize(vpod: LLMInferenceInstanceBase) -> dict[str, object]:
            return {
                "id": vpod.id,
                "npu_type": vpod.config.llm_config.name,
                "num_chips": vpod.config.llm_config.num_chips,
                "pcfg": util.get_pstr(vpod.config.llm_config),
                "num_processed_tokens": vpod.num_processed_tokens,
            }

        stats = {
            "prefill": [serialize(vpod) for vpod in self.prefill_vpods.values()],
            "decode": [serialize(vpod) for vpod in self.decode_vpods.values()],
        }
        output_path = os.path.join(
            self.simulator.config.output_dir, "static_vpod_stats.json"
        )
        with open(output_path, "w") as output_file:
            json.dump(stats, output_file, indent=4)
        logging.info("Dumped static vPod statistics to %s", output_path)
