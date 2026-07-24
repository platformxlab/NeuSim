from collections import deque
from types import SimpleNamespace

import pytest

import neusim.fleetsim.dvfs_scheduler as dvfs_scheduler
import neusim.fleetsim.LLMInferenceEndpoint as endpoint
import neusim.npusim.frontend.query_results_helper_lib as results_lib
from neusim.configs.systems.NPUFleetConfig import NPUFleetConfig
from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
    StaticVPodAllocation,
    StaticVPodEntry,
)
from neusim.fleetsim.LLMInferenceEndpoint import (
    LLMInferenceDecodeInstance,
    LLMInferencePrefillInstance,
    get_decode_schedule_num_iterations,
)
from neusim.fleetsim.LLMInferenceEvents import (
    LLMInferenceDecodeIterationEndEvent,
    LLMInferenceDecodeIterationStartEvent,
    LLMInferenceEngineReadyEvent,
    LLMInferencePrefillEndEvent,
    LLMInferencePrefillStartEvent,
)
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.fleetsim.npusim_backend_interface import (
    InferenceBatchMetrics,
    PhaseMetrics,
)


class _RecordingSimulator:
    def __init__(self, config: NPUFleetConfig):
        self.config = config
        self.timestamp = 0
        self.events = []

    def put(self, event) -> None:
        self.events.append(event)


def _pipeline_test_setup():
    static_entry = StaticVPodEntry(
        count=1,
        npu_type="4",
        num_chips=8,
        batch_size=8,
        tp=8,
    )
    config = NPUFleetConfig(
        workload_config=LLMInferenceWorkloadConfig(
            static_vpod_allocation=StaticVPodAllocation(
                prefill=static_entry,
                decode=static_entry,
            )
        )
    )
    pod_config = config.workload_config.llm_config.model_copy(
        update={
            "name": "4",
            "num_chips": 8,
            "microbatch_size_ici": 8,
            "pipeline_parallelism_degree": 4,
            "input_seqlen": 32,
            "output_seqlen": 8,
        }
    )
    simulator = _RecordingSimulator(config)
    return config, pod_config, simulator


def _stage_batch_cost(stage_time_ns: int, pod_config) -> float:
    return (
        stage_time_ns
        / 1e9
        * results_lib.VERSION_TO_COST[pod_config.name]
        / 3600
        * pod_config.num_chips
    )


def _decoding_request(output_seqlen: int, current_step: int) -> LLMRequest:
    request = LLMRequest(input_seqlen=32, output_seqlen=output_seqlen)
    request.mark_prefill_started(0)
    request.mark_prefill_finished(1)
    request.mark_decode_iteration_started(1)
    if current_step > 1:
        request.mark_decode_iteration_finished(2, current_step - 1)
    return request


def test_decode_schedule_uses_exact_partial_final_chunk() -> None:
    request = _decoding_request(output_seqlen=6, current_step=1)

    assert get_decode_schedule_num_iterations([request], 4, 256) == 4
    request.mark_decode_iteration_finished(10, 4)
    assert get_decode_schedule_num_iterations([request], 4, 256) == 1


def test_decode_schedule_caps_batch_at_shortest_remaining_request() -> None:
    nearly_finished = _decoding_request(output_seqlen=6, current_step=5)
    longer = _decoding_request(output_seqlen=10, current_step=5)

    assert get_decode_schedule_num_iterations([nearly_finished, longer], 4, 256) == 1


def test_decode_schedule_honors_maximum_chunk_size() -> None:
    request = _decoding_request(output_seqlen=20_000, current_step=1)

    assert get_decode_schedule_num_iterations([request], 4, 256) == 256


def test_decode_schedule_rejects_empty_or_completed_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        get_decode_schedule_num_iterations([], 4, 256)

    finished = _decoding_request(output_seqlen=2, current_step=2)
    with pytest.raises(ValueError, match="complete"):
        get_decode_schedule_num_iterations([finished], 4, 256)


def test_prefill_endpoint_uses_pipeline_latency_but_one_stage_cost(
    monkeypatch,
) -> None:
    config, pod_config, simulator = _pipeline_test_setup()
    phase = PhaseMetrics(stage_time_ns=30, energy_per_chip_J=6.0)
    metrics = InferenceBatchMetrics(
        prefill=phase,
        decode=phase,
    )
    monkeypatch.setattr(
        endpoint.sim_backend,
        "run_inference_request_batch",
        lambda *_args, **_kwargs: metrics,
    )
    instance = LLMInferencePrefillInstance(
        simulator, "prefill", config.workload_config, pod_config
    )
    requests = [
        LLMRequest(input_seqlen=32, output_seqlen=8, timestamp=100) for _ in range(2)
    ]

    instance.execute_batch_requests(
        LLMInferencePrefillStartEvent(requests, 100, instance.id)
    )

    stage_time_ns = 30
    pipeline_latency_ns = stage_time_ns * pod_config.pipeline_parallelism_degree
    expected_request_cost = _stage_batch_cost(stage_time_ns, pod_config) / len(requests)
    end_event = next(
        event
        for event in simulator.events
        if isinstance(event, LLMInferencePrefillEndEvent)
    )
    ready_event = next(
        event
        for event in simulator.events
        if isinstance(event, LLMInferenceEngineReadyEvent)
    )

    assert end_event.timestamp == 100 + pipeline_latency_ns
    assert ready_event.timestamp == 100 + stage_time_ns
    for request in requests:
        assert request.ideal_TTFT_ns == pipeline_latency_ns
        assert request.prefill_cost_dollars == pytest.approx(expected_request_cost)
        assert request.prefill_cost_dollars != pytest.approx(
            expected_request_cost * pod_config.pipeline_parallelism_degree
        )


def test_prefill_endpoint_applies_static_dvfs_plan(monkeypatch) -> None:
    config, pod_config, simulator = _pipeline_test_setup()
    config.workload_config.enable_dvfs_power_model = True
    config.workload_config.enable_dvfs = True
    config.workload_config.slo_json_path = "unused-in-unit-test.json"
    phase = PhaseMetrics(stage_time_ns=30, energy_per_chip_J=6.0)
    metrics = InferenceBatchMetrics(prefill=phase, decode=phase)
    monkeypatch.setattr(
        endpoint.sim_backend,
        "run_inference_request_batch",
        lambda *_args, **_kwargs: metrics,
    )
    monkeypatch.setattr(
        endpoint.sim_backend,
        "make_frozen_config_for_batch",
        lambda *_args, **_kwargs: object(),
    )
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 80, 24.0, 0.1

    monkeypatch.setattr(dvfs_scheduler, "compute_dvfs_plan_for_batch", fake_plan)
    simulator.metrics_server = SimpleNamespace()
    simulator.llm_inference_endpoint = SimpleNamespace(
        prefill_request_queue=deque(), decode_request_queue=deque()
    )
    instance = LLMInferencePrefillInstance(
        simulator, "prefill-dvfs", config.workload_config, pod_config
    )
    requests = [
        LLMRequest(input_seqlen=32, output_seqlen=8, timestamp=100) for _ in range(2)
    ]

    instance.execute_batch_requests(
        LLMInferencePrefillStartEvent(requests, 100, instance.id)
    )

    end_event = next(
        event
        for event in simulator.events
        if isinstance(event, LLMInferencePrefillEndEvent)
    )
    ready_event = next(
        event
        for event in simulator.events
        if isinstance(event, LLMInferenceEngineReadyEvent)
    )
    assert end_event.timestamp == 180
    assert ready_event.timestamp == 120
    assert captured["phase_metrics"] is phase
    assert captured["prefill_or_decode"] == "prefill"
    for request in requests:
        assert request.prefill_energy_J == 12.0
        assert request.ideal_TTFT_ns == 80


def test_decode_endpoint_uses_pipeline_latency_but_one_stage_cost(
    monkeypatch,
) -> None:
    config, pod_config, simulator = _pipeline_test_setup()
    phase = PhaseMetrics(stage_time_ns=14, energy_per_chip_J=4.0)
    metrics = InferenceBatchMetrics(
        prefill=phase,
        decode=phase,
    )
    monkeypatch.setattr(
        endpoint.sim_backend,
        "run_inference_request_batch",
        lambda *_args, **_kwargs: metrics,
    )
    instance = LLMInferenceDecodeInstance(
        simulator, "decode", config.workload_config, pod_config
    )
    requests = [
        LLMRequest(input_seqlen=32, output_seqlen=8, timestamp=0) for _ in range(2)
    ]
    for request in requests:
        request.mark_prefill_started(0)
        request.mark_prefill_finished(200)

    num_iterations = 3
    instance.execute_batch_requests(
        LLMInferenceDecodeIterationStartEvent(
            requests, 200, instance.id, num_iterations
        )
    )

    stage_time_ns = 14
    pipeline_latency_ns = stage_time_ns * pod_config.pipeline_parallelism_degree
    expected_token_cost = _stage_batch_cost(stage_time_ns, pod_config) / len(requests)
    end_event = next(
        event
        for event in simulator.events
        if isinstance(event, LLMInferenceDecodeIterationEndEvent)
    )
    ready_event = next(
        event
        for event in simulator.events
        if isinstance(event, LLMInferenceEngineReadyEvent)
    )

    assert end_event.timestamp == 200 + pipeline_latency_ns * num_iterations
    assert ready_event.timestamp == 200 + stage_time_ns
    for request in requests:
        assert request.ideal_TPOT_ns == pipeline_latency_ns
        assert request.decode_cost_dollars == pytest.approx(
            expected_token_cost * num_iterations
        )
        assert request.decode_cost_dollars != pytest.approx(
            expected_token_cost
            * pod_config.pipeline_parallelism_degree
            * num_iterations
        )
