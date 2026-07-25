"""Tests for FleetSim's deterministic output-length prediction routing."""

from collections import deque
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import neusim.fleetsim.vPodAutoScaler_lib as autoscaler_lib
from neusim.configs.systems.NPUFleetConfig import NPUFleetConfig
from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
)
from neusim.fleetsim.LLMInferenceEndpoint import (
    LLMInferenceEndpoint,
    select_decode_config_for_output_prediction,
)
from neusim.fleetsim.LoadGenerator import LLMRequest


def _decoding_request(output_seqlen: int, current_step: int) -> LLMRequest:
    request = LLMRequest(input_seqlen=32, output_seqlen=output_seqlen)
    request.current_decode_step = current_step
    return request


def _routing_configs(config: NPUFleetConfig):
    model = config.workload_config.llm_config
    correct = model.model_copy(update={"name": "5p", "num_chips": 8})
    too_short = model.model_copy(update={"name": "4", "num_chips": 4})
    longer_a = model.model_copy(update={"name": "6e", "num_chips": 16})
    longer_b = model.model_copy(update={"name": "5e", "num_chips": 32})
    return correct, too_short, longer_a, longer_b


def _dispatch_group_with_accuracy(
    monkeypatch: pytest.MonkeyPatch,
    accuracy: float,
    *,
    exact_deployed_match: bool,
) -> str:
    """Run one request through the real MPA grouping/dispatch path."""
    fleet_config = NPUFleetConfig()
    fleet_config.workload_config.output_prediction_accuracy = accuracy
    base = fleet_config.workload_config.llm_config
    correct_template = base.model_copy(
        update={
            "name": "5p",
            "num_chips": 8,
            "input_seqlen": 32,
            "output_seqlen": 1024,
        }
    )
    wrong_template = base.model_copy(
        update={
            "name": "6e",
            "num_chips": 16,
            "input_seqlen": 32,
            "output_seqlen": 4096,
        }
    )
    deployed_correct = correct_template.model_copy()
    deployed_wrong = wrong_template.model_copy()

    # The endpoint normalizes deployed configs to (32, 32). The autoscaler
    # lookup still returns configs carrying the lookup request's real sequence
    # lengths, so matching must use stable hardware/parallelism identity.
    lookup_config = (
        correct_template.model_copy()
        if exact_deployed_match
        else base.model_copy(
            update={
                "name": "4",
                "num_chips": 4,
                "input_seqlen": 32,
                "output_seqlen": 1024,
            }
        )
    )
    seqlen_pair = (32, 1024)
    monkeypatch.setattr(
        autoscaler_lib,
        "get_seqlen_to_configs_mapping",
        lambda pairs, _config, mode: {
            pair: [lookup_config] for pair in pairs if mode == "decode"
        },
    )
    ranges = {
        "5p": (32, 1200),
        "6e": (1201, 5000),
    }
    monkeypatch.setattr(
        autoscaler_lib,
        "get_seqlen_range_for_config",
        lambda _pairs, config, _fleet_config, mode: (
            ranges[config.name] if mode == "decode" else (-1, -1)
        ),
    )

    autoscaler = SimpleNamespace(
        get_seqlens_for_config=lambda pairs, config: (
            autoscaler_lib.get_seqlens_for_config(
                pairs, config, fleet_config, "decode"
            )
        ),
        config_to_seqlen_range_fallback={},
    )
    endpoint = object.__new__(LLMInferenceEndpoint)
    endpoint.simulator = SimpleNamespace(
        config=fleet_config,
        name="prediction-routing-test",
    )
    endpoint.decode_autoscaler = autoscaler
    endpoint.decode_request_queue = deque()

    correct_vpod = SimpleNamespace(
        config=SimpleNamespace(llm_config=deployed_correct)
    )
    wrong_vpod = SimpleNamespace(config=SimpleNamespace(llm_config=deployed_wrong))
    routed_groups: list[str] = []

    def record_dispatch(_timestamp, vpods, requests, phase):
        assert phase == "decode"
        routed_groups.extend(vpods[0].config.llm_config.name for _ in requests)
        requests.clear()

    endpoint.dispatch_requests_to_vpods = record_dispatch
    request = _decoding_request(output_seqlen=seqlen_pair[1], current_step=1)
    request.id = "prediction-routing-request"
    endpoint.decode_request_queue.append(request)

    endpoint.dispatch_requests_to_vpods_mpa(
        10,
        [correct_vpod, wrong_vpod],
        "decode",
    )

    assert not endpoint.decode_request_queue
    assert len(routed_groups) == 1
    return routed_groups[0]


def test_output_prediction_config_defaults_and_validation() -> None:
    workload = LLMInferenceWorkloadConfig()

    assert workload.output_prediction_accuracy == 1.0
    assert workload.output_prediction_seed == 42
    for invalid_accuracy in (-0.01, 1.01):
        with pytest.raises(ValidationError, match="output_prediction_accuracy"):
            LLMInferenceWorkloadConfig(output_prediction_accuracy=invalid_accuracy)


def test_default_prediction_always_preserves_correct_decode_group() -> None:
    config = NPUFleetConfig()
    correct, _, longer_a, _ = _routing_configs(config)
    request = _decoding_request(output_seqlen=1024, current_step=500)
    request.id = "stable-request"
    ranges = {correct: (32, 1200), longer_a: (32, 2400)}

    assert (
        select_decode_config_for_output_prediction(
            request, correct, ranges, config.workload_config
        )
        is correct
    )


def test_misprediction_uses_only_a_distinct_group_long_enough_for_request() -> None:
    config = NPUFleetConfig()
    config.workload_config.output_prediction_accuracy = 0.0
    correct, too_short, longer_a, _ = _routing_configs(config)
    request = _decoding_request(output_seqlen=1024, current_step=500)
    request.id = "stable-request"
    ranges = {
        correct: (32, 1200),
        too_short: (32, 500),
        longer_a: (32, 2400),
    }

    selected = select_decode_config_for_output_prediction(
        request, correct, ranges, config.workload_config
    )

    assert selected is longer_a
    assert ranges[selected][1] >= request.input_seqlen + request.output_seqlen


def test_prediction_is_fixed_for_request_lifetime() -> None:
    config = NPUFleetConfig()
    config.workload_config.output_prediction_accuracy = 0.0
    config.workload_config.output_prediction_seed = 91
    correct, _, longer_a, longer_b = _routing_configs(config)
    request = _decoding_request(output_seqlen=1024, current_step=1)
    request.id = "17"
    ranges = {
        correct: (32, 1200),
        longer_a: (32, 2400),
        longer_b: (32, 4800),
    }

    first = select_decode_config_for_output_prediction(
        request, correct, ranges, config.workload_config
    )
    reversed_ranges = dict(reversed(list(ranges.items())))
    repeated = select_decode_config_for_output_prediction(
        request, correct, reversed_ranges, config.workload_config
    )
    assert first is repeated

    request.current_decode_step = 3
    assert (
        select_decode_config_for_output_prediction(
            request, correct, ranges, config.workload_config
        )
        is first
    )
    request.current_decode_step = 5
    assert (
        select_decode_config_for_output_prediction(
            request, correct, ranges, config.workload_config
        )
        is first
    )


def test_accuracy_is_a_deterministic_bernoulli_rate() -> None:
    config = NPUFleetConfig()
    config.workload_config.output_prediction_accuracy = 0.6
    config.workload_config.output_prediction_seed = 123
    correct, _, longer_a, _ = _routing_configs(config)
    request = _decoding_request(output_seqlen=1024, current_step=1)
    ranges = {correct: (32, 1200), longer_a: (32, 2400)}

    correct_count = 0
    num_requests = 10_000
    for request_index in range(num_requests):
        request.id = str(request_index)
        selected = select_decode_config_for_output_prediction(
            request, correct, ranges, config.workload_config
        )
        correct_count += selected is correct

    assert correct_count / num_requests == pytest.approx(0.6, abs=0.015)


def test_misprediction_falls_back_when_no_legal_wrong_group_exists() -> None:
    config = NPUFleetConfig()
    config.workload_config.output_prediction_accuracy = 0.0
    correct, too_short, _, _ = _routing_configs(config)
    request = _decoding_request(output_seqlen=1024, current_step=500)
    request.id = "stable-request"
    ranges = {correct: (32, 1200), too_short: (32, 500)}

    assert (
        select_decode_config_for_output_prediction(
            request, correct, ranges, config.workload_config
        )
        is correct
    )


@pytest.mark.parametrize("exact_deployed_match", [True, False])
def test_mpa_dispatch_routes_prediction_miss_to_a_different_group(
    monkeypatch: pytest.MonkeyPatch,
    exact_deployed_match: bool,
) -> None:
    correct_group = _dispatch_group_with_accuracy(
        monkeypatch,
        1.0,
        exact_deployed_match=exact_deployed_match,
    )
    missed_group = _dispatch_group_with_accuracy(
        monkeypatch,
        0.0,
        exact_deployed_match=exact_deployed_match,
    )

    assert correct_group == "5p"
    assert missed_group == "6e"
    assert missed_group != correct_group
