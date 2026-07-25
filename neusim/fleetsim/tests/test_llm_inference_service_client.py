from types import SimpleNamespace
from unittest.mock import patch

import pytest

from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
)
from neusim.eventsim.EventSim import EventSimulator
from neusim.fleetsim.LLMInferenceEvents import LLMInferenceRequestEnqueueEvent
from neusim.fleetsim.LLMInferenceServiceClient import LLMInferenceServiceClient
from neusim.fleetsim.LoadGenerator import LLMRequest, LoadGenerator


def _make_client(generated_requests, **limits):
    simulator = EventSimulator("client-test")
    simulator.config = SimpleNamespace(
        workload_config=LLMInferenceWorkloadConfig(**limits)
    )
    with patch.object(LoadGenerator, "generate", return_value=generated_requests):
        client = LLMInferenceServiceClient(simulator)
    return client, simulator


@pytest.mark.parametrize(
    ("limits", "expected_indexes"),
    [
        ({"max_num_requests": 2}, [0, 1]),
        ({"max_timestamp": 20}, [0, 1, 2]),
        ({"max_num_requests": 2, "max_timestamp": 20}, [0, 1]),
    ],
)
def test_constructor_exposes_only_requests_that_will_be_enqueued(
    limits, expected_indexes
):
    generated = [LLMRequest(32, 4, timestamp) for timestamp in (0, 10, 20, 30)]

    client, _ = _make_client(generated, **limits)

    assert client.requests == [generated[index] for index in expected_indexes]


def test_constructor_filters_timestamp_before_applying_request_count():
    generated = [LLMRequest(32, 4, timestamp) for timestamp in (20, 0, 10)]

    client, _ = _make_client(
        generated, max_num_requests=2, max_timestamp=10
    )

    assert client.requests == [generated[1], generated[2]]


def test_initialize_enqueues_exactly_constructor_visible_subset():
    generated = [LLMRequest(32, 4, timestamp) for timestamp in (0, 10, 20)]
    client, simulator = _make_client(
        generated, max_num_requests=2, max_timestamp=10
    )

    client.initialize()

    events = [simulator.get(), simulator.get()]
    assert all(isinstance(event, LLMInferenceRequestEnqueueEvent) for event in events)
    assert [event.request for event in events] == client.requests
    assert client.total_num_enqueued_requests == 2
    assert client.next_request_index == 2
    assert simulator.event_queue_length() == 0
