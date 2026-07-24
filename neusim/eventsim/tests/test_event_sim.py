"""Behavioral tests for the EventSimulator."""

from collections.abc import Iterator

import pytest

from neusim.eventsim import Event, EventListener, EventSimulator, PriorityQueue


class NamedEvent(Event):
    __slots__ = ["name"]

    def __init__(self, timestamp: int, name: str) -> None:
        super().__init__(timestamp)
        self.name = name


class ChildNamedEvent(NamedEvent):
    pass


class OtherEvent(Event):
    pass


def test_priority_queue_returns_smallest_item_first() -> None:
    queue: PriorityQueue[int] = PriorityQueue()
    for value in [5, 1, 3, 2, 4]:
        queue.put(value)

    assert [queue.get() for _ in range(5)] == [1, 2, 3, 4, 5]
    assert len(queue) == 0


def test_events_run_in_timestamp_order_and_advance_simulation_time() -> None:
    simulator = EventSimulator()
    received: list[tuple[str, int]] = []
    simulator.add_event_listener(
        Event.get_universal_listener(
            lambda event: received.append((event.name, simulator.timestamp))  # type: ignore[attr-defined]
        )
    )
    for event in [
        NamedEvent(30, "last"),
        NamedEvent(10, "first"),
        NamedEvent(20, "middle"),
    ]:
        simulator.put(event)

    simulator.run()

    assert received == [("first", 10), ("middle", 20), ("last", 30)]
    assert simulator.timestamp == 30
    assert simulator.event_queue_length() == 0


def test_equal_timestamp_events_run_in_insertion_order() -> None:
    simulator = EventSimulator()
    received: list[str] = []
    simulator.add_event_listener(
        NamedEvent.get_type_listener(lambda event: received.append(event.name))
    )

    for name in ["first", "second", "third"]:
        simulator.put(NamedEvent(10, name))

    simulator.run()

    assert received == ["first", "second", "third"]


def test_listener_conditions_and_global_priority_order() -> None:
    simulator = EventSimulator()
    callbacks: list[str] = []

    simulator.add_event_listener(
        EventListener(
            NamedEvent,
            lambda event: event.name == "wanted",
            lambda event: callbacks.append("low"),
            priority=20,
        )
    )
    simulator.add_event_listener(
        Event.get_universal_listener(
            lambda event: callbacks.append("universal"), priority=10
        )
    )
    simulator.add_event_listener(
        NamedEvent.get_type_listener(
            lambda event: callbacks.append("high"), priority=0
        )
    )

    simulator.put(NamedEvent(1, "ignored-by-condition"))
    simulator.put(NamedEvent(2, "wanted"))
    simulator.run()

    assert callbacks == [
        "high",
        "universal",
        "high",
        "universal",
        "low",
    ]


def test_base_and_universal_listeners_receive_subclass_events() -> None:
    """Regression: dispatch must not look up only the concrete event type."""
    simulator = EventSimulator()
    callbacks: list[str] = []
    simulator.add_event_listener(
        Event.get_universal_listener(
            lambda event: callbacks.append("universal"), priority=30
        )
    )
    simulator.add_event_listener(
        NamedEvent.get_type_listener(
            lambda event: callbacks.append("base"), priority=20
        )
    )
    simulator.add_event_listener(
        ChildNamedEvent.get_type_listener(
            lambda event: callbacks.append("concrete"), priority=10
        )
    )

    simulator.put(ChildNamedEvent(1, "child"))
    simulator.run()

    assert callbacks == ["concrete", "base", "universal"]


def test_unrelated_event_listener_is_not_called() -> None:
    simulator = EventSimulator()
    callbacks: list[Event] = []
    simulator.add_event_listener(
        OtherEvent.get_type_listener(lambda event: callbacks.append(event))
    )

    simulator.put(NamedEvent(1, "named"))
    simulator.run()

    assert callbacks == []


def test_listener_removal_uses_metadata_criteria() -> None:
    simulator = EventSimulator()
    callbacks: list[str] = []
    simulator.add_event_listener(
        NamedEvent.get_type_listener(
            lambda event: callbacks.append("remove"),
            metadata={"owner": "temporary"},
        )
    )
    simulator.add_event_listener(
        NamedEvent.get_type_listener(
            lambda event: callbacks.append("keep"), metadata={"owner": "persistent"}
        )
    )

    assert simulator.num_event_listeners() == 2
    simulator.remove_event_listeners_by_criteria(
        lambda listener: listener.metadata.get("owner") == "temporary"
    )
    assert simulator.num_event_listeners() == 1

    simulator.put(NamedEvent(1, "event"))
    simulator.run()
    assert callbacks == ["keep"]


def test_listener_changes_during_callback_apply_to_next_event() -> None:
    simulator = EventSimulator()
    callbacks: list[str] = []

    replacement = NamedEvent.get_type_listener(
        lambda event: callbacks.append("replacement"), metadata={"generation": 2}
    )

    def replace_listeners(event: NamedEvent) -> None:
        callbacks.append("replacer")
        simulator.remove_event_listeners_by_criteria(
            lambda listener: listener.metadata.get("generation") == 1
        )
        simulator.add_event_listener(replacement)

    simulator.add_event_listener(
        NamedEvent.get_type_listener(
            replace_listeners, priority=0, metadata={"generation": 1}
        )
    )
    simulator.add_event_listener(
        NamedEvent.get_type_listener(
            lambda event: callbacks.append("original"),
            priority=1,
            metadata={"generation": 1},
        )
    )
    simulator.put(NamedEvent(1, "first"))
    simulator.put(NamedEvent(2, "second"))

    simulator.run()

    assert callbacks == ["replacer", "original", "replacement"]


def test_callback_can_schedule_followup_event() -> None:
    simulator = EventSimulator()
    callbacks: list[tuple[str, int]] = []

    def schedule_followup(event: NamedEvent) -> None:
        callbacks.append(("initial", simulator.timestamp))
        simulator.put(OtherEvent(event.timestamp + 5))

    simulator.add_event_listener(NamedEvent.get_type_listener(schedule_followup))
    simulator.add_event_listener(
        OtherEvent.get_type_listener(
            lambda event: callbacks.append(("followup", simulator.timestamp))
        )
    )
    simulator.put(NamedEvent(10, "initial"))

    simulator.run()

    assert callbacks == [("initial", 10), ("followup", 15)]
    assert simulator.timestamp == 15


def test_scheduling_in_the_past_is_rejected_but_current_time_is_allowed() -> None:
    simulator = EventSimulator()
    simulator.put(Event(5))
    assert simulator.get() is not None

    with pytest.raises(AssertionError, match="must be greater"):
        simulator.put(Event(4))

    simulator.put(Event(5))
    assert simulator.event_queue_length() == 1


def test_cancel_by_event_or_uuid_handles_equal_timestamps() -> None:
    simulator = EventSimulator()
    received: list[str] = []
    simulator.add_event_listener(
        NamedEvent.get_type_listener(lambda event: received.append(event.name))
    )
    keep = NamedEvent(10, "keep")
    cancel_by_object = NamedEvent(10, "object")
    cancel_by_uuid = NamedEvent(10, "uuid")
    for event in [keep, cancel_by_object, cancel_by_uuid]:
        simulator.put(event)

    assert simulator.cancel(cancel_by_object)
    assert not simulator.cancel(cancel_by_object)
    assert simulator.cancel(cancel_by_uuid.uuid)
    assert not simulator.cancel("missing-uuid")
    with pytest.raises(TypeError):
        simulator.cancel(123)  # type: ignore[arg-type]

    simulator.run()
    assert received == ["keep"]


def test_queue_breakdown_reflects_scheduling_and_canceling() -> None:
    simulator = EventSimulator()
    named = NamedEvent(1, "named")
    simulator.put(named)
    simulator.put(NamedEvent(2, "another"))
    simulator.put(OtherEvent(3))

    assert simulator.get_event_queue_breakdown_snapshot() == {
        "NamedEvent": 2,
        "OtherEvent": 1,
    }
    simulator.cancel(named)
    assert simulator.get_event_queue_breakdown_snapshot() == {
        "NamedEvent": 1,
        "OtherEvent": 1,
    }


def test_max_events_stops_and_run_can_resume() -> None:
    simulator = EventSimulator()
    received: list[int] = []
    simulator.add_event_listener(
        Event.get_universal_listener(lambda event: received.append(event.timestamp))
    )
    for timestamp in [1, 2, 3]:
        simulator.put(Event(timestamp))

    simulator.run(max_events=2)
    assert received == [1, 2]
    assert simulator.timestamp == 2
    assert simulator.event_queue_length() == 1

    simulator.run(max_events=0)
    assert received == [1, 2]
    assert simulator.event_queue_length() == 1

    simulator.run()
    assert received == [1, 2, 3]
    assert simulator.timestamp == 3
    assert simulator.event_queue_length() == 0


def test_get_and_run_are_safe_with_an_empty_queue() -> None:
    simulator = EventSimulator()

    assert simulator.get() is None
    simulator.run()
    assert simulator.timestamp == 0


def test_profiling_aggregates_elapsed_time_by_concrete_event_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulator = EventSimulator("profile-test")
    simulator.enable_profile = True
    simulator.put(NamedEvent(1, "named"))
    simulator.put(OtherEvent(2))

    clock_values: Iterator[float] = iter([1.0, 1.01, 2.0, 2.02])
    monkeypatch.setattr(
        "neusim.eventsim.EventSim.time.perf_counter", lambda: next(clock_values)
    )
    summaries: list[
        tuple[str, dict[str, float], dict[str, int], int]
    ] = []

    def capture_summary(
        label: str,
        cumulative_time: dict[str, float],
        count: dict[str, int],
        top_n: int = 0,
    ) -> None:
        summaries.append((label, cumulative_time.copy(), count.copy(), top_n))

    monkeypatch.setattr(simulator, "_print_profile_summary", capture_summary)

    simulator.run()

    assert len(summaries) == 1
    label, cumulative_time, count, top_n = summaries[0]
    assert label == "Final event profiling summary"
    assert cumulative_time == pytest.approx({"NamedEvent": 0.01, "OtherEvent": 0.02})
    assert count == {"NamedEvent": 1, "OtherEvent": 1}
    assert top_n == 0
