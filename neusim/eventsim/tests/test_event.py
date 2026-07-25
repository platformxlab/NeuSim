"""Tests for Event and EventListener primitives."""

from neusim.eventsim import Event, EventListener


class WorkEvent(Event):
    pass


class ChildWorkEvent(WorkEvent):
    pass


def test_events_compare_by_timestamp() -> None:
    early = Event(10)
    same_time = Event(10)
    late = Event(20)

    assert early == same_time
    assert early != late
    assert early <= same_time
    assert early < late
    assert late > early
    assert late >= same_time


def test_events_have_unique_ids_and_informative_representations() -> None:
    first = WorkEvent(7)
    second = WorkEvent(7)

    assert first.uuid != second.uuid
    assert first.uuid in str(first)
    assert "WorkEvent" in repr(first)
    assert "at 7" in str(first)


def test_listener_checks_condition_and_preserves_metadata() -> None:
    received: list[int] = []
    metadata = {"owner": "test"}
    listener = EventListener(
        WorkEvent,
        lambda event: event.timestamp % 2 == 0,
        lambda event: received.append(event.timestamp),
        metadata=metadata,
    )

    listener(WorkEvent(1))
    listener(WorkEvent(2))

    assert received == [2]
    assert listener.metadata is metadata


def test_listener_comparison_uses_priority() -> None:
    high_priority = WorkEvent.get_type_listener(lambda event: None, priority=1)
    low_priority = WorkEvent.get_type_listener(lambda event: None, priority=20)

    assert high_priority < low_priority
    assert high_priority <= low_priority
    assert low_priority > high_priority
    assert low_priority >= high_priority


def test_type_and_universal_listener_factories_match_subclasses() -> None:
    type_listener = WorkEvent.get_type_listener(lambda event: None)
    universal_listener = Event.get_universal_listener(lambda event: None)

    assert type_listener.event_type is WorkEvent
    assert type_listener.cond(WorkEvent(1))
    assert type_listener.cond(ChildWorkEvent(1))
    assert not type_listener.cond(Event(1))

    assert universal_listener.event_type is Event
    assert universal_listener.cond(Event(1))
    assert universal_listener.cond(ChildWorkEvent(1))
