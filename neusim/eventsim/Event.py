"""Event and listener primitives for discrete-event simulation."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from absl import logging

_EventType = TypeVar("_EventType", bound="Event")


class EventListener(Generic[_EventType]):
    """Invoke a callback for matching events.

    Lower ``priority`` values run first. ``metadata`` is intentionally opaque to
    EventSim and can be used by owners to find and remove groups of listeners.
    """

    __slots__ = ["event_type", "cond", "callback", "priority", "metadata"]

    def __init__(
        self,
        event_type: type[_EventType],
        cond: Callable[[_EventType], bool],
        callback: Callable[[_EventType], None],
        priority: int = 999,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.event_type = event_type
        self.cond = cond
        self.callback = callback
        self.priority = priority
        self.metadata = {} if metadata is None else metadata

    def __call__(self, event: _EventType) -> None:
        """Invoke the callback when this listener's condition is satisfied."""
        if self.cond(event):
            if logging.level_debug():
                logging.debug("%s called for %s", self.callback, event)
            self.callback(event)

    def __str__(self) -> str:
        return f"EventListener({self.cond}, {self.callback})"

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other: EventListener[Any]) -> bool:
        return self.priority < other.priority

    def __le__(self, other: EventListener[Any]) -> bool:
        return self.priority <= other.priority

    def __gt__(self, other: EventListener[Any]) -> bool:
        return self.priority > other.priority

    def __ge__(self, other: EventListener[Any]) -> bool:
        return self.priority >= other.priority


class Event:
    """Base event ordered by its integer simulation timestamp."""

    __slots__ = ["timestamp", "uuid"]

    def __init__(self, timestamp: int) -> None:
        self.timestamp = timestamp
        self.uuid = uuid.uuid4().hex

    def __str__(self) -> str:
        return f"Event {self.uuid} ({type(self)}) at {self.timestamp}: "

    def __repr__(self) -> str:
        return str(self)

    def __lt__(self, other: Event) -> bool:
        return self.timestamp < other.timestamp

    def __le__(self, other: Event) -> bool:
        return self.timestamp <= other.timestamp

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return self.timestamp == other.timestamp

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Event):
            return NotImplemented
        return self.timestamp != other.timestamp

    def __gt__(self, other: Event) -> bool:
        return self.timestamp > other.timestamp

    def __ge__(self, other: Event) -> bool:
        return self.timestamp >= other.timestamp

    @classmethod
    def get_type_listener(
        cls: type[_EventType],
        callback: Callable[[_EventType], None],
        priority: int = 999,
        metadata: dict[str, Any] | None = None,
    ) -> EventListener[_EventType]:
        """Create a listener for instances of this event class or subclasses."""
        return EventListener(
            cls,
            lambda event: isinstance(event, cls),
            callback,
            priority=priority,
            metadata=metadata,
        )

    @staticmethod
    def get_universal_listener(
        callback: Callable[[Event], None],
        priority: int = 999,
        metadata: dict[str, Any] | None = None,
    ) -> EventListener[Event]:
        """Create a listener that receives every processed event."""
        return EventListener(
            Event,
            lambda event: True,
            callback,
            priority=priority,
            metadata=metadata,
        )
