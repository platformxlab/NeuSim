"""Priority-queue based discrete-event simulator."""

from __future__ import annotations

import bisect
import heapq
import itertools
import json
import time
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import tqdm
from absl import logging

from neusim.eventsim.Event import Event, EventListener

_QueueElement = TypeVar("_QueueElement")


class PriorityQueue(Generic[_QueueElement]):
    """A minimal, non-locking priority queue backed by :mod:`heapq`."""

    def __init__(self) -> None:
        self._queue: list[tuple[_QueueElement, int]] = []
        self._sequence = itertools.count()

    def __len__(self) -> int:
        return len(self._queue)

    def put(self, item: _QueueElement) -> None:
        heapq.heappush(self._queue, (item, next(self._sequence)))

    def get(self) -> _QueueElement:
        return heapq.heappop(self._queue)[0]

    def remove(
        self, criteria: Callable[[_QueueElement], bool]
    ) -> _QueueElement | None:
        """Remove and return the first element matching ``criteria``."""
        for index, (item, _sequence) in enumerate(self._queue):
            if criteria(item):
                removed, _sequence = self._queue.pop(index)
                heapq.heapify(self._queue)
                return removed
        return None


class EventSimulator:
    """Basic event-driven simulator."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or "EventSimulator"
        self.queue: PriorityQueue[Event] = PriorityQueue()
        self.event_listeners: dict[type[Event], list[EventListener[Any]]] = {}
        self.timestamp = 0
        self.enable_profile = False

        self.progress_bar_total = 0
        self.progress_bar_update: Callable[[tqdm.tqdm], None] | None = None

    def __str__(self) -> str:
        return f"EventQueue with length {len(self.queue)}"

    def put(self, event: Event) -> None:
        """Schedule ``event`` in nondecreasing timestamp order."""
        assert event.timestamp >= self.timestamp, (
            f"Event timestamp {event.timestamp} must be greater than or equal "
            f"to the current simulator timestamp {self.timestamp}"
        )
        self.queue.put(event)
        if logging.level_debug():
            logging.debug(
                "%s added to the queue at timestamp %d. event queue size = %d",
                event,
                self.timestamp,
                self.event_queue_length(),
            )

    def cancel(self, event: Event | str) -> bool:
        """Cancel a queued event by object or UUID.

        Returns ``True`` only when a queued event was removed. Matching by UUID
        is important because distinct events at the same timestamp compare equal.
        """
        event_uuid = event.uuid if isinstance(event, Event) else event
        if not isinstance(event_uuid, str):
            raise TypeError("event must be an Event or event UUID string")
        return self.queue.remove(lambda queued: queued.uuid == event_uuid) is not None

    def _matching_listeners(self, event: Event) -> list[EventListener[Any]]:
        """Return matching listeners in global priority order.

        Listener registration is keyed by type for efficient common-case lookup,
        but base-class and universal listeners must also receive subclass events.
        """
        listeners = [
            listener
            for event_type, registered in self.event_listeners.items()
            if isinstance(event, event_type)
            for listener in registered
            if listener.cond(event)
        ]
        listeners.sort(key=lambda listener: listener.priority)
        return listeners

    def get(self) -> Event | None:
        """Process and return the earliest queued event, or ``None`` if empty."""
        if self.event_queue_length() == 0:
            return None

        event = self.queue.get()
        self.timestamp = max(self.timestamp, event.timestamp)

        # Snapshot matching listeners so callbacks may safely add or remove
        # registrations without mutating this iteration.
        for listener in self._matching_listeners(event):
            if logging.level_debug():
                logging.debug("%s called for %s", listener.callback, event)
            listener.callback(event)

        return event

    def event_queue_length(self) -> int:
        return len(self.queue)

    def num_event_listeners(self) -> int:
        return sum(len(listeners) for listeners in self.event_listeners.values())

    def _print_profile_summary(
        self,
        label: str,
        event_type_cumtime: dict[str, float],
        event_type_count: dict[str, int],
        top_n: int = 0,
    ) -> None:
        """Log an event profiling summary from exact per-event timings."""
        sorted_types = sorted(event_type_cumtime.items(), key=lambda item: -item[1])
        if top_n > 0:
            sorted_types = sorted_types[:top_n]
        total = sum(event_type_cumtime.values())
        if total <= 0:
            return

        profile_lines = []
        for event_name, cumulative_time in sorted_types:
            count = event_type_count.get(event_name, 0)
            average_ms = cumulative_time / count * 1000 if count > 0 else 0.0
            percentage = cumulative_time / total * 100
            profile_lines.append(
                f"  {event_name}: {cumulative_time:.3f}s ({percentage:.1f}%), "
                f"count={count}, avg={average_ms:.3f}ms"
            )
        logging.info(
            "%s: %s (total %.3fs):\n%s",
            self.name,
            label,
            total,
            "\n".join(profile_lines),
        )

    def run(self, max_events: int = -1) -> None:
        """Run until the queue is empty or ``max_events`` have been processed."""
        events_processed = 0
        progress_bar = None
        if self.progress_bar_update:
            progress_bar = tqdm.tqdm(total=self.progress_bar_total)

        start_time = time.time()
        current_wall_time = start_time

        profiling = self.enable_profile
        event_type_cumtime: dict[str, float] = {}
        event_type_count: dict[str, int] = {}
        slow_event_threshold = 0.1

        while self.event_queue_length() > 0 and (
            max_events == -1 or events_processed < max_events
        ):
            if profiling:
                event_start = time.perf_counter()
                event = self.get()
                elapsed = time.perf_counter() - event_start
                assert event is not None
                event_name = type(event).__name__
                event_type_cumtime[event_name] = (
                    event_type_cumtime.get(event_name, 0.0) + elapsed
                )
                event_type_count[event_name] = event_type_count.get(event_name, 0) + 1
                if elapsed > slow_event_threshold:
                    logging.warning(
                        "%s: Slow event #%d %s (%.3fs) at sim time %d, "
                        "queue size %d",
                        self.name,
                        events_processed,
                        event_name,
                        elapsed,
                        self.timestamp,
                        self.event_queue_length(),
                    )
            else:
                event = self.get()
                assert event is not None
            events_processed += 1

            if logging.level_debug():
                logging.debug(
                    "Event %s processed at timestamp %d.", event.uuid, self.timestamp
                )
                logging.debug(
                    "%s: Processed %d events, current timestamp: %d, "
                    "event queue length: %d",
                    self.name,
                    events_processed,
                    self.timestamp,
                    self.event_queue_length(),
                )

            if events_processed % 10000 == 0:
                elapsed_time = time.time() - current_wall_time
                current_wall_time = time.time()
                logging.info(
                    "%s: Simulation at %f hours. Processed %d events so far. "
                    "Simulation speed: %.2f events/sec.",
                    self.name,
                    self.timestamp / (3600 * int(1e9)),
                    events_processed,
                    10000 / elapsed_time,
                )

                if profiling:
                    self._print_profile_summary(
                        "Event profiling (top 10 by cumtime)",
                        event_type_cumtime,
                        event_type_count,
                        top_n=10,
                    )

                if logging.level_debug():
                    logging.debug(
                        "%s: Current event queue breakdown: %s",
                        self.name,
                        json.dumps(self.get_event_queue_breakdown_snapshot(), indent=4),
                    )

            if self.progress_bar_update:
                assert progress_bar is not None
                self.progress_bar_update(progress_bar)

        if progress_bar is not None:
            progress_bar.close()

        logging.info(
            "%s: Simulator finished after processing %d events. Final timestamp: "
            "%d. Time taken: %.2f hours.",
            self.name,
            events_processed,
            self.timestamp,
            (time.time() - start_time) / 3600,
        )

        if profiling:
            self._print_profile_summary(
                "Final event profiling summary",
                event_type_cumtime,
                event_type_count,
            )

    def add_event_listener(self, event_listener: EventListener[Any]) -> None:
        """Register a listener, keeping each type bucket priority-sorted."""
        if event_listener.event_type not in self.event_listeners:
            self.event_listeners[event_listener.event_type] = []
        bisect.insort(
            self.event_listeners[event_listener.event_type], event_listener
        )
        logging.debug("Added %s", event_listener)
        logging.debug("Num of event listeners: %d", self.num_event_listeners())

    def remove_event_listeners_by_criteria(
        self, criteria: Callable[[EventListener[Any]], bool]
    ) -> None:
        """Remove all listeners for which ``criteria`` returns ``True``."""
        debug_logging = logging.level_debug()
        original_count = self.num_event_listeners() if debug_logging else 0
        for event_type, listeners in self.event_listeners.items():
            self.event_listeners[event_type] = [
                listener for listener in listeners if not criteria(listener)
            ]
        if debug_logging:
            removed_count = original_count - self.num_event_listeners()
            logging.debug("Removed %d event listeners by criteria.", removed_count)
            logging.debug("Num of event listeners: %d", self.num_event_listeners())

    def get_event_queue_breakdown_snapshot(self) -> dict[str, int]:
        """Return the number of queued events grouped by concrete type name."""
        event_type_counts: dict[str, int] = {}
        for event, _sequence in self.queue._queue:
            event_name = type(event).__name__
            event_type_counts[event_name] = event_type_counts.get(event_name, 0) + 1
        return event_type_counts
