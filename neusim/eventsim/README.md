# Event-driven simulation backend

`neusim.eventsim` provides the event queue and listener primitives used by
FleetSim. Events are processed in nondecreasing timestamp order; events with an
equal timestamp retain insertion order. Listeners can
target a concrete event class, a base class, or every `Event`; lower listener
priority values run first.

```python
from neusim.eventsim import Event, EventSimulator

simulator = EventSimulator()
simulator.add_event_listener(
    Event.get_universal_listener(lambda event: print(event), priority=10)
)
event = Event(timestamp=100)
simulator.put(event)
simulator.run()
```

`EventSimulator.cancel(event)` (or `cancel(event.uuid)`) removes an event that
has not yet run and reports whether it was found. `run(max_events=N)` leaves any
remaining events queued so the simulation can be resumed. Set `enable_profile`
to collect and log cumulative callback/dispatch time by concrete event type.
