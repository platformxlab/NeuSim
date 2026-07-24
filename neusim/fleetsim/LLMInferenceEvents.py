from collections.abc import Callable, Sequence
from typing import Any

from neusim.eventsim.Event import Event, EventListener
from neusim.fleetsim.LoadGenerator import LLMRequest


class LLMInferenceRequestEnqueueNextBatchEvent(Event):
    """
    This event is used to signal the client to enqueue the next batch of requests.
    This is helpful in keeping the total event queue size manageable
    compared to issuing all LLMInferenceRequestEnqueueEvents at once.
    """

    pass


class LLMInferenceRequestEnqueueEvent(Event):
    __slots__ = ["request"]

    def __init__(self, request: LLMRequest):
        super().__init__(request.enqueue_timestamp)
        self.request: LLMRequest = request

    def __str__(self):
        return (
            super().__str__()
            + f"Request {self.request.id} with input length {self.request.input_seqlen} and output length {self.request.output_seqlen}."
        )


class LLMInferenceEngineEvent(Event):
    __slots__ = ["worker_id"]

    def __init__(self, timestamp: int, worker_id: str):
        super().__init__(timestamp)
        self.worker_id: str = worker_id
        """The uuid of the worker (vPod) that will handle this prefill."""

    @classmethod
    def get_worker_id_listener[_EventType: "LLMInferenceEngineEvent"](
        cls,
        callback: Callable[[_EventType], None],
        worker_id: str,
        priority: int = 999,
        metadata: dict[str, Any] | None = None,
    ) -> EventListener:
        """
        Get an event listener that will call the callback when any event of this type with the same worker_id is processed.
        The callback will be called with the event as the argument.
        """
        return EventListener(
            cls,
            lambda e: isinstance(e, cls) and e.worker_id == worker_id,
            callback,
            priority=priority,
            metadata={
                "type": "worker_id_listener",
                "worker_id": worker_id,
                **(metadata or {}),
            },
        )

    def __str__(self):
        return super().__str__() + f"Worker: {self.worker_id}. "


class LLMInferencePrefillStartEvent(LLMInferenceEngineEvent):
    __slots__ = ["requests"]

    def __init__(self, requests: Sequence[LLMRequest], timestamp: int, worker_id: str):
        super().__init__(timestamp, worker_id)
        self.requests: Sequence[LLMRequest] = requests

    def __str__(self):
        return super().__str__() + f"Requests: {self.requests}."


class LLMInferencePrefillEndEvent(LLMInferenceEngineEvent):
    __slots__ = ["requests"]

    def __init__(self, requests: Sequence[LLMRequest], timestamp: int, worker_id: str):
        super().__init__(timestamp, worker_id)
        self.requests: Sequence[LLMRequest] = requests

    def __str__(self):
        return super().__str__() + f"Requests: {self.requests}."


class LLMInferenceDecodeIterationStartEvent(LLMInferenceEngineEvent):
    __slots__ = ["requests", "num_iterations"]

    def __init__(
        self,
        requests: Sequence[LLMRequest],
        timestamp: int,
        worker_id: str,
        num_iterations: int,
    ):
        super().__init__(timestamp, worker_id)
        self.requests: Sequence[LLMRequest] = requests
        self.num_iterations: int = num_iterations

    def __str__(self):
        return super().__str__() + f"Requests: {self.requests}."


class LLMInferenceDecodeIterationEndEvent(LLMInferenceEngineEvent):
    __slots__ = ["requests", "num_iterations"]

    def __init__(
        self,
        requests: Sequence[LLMRequest],
        timestamp: int,
        worker_id: str,
        num_iterations: int,
    ):
        super().__init__(timestamp, worker_id)
        self.requests: Sequence[LLMRequest] = requests
        self.num_iterations: int = num_iterations

    def __str__(self):
        return super().__str__() + f"Requests: {self.requests}."


class LLMInferenceEngineReadyEvent(LLMInferenceEngineEvent):
    """
    This event is used to signal that the LLM inference engine is ready to process more requests.
    It is used to handle pipeline parallelism, where the engine can process the next batch of
    requests as soon as the previous batch has finished the first stage.
    """

    pass
