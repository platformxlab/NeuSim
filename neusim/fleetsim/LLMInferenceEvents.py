from collections.abc import Callable, Sequence
from typing import Any

from neusim.configs.models.ModelConfig import ModelConfig

# from neusim.configs.systems.NPUFleetConfig import PhysicalCubeConfig
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


class vPodReconfigurationStartEvent(LLMInferenceEngineEvent):
    __slots__ = ["target_config", "delay_ns"]

    def __init__(
        self, timestamp: int, worker_id: str, target_config: ModelConfig, delay_ns: int
    ):
        super().__init__(timestamp, worker_id)
        self.target_config: ModelConfig = target_config
        """Target configuration."""
        self.delay_ns: int = delay_ns
        """Reconfiguration delay."""


class vPodReconfigurationEndEvent(LLMInferenceEngineEvent):
    pass


class vPodHorizontalScalingEvent(Event):
    """
    This event is fired as a horizontal scaling recommendation
    to adjust the number of vPod replicas.
    """

    __slots__ = ["replica_count", "prefill_or_decode"]

    def __init__(self, timestamp: int, replica_count: int, prefill_or_decode: str):
        super().__init__(timestamp)
        self.replica_count: int = replica_count
        """The number of replicas that the LLM inference endpoint should scale to."""
        self.prefill_or_decode: str = prefill_or_decode
        """Indicates whether the scaling is for prefill or decode."""

    def __str__(self):
        return (
            super().__str__()
            + f"Target Replica Count: {self.replica_count}. {self.prefill_or_decode}."
        )


class vPodVerticalScalingEvent(Event):
    """
    This event is fired as a vertical scaling recommendation
    to adjust the configuration of all vPod instances.
    """

    __slots__ = ["vPod_config", "prefill_or_decode"]

    def __init__(
        self, timestamp: int, vPod_config: ModelConfig, prefill_or_decode: str
    ):
        super().__init__(timestamp)
        self.vPod_config: ModelConfig = vPod_config
        """The recommended configuration for the vPod instance."""
        self.prefill_or_decode: str = prefill_or_decode
        """Indicates whether the scaling is for prefill or decode."""

    def __str__(self):
        return (
            super().__str__()
            + f"Target # of chips: {self.vPod_config.num_chips}. {self.prefill_or_decode}."
        )


class vPodMultiDimensionalScalingEvent(Event):
    """
    This event is fired as a multi-dimensional scaling recommendation
    to adjust both the number of vPod replicas and the configuration of all vPod instances.
    """

    __slots__ = [
        "vPod_allocations",
        "prefill_or_decode",
        "do_not_scale_down",
        "is_routine",
    ]

    def __init__(
        self,
        timestamp: int,
        vPod_allocations: list[tuple[Sequence[ModelConfig], int]] | None,
        prefill_or_decode: str,
        do_not_scale_down: bool = False,
        is_routine: bool = True,
    ):
        super().__init__(timestamp)
        self.vPod_allocations: list[
            tuple[Sequence[ModelConfig], int]
        ] | None = vPod_allocations
        """The vPod allocations for the scaling event. Each allocation is a tuple of (replica_count, config)."""
        self.prefill_or_decode: str = prefill_or_decode
        """Indicates whether the scaling is for prefill or decode."""
        self.do_not_scale_down: bool = do_not_scale_down
        self.is_routine: bool = is_routine

    def __str__(self):
        return (
            super().__str__()
            + f"{self.prefill_or_decode}: vPod Allocations: {self.vPod_allocations}"
        )


class vPodHorizontalScalingRecommendationEvent(Event):
    """
    This event triggers the auto-scaler recommender to generate a new horizontal scaling recommendation
    based on current workload metrics.
    @self.do_not_scale_down: If True, the auto-scaler should not scale down the number of replicas.
        This is useful for event-driven HS when new request arrives.
    @self.is_routine: If True, the scaling recommendation is part of the regular scaling routine.
        This is useful for distinguishing between routine HS events and request-triggered HS events.
    """

    __slots__ = [
        "prefill_or_decode",
        "autoscaler_id",
        "do_not_scale_down",
        "is_routine",
    ]

    def __init__(
        self,
        timestamp: int,
        prefill_or_decode: str,
        autoscaler_id: str,
        do_not_scale_down: bool = False,
        is_routine: bool = True,
    ):
        super().__init__(timestamp)
        self.prefill_or_decode: str = prefill_or_decode
        """Indicates whether the scaling is for prefill or decode."""
        self.autoscaler_id: str = autoscaler_id
        """The ID of the auto-scaler that generated the recommendation."""
        self.do_not_scale_down: bool = do_not_scale_down
        self.is_routine: bool = is_routine

    def __str__(self):
        return (
            super().__str__()
            + f"Prefill/Decode: {self.prefill_or_decode}. Autoscaler ID: {self.autoscaler_id}. do_not_scale_down={self.do_not_scale_down}. is_routine={self.is_routine}."
        )

    @classmethod
    def get_autoscaler_id_listener(
        cls,
        callback: Callable[["vPodHorizontalScalingRecommendationEvent"], None],
        autoscaler_id: str,
        priority: int = 999,
    ) -> EventListener:
        """
        Get an event listener that will call the callback when any event of this type with the same autoscaler_id is processed.
        The callback will be called with the event as the argument.
        """
        return EventListener(
            cls,
            lambda e: isinstance(e, cls) and e.autoscaler_id == autoscaler_id,
            callback,
            priority=priority,
        )


class vPodVerticalScalingRecommendationEvent(Event):
    """
    This event triggers the auto-scaler recommender to generate a new vertical scaling recommendation
    based on current workload metrics.
    """

    __slots__ = ["prefill_or_decode", "autoscaler_id"]

    def __init__(self, timestamp: int, prefill_or_decode: str, autoscaler_id: str):
        super().__init__(timestamp)
        self.prefill_or_decode: str = prefill_or_decode
        """Indicates whether the scaling is for prefill or decode."""
        self.autoscaler_id: str = autoscaler_id
        """The ID of the auto-scaler that generated the recommendation."""

    def __str__(self):
        return (
            super().__str__()
            + f"Prefill/Decode: {self.prefill_or_decode}. Autoscaler ID: {self.autoscaler_id}."
        )

    @classmethod
    def get_autoscaler_id_listener(
        cls,
        callback: Callable[["vPodVerticalScalingRecommendationEvent"], None],
        autoscaler_id: str,
        priority: int = 999,
    ) -> EventListener:
        """
        Get an event listener that will call the callback when any event of this type with the same autoscaler_id is processed.
        The callback will be called with the event as the argument.
        """
        return EventListener(
            cls,
            lambda e: isinstance(e, cls) and e.autoscaler_id == autoscaler_id,
            callback,
            priority=priority,
        )


class vPodMultiDimensionalScalingRecommendationEvent(Event):
    """
    This event triggers the auto-scaler recommender to generate a new multi-dimensional scaling recommendation
    based on current workload metrics.
    """

    __slots__ = [
        "prefill_or_decode",
        "autoscaler_id",
        "do_not_scale_down",
        "is_routine",
    ]

    def __init__(
        self,
        timestamp: int,
        prefill_or_decode: str,
        autoscaler_id: str,
        do_not_scale_down: bool = False,
        is_routine: bool = True,
    ):
        super().__init__(timestamp)
        self.prefill_or_decode: str = prefill_or_decode
        """Indicates whether the scaling is for prefill or decode."""
        self.autoscaler_id: str = autoscaler_id
        """The ID of the auto-scaler that generated the recommendation."""
        self.do_not_scale_down: bool = do_not_scale_down
        self.is_routine: bool = is_routine

    def __str__(self):
        return (
            super().__str__()
            + f"Prefill/Decode: {self.prefill_or_decode}. Autoscaler ID: {self.autoscaler_id}. do_not_scale_down={self.do_not_scale_down}. is_routine={self.is_routine}."
        )

    @classmethod
    def get_autoscaler_id_listener(
        cls,
        callback: Callable[["vPodMultiDimensionalScalingRecommendationEvent"], None],
        autoscaler_id: str,
        priority: int = 999,
    ) -> EventListener:
        """
        Get an event listener that will call the callback when any event of this type with the same autoscaler_id is processed.
        The callback will be called with the event as the argument.
        """
        return EventListener(
            cls,
            lambda e: isinstance(e, cls) and e.autoscaler_id == autoscaler_id,
            callback,
            priority=priority,
        )


class vPodCreateEvent(LLMInferenceEngineEvent):
    """
    This event is used to signal the creation of a new vPod instance.
    """

    pass


class vPodDestroyEvent(LLMInferenceEngineEvent):
    """
    This event is used to signal the destruction of a vPod instance.
    """

    pass
