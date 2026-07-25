import gzip
import pickle
import typing
from copy import deepcopy

if typing.TYPE_CHECKING:
    from neusim.fleetsim.NPUFleetSimulator import NPUFleetSimulator
from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
)
from neusim.fleetsim.LLMInferenceEvents import (
    LLMInferenceRequestEnqueueEvent,
    LLMInferenceRequestEnqueueNextBatchEvent,
)
from neusim.fleetsim.LoadGenerator import LLMRequest, LoadGenerator
from neusim.fleetsim.SimObject import SimObject


class LLMInferenceServiceClient(SimObject):
    """
    Client for sending inference requests to the LLM service.
    """

    def __init__(self, simulator: "NPUFleetSimulator"):
        super().__init__("LLMInferenceServiceClient", simulator)
        self.config: LLMInferenceWorkloadConfig = deepcopy(
            simulator.config.workload_config
        )
        self.loadgen: LoadGenerator = LoadGenerator(self.config)

        # Build the effective workload before the endpoint creates autoscalers.
        # Autoscalers size initial allocations directly from ``client.requests``.
        requests = self.loadgen.generate()
        if self.config.max_timestamp != -1:
            requests = [
                request
                for request in requests
                if request.enqueue_timestamp <= self.config.max_timestamp
            ]
        if self.config.max_num_requests != -1:
            requests = requests[: self.config.max_num_requests]
        self.requests: list[LLMRequest] = requests
        self.total_num_enqueued_requests: int = 0
        """
        Total number of requests that have been enqueued to the event queue.
        This is the real number of requests that will be processed by the simulator.
        """

        self.next_request_index: int = 0
        """
        Index of the next request to be enqueued.
        This is used to keep track of which requests have been enqueued.
        """

    def initialize(self):
        """
        Initialize the client by enqueuing all requests.
        These requests will be processed by the simulator according to their enqueue timestamps.
        """
        # self.requests = self.loadgen.generate()

        self.simulator.add_event_listener(
            LLMInferenceRequestEnqueueNextBatchEvent.get_type_listener(
                self.enqueue_all_requests,
            )
        )

        self.enqueue_all_requests()

    def enqueue_all_requests(
        self, event: LLMInferenceRequestEnqueueNextBatchEvent | None = None
    ):
        """
        Enqueue all requests to the event queue.
        """
        next_request_end_index = min(self.next_request_index + 100, len(self.requests))
        for req in self.requests[self.next_request_index : next_request_end_index]:
            if (
                self.config.max_timestamp != -1
                and req.enqueue_timestamp > self.config.max_timestamp
            ):
                continue
            if (
                self.config.max_num_requests != -1
                and self.total_num_enqueued_requests >= self.config.max_num_requests
            ):
                break
            self.total_num_enqueued_requests += 1
            self.simulator.put(LLMInferenceRequestEnqueueEvent(req))
        self.next_request_index = next_request_end_index

        if self.next_request_index >= len(self.requests):
            return
        if (
            self.config.max_num_requests != -1
            and self.total_num_enqueued_requests >= self.config.max_num_requests
        ):
            return
        if (
            self.config.max_timestamp != -1
            and self.requests[self.next_request_index].enqueue_timestamp
            > self.config.max_timestamp
        ):
            return
        # Enqueue the next batch event
        self.simulator.put(
            LLMInferenceRequestEnqueueNextBatchEvent(
                max(
                    0, self.requests[next_request_end_index - 1].enqueue_timestamp - 100
                )  # slightly before the last request's enqueue time
            )
        )

    def save_to_checkpoint(self, checkpoint_path: str):
        cpt = {"config": self.config, "requests": self.requests}
        with gzip.open(checkpoint_path, "wb") as f:
            pickle.dump(cpt, f)

    def load_from_checkpoint(self, checkpoint_path: str):
        with gzip.open(checkpoint_path, "rb") as f:
            cpt = pickle.load(f)
            self.config = cpt["config"]
            self.requests = cpt["requests"]
            self.loadgen = LoadGenerator(self.config)
