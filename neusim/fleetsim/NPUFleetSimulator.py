import os

import neusim.fleetsim.npusim_backend_interface as npusim_backend
from neusim.configs.systems.NPUFleetConfig import NPUFleetConfig
from neusim.eventsim.EventSim import EventSimulator
from neusim.fleetsim.LLMInferenceEndpoint import LLMInferenceEndpoint
from neusim.fleetsim.LLMInferenceServiceClient import LLMInferenceServiceClient
from neusim.fleetsim.MetricsServer import MetricsServer
from neusim.fleetsim.NPUClusterManager import NPUClusterManager
from neusim.fleetsim.SimObject import SimObject


class NPUFleetSimulator(EventSimulator):
    """
    Simulator for a fleet of (possibly heterogeneous) NPU pods.
    """

    def __init__(self, config: NPUFleetConfig, name: str | None = None):
        super().__init__(name or "NPUFleetSimulator")

        self.config: NPUFleetConfig = config
        self.sim_objects: list[SimObject] = []
        self.cluster_manager: NPUClusterManager | None = None
        self.llm_inference_endpoint: LLMInferenceEndpoint | None = None
        self.metrics_server: MetricsServer | None = None
        self.client: LLMInferenceServiceClient | None = None

        # initalize npusim backend cache dir
        if not os.path.exists(self.config.npusim_backend_cache_dir):
            os.makedirs(self.config.npusim_backend_cache_dir, exist_ok=True)
        if self.config.npusim_backend_cache_use_mmap:
            npusim_backend.set_npusim_backend_cache_dir(
                self.config.npusim_backend_cache_dir,
                mmap_mode="r",
            )
        else:
            npusim_backend.set_npusim_backend_cache_dir(
                self.config.npusim_backend_cache_dir
            )

        self._initialize_sim_objects()

        # setup profiling
        self.enable_profile = self.config.enable_profile
        npusim_backend.set_enable_profile(self.enable_profile)

        # setup tqdm
        if self.config.tqdm:
            self.progress_bar_total = self.event_queue_length()

            def pbar_update(pbar):
                pbar.n = len(self.metrics_server.request_trace)  # type: ignore
                pbar.refresh()

            self.progress_bar_update = pbar_update

    def _initialize_sim_objects(self):
        """
        Initialize simulation objects based on the fleet configuration.
        Upon initialization, some sim objects may add initial events to the event queue.
        """
        # cluster scheduler
        self.cluster_manager = NPUClusterManager(self)
        self.sim_objects.append(self.cluster_manager)

        # Client for generating requests
        # We only have one client for now, which will generate requests
        # based on the workload configuration (LLMInferenceWorkloadConfig).
        # The single client can be used to mimic multi-tenant traffic
        # by enqueuing multiple requests with the same or close timestamps.
        self.client = LLMInferenceServiceClient(self)
        self.sim_objects.append(self.client)

        # Collect completed-request timings and aggregate simulation statistics.
        self.metrics_server = MetricsServer(self)
        self.sim_objects.append(self.metrics_server)

        # LLM inference endpoint
        # A single endpoint initializes the fixed prefill/decode vPod deployment.
        self.llm_inference_endpoint = LLMInferenceEndpoint("LLMInferenceEndpoint", self)
        self.sim_objects.append(self.llm_inference_endpoint)

        for sim_obj in self.sim_objects:
            sim_obj.initialize()

    def _put_initial_events(self):
        """
        Put initial events into the event queue.
        """
        pass  # no need to do anything in the top-level module

    def run(self, max_events: int = -1):
        """
        Run the simulator until the event queue is empty.
        This will process all events in the queue and update the state of simulation objects.
        """
        self._put_initial_events()
        super().run(max_events)

    def dump_simulation_stats(self):
        """
        Dump simulation statistics to a file.
        """
        for sim_obj in self.sim_objects:
            sim_obj.dump_simulation_stats()
