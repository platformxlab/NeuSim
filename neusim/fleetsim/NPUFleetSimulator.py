import gzip
import os
import pickle
import random

from absl import logging

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
        self.cluster_manager = NPUClusterManager(
            self, random_generator=random.Random(114514)
        )
        self.sim_objects.append(self.cluster_manager)

        # Client for generating requests
        # We only have one client for now, which will generate requests
        # based on the workload configuration (LLMInferenceWorkloadConfig).
        # The single client can be used to mimic multi-tenant traffic
        # by enqueuing multiple requests with the same or close timestamps.
        self.client = LLMInferenceServiceClient(self)
        self.sim_objects.append(self.client)

        # Metrics server for monitoring the metrics used for auto-scaling.
        # The metrics server will also help us collect simuation statistics.
        self.metrics_server = MetricsServer(self)
        self.sim_objects.append(self.metrics_server)

        # LLM inference endpoint
        # We use a single endpoint for each model deployment.
        # The endpoint will initialize all vPods for initial allocation as well as the autoscaler
        # based on the workload configuration.
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

    def save_to_checkpoint(self, save_path: str | None = None):
        if save_path is None:
            save_path = os.path.join(
                self.config.output_dir, f"checkpoint_{self.timestamp}"
            )
        os.makedirs(save_path, exist_ok=True)

        # save config
        with gzip.open(os.path.join(save_path, "config.pkl.gz"), "wb") as f:
            pickle.dump(self.config, f)

        # save sim_objs
        if self.cluster_manager:
            self.cluster_manager.save_to_checkpoint(
                os.path.join(save_path, "cluster_manager.pkl")
            )
        if self.llm_inference_endpoint:
            self.llm_inference_endpoint.save_to_checkpoint(
                os.path.join(save_path, "llm_inference_endpoint.pkl")
            )
        if self.metrics_server:
            self.metrics_server.save_to_checkpoint(
                os.path.join(save_path, "metrics_server.pkl")
            )
        if self.client:
            self.client.save_to_checkpoint(os.path.join(save_path, "client.pkl"))

        logging.info(f"Checkpointed simulator instance to {save_path}")
