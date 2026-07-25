## Top-level sim object class for the NPU fleet simulator.
## Can be extended to represent different types of simulation objects, such as workload executors, NPU pods, schedulers, etc.

import typing

from absl import logging

if typing.TYPE_CHECKING:
    from neusim.fleetsim.NPUFleetSimulator import NPUFleetSimulator


class SimObject:
    """
    Base class for simulation objects in the NPU fleet simulator.
    Each simulation object can have its own state and behavior.
    """

    __slots__ = ["name", "simulator"]

    def __init__(self, name: str, simulator: "NPUFleetSimulator"):
        self.name: str = name
        self.simulator: "NPUFleetSimulator" = simulator

    def initialize(self):
        """
        Initialize the simulation object.
        This method is called by the top-level simulator before the simulation starts.
        This method should be overridden if the subclass wants to perform any necessary setup.
        """
        pass

    def dump_simulation_stats(self):
        """
        Dump simulation statistics for this object.
        This method can be overridden to provide specific statistics for the object.
        """
        pass

    def save_to_checkpoint(self, checkpoint_path: str):
        """
        Checkpoint necessary state into a python pickle file.
        """
        logging.warning(
            "Checkpointing for %s (type=%s) is not implemented.", self.name, type(self)
        )

    def load_from_checkpoint(self, checkpoint_path: str):
        """
        Load necessary state from a python pickle file.
        """
        logging.warning(
            "Loading checkpoint for %s (type=%s) is not implemented.",
            self.name,
            type(self),
        )

    def __str__(self):
        return f"SimObject {self.name}"

    def __repr__(self):
        return str(self)
