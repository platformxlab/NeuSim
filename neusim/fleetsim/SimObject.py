## Top-level sim object class for the NPU fleet simulator.
## Can be extended to represent different types of simulation objects, such as workload executors, NPU pods, schedulers, etc.

import typing

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

    def __str__(self):
        return f"SimObject {self.name}"

    def __repr__(self):
        return str(self)
