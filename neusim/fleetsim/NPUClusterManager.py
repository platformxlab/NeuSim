"""Static chip-config inventory used by FleetSim."""

import json
import os
from typing import TYPE_CHECKING

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.systems.NPUFleetConfig import NPUClusterSchedulerConfig
from neusim.fleetsim.SimObject import SimObject

if TYPE_CHECKING:
    from neusim.fleetsim.NPUFleetSimulator import NPUFleetSimulator


class NPUClusterManager(SimObject):
    """Load exactly the chip definitions named by the static vPod allocation."""

    def __init__(self, simulator: "NPUFleetSimulator"):
        super().__init__("NPUClusterManager", simulator)
        self.scheduler_config: NPUClusterSchedulerConfig = (
            simulator.config.cluster_scheduler_config
        )
        self.npu_types: tuple[str, ...] = simulator.config.npu_types
        self._validate_static_inventory()
        self.chip_configs: dict[str, ChipConfig] = self._load_chip_configs()

    def _validate_static_inventory(self) -> None:
        if not self.npu_types:
            raise ValueError("static_vpod_allocation must reference an NPU type")
        if len(set(self.npu_types)) != len(self.npu_types):
            raise ValueError(f"NPU types must be unique, got {self.npu_types}")
        if any(not isinstance(value, str) or not value for value in self.npu_types):
            raise ValueError(f"NPU types must be non-empty strings, got {self.npu_types}")

    def _load_chip_configs(self) -> dict[str, ChipConfig]:
        chip_configs: dict[str, ChipConfig] = {}
        for npu_type in self.npu_types:
            chip_config_file = os.path.join(
                self.scheduler_config.chip_config_path, f"tpuv{npu_type}.json"
            )
            with open(chip_config_file) as config_file:
                chip_configs[npu_type] = ChipConfig.model_validate(
                    json.load(config_file)
                )
        return chip_configs
