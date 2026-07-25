### NPU Cluster Manager object. Emulates whether an NPU pod allocation request can be satisfied or not.

import gzip
import json
import math
import os
import pickle
import random
from collections.abc import Sequence
from typing import TYPE_CHECKING

from absl import logging

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.systems.NPUFleetConfig import NPUClusterSchedulerConfig
from neusim.fleetsim.SimObject import SimObject

if TYPE_CHECKING:
    from neusim.fleetsim.NPUFleetSimulator import NPUFleetSimulator


class NPUPodAllocationRequest:
    """
    Represents a request for allocating an NPU pod.
    """

    def __init__(
        self,
        timestamp: int,
        npu_type: str,
        num_chips: int | None = None,
        pod_shape: Sequence[int] | None = None,
    ):
        if (num_chips is None) == (pod_shape is None):
            raise ValueError("Specify exactly one of num_chips or pod_shape.")
        if timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {timestamp}.")
        if not isinstance(npu_type, str) or not npu_type:
            raise ValueError("npu_type must be a non-empty string.")

        normalized_shape: tuple[int, ...] | None = None
        if pod_shape is not None:
            normalized_shape = tuple(pod_shape)
            if not normalized_shape or any(dim <= 0 for dim in normalized_shape):
                raise ValueError(
                    f"pod_shape must contain only positive dimensions, got {pod_shape}."
                )
            num_chips = math.prod(normalized_shape)
        elif num_chips is None or num_chips <= 0:
            raise ValueError(f"num_chips must be positive, got {num_chips}.")

        self.npu_type = npu_type
        self.num_chips = num_chips
        self.pod_shape = normalized_shape
        self.timestamp = timestamp


class NPUClusterManager(SimObject):
    """
    Manages the allocation of NPU pods in the fleet simulator.
    This class can be extended to implement specific allocation strategies.
    """

    def __init__(self, simulator: "NPUFleetSimulator", random_generator=None):
        super().__init__("NPUClusterManager", simulator)
        self.scheduler_config: NPUClusterSchedulerConfig = (
            simulator.config.cluster_scheduler_config
        )
        self._validate_scheduler_config()
        # random_generator.random() must return a float in [0, 1).
        self.random_generator = random_generator or random.Random()
        self.last_allocation_success: bool | None = None
        self.last_allocation_request_timestamp: int = 0
        self.last_allocation_npu_type: str | None = None
        self.allocation_update_interval_ns: int = int(1e9 * 60 * 10)  # 10 min
        self._allocation_outcomes: dict[str, tuple[bool, int]] = {}

        # load chip configs
        self.chip_configs: dict[str, ChipConfig] = self._load_chip_configs()

        # chip usage tracking
        self._allocated_chips: dict[str, int] = {
            v: 0 for v in self.scheduler_config.npu_types
        }

    def _validate_scheduler_config(self) -> None:
        npu_types = list(self.scheduler_config.npu_types)
        probabilities = list(self.scheduler_config.satisfaction_probability)
        max_chips = self.scheduler_config.max_chips_per_version

        if not npu_types:
            raise ValueError("NPUClusterSchedulerConfig.npu_types must not be empty.")
        if len(set(npu_types)) != len(npu_types):
            raise ValueError(f"npu_types must be unique, got {npu_types}.")
        if any(not isinstance(npu_type, str) or not npu_type for npu_type in npu_types):
            raise ValueError(
                f"npu_types must contain non-empty strings, got {npu_types}."
            )
        for phase in ("prefill", "decode"):
            field_name = f"{phase}_npu_types"
            phase_npu_types = getattr(self.scheduler_config, field_name, None)
            if phase_npu_types is None:
                continue
            if not phase_npu_types:
                raise ValueError(f"{field_name} must not be empty when provided.")
            if len(set(phase_npu_types)) != len(phase_npu_types):
                raise ValueError(
                    f"{field_name} must contain unique NPU types, "
                    f"got {phase_npu_types}."
                )
            unknown_types = set(phase_npu_types) - set(npu_types)
            if unknown_types:
                raise ValueError(
                    f"{field_name} contains types not present in npu_types: "
                    f"{sorted(unknown_types)}."
                )
        if len(probabilities) not in {1, len(npu_types)}:
            raise ValueError(
                "satisfaction_probability must contain one shared value or one value "
                f"per NPU type; got {len(probabilities)} values for {len(npu_types)} types."
            )
        if any(not 0.0 <= probability <= 1.0 for probability in probabilities):
            raise ValueError(
                f"satisfaction_probability values must be in [0, 1], got {probabilities}."
            )
        if max_chips is not None:
            unknown_types = set(max_chips) - set(npu_types)
            if unknown_types:
                raise ValueError(
                    "max_chips_per_version contains types not present in npu_types: "
                    f"{sorted(unknown_types)}."
                )
            if any(limit < 0 for limit in max_chips.values()):
                raise ValueError(
                    f"max_chips_per_version values must be non-negative, got {max_chips}."
                )

    def _load_chip_configs(self) -> dict[str, ChipConfig]:
        """
        Load chip configurations from the specified path.
        """
        chip_configs = {}
        for v in self.scheduler_config.npu_types:
            chip_config_file = os.path.join(
                self.scheduler_config.chip_config_path, f"tpuv{v}.json"
            )
            with open(chip_config_file) as f:
                config_dict = json.load(f)
                chip_config = ChipConfig.model_validate(config_dict)
                chip_configs[v] = chip_config
        return chip_configs

    def get_allocation_satisfaction_probability(self, npu_type: str) -> float:
        """
        Get the probability of satisfying an NPU pod allocation request for a given NPU type.
        """
        if npu_type not in self.scheduler_config.npu_types:
            raise ValueError(
                f"NPU type {npu_type} is not supported by the cluster scheduler."
            )

        if len(self.scheduler_config.satisfaction_probability) == 1:
            # If there is only one probability in the list, use the same probability for all requests.
            return self.scheduler_config.satisfaction_probability[0]

        index = self.scheduler_config.npu_types.index(npu_type)
        return self.scheduler_config.satisfaction_probability[index]

    def _has_capacity(self, npu_type: str, num_chips: int) -> bool:
        """
        Check if adding num_chips of npu_type would exceed the configured max.
        Returns True if no limit is set for that version.
        """
        self._validate_allocation_size(npu_type, num_chips)
        max_chips = self.scheduler_config.max_chips_per_version
        if max_chips is None or npu_type not in max_chips:
            return True
        return self._allocated_chips.get(npu_type, 0) + num_chips <= max_chips[npu_type]

    def track_allocate(self, npu_type: str, num_chips: int):
        """
        Increment the allocated chip count for the given NPU type.
        """
        self._validate_allocation_size(npu_type, num_chips)
        self._allocated_chips[npu_type] += num_chips
        if logging.level_debug():
            logging.debug(
                "track_allocate: %s += %d chips (total: %d)",
                npu_type,
                num_chips,
                self._allocated_chips[npu_type],
            )

    def track_deallocate(self, npu_type: str, num_chips: int):
        """
        Decrement the allocated chip count for the given NPU type.
        """
        self._validate_allocation_size(npu_type, num_chips)
        allocated = self._allocated_chips[npu_type]
        if num_chips > allocated:
            raise ValueError(
                f"Cannot deallocate {num_chips} {npu_type} chips; only {allocated} are allocated."
            )
        self._allocated_chips[npu_type] -= num_chips
        if logging.level_debug():
            logging.debug(
                "track_deallocate: %s -= %d chips (total: %d)",
                npu_type,
                num_chips,
                self._allocated_chips[npu_type],
            )

    def get_allocated_chips(self) -> dict[str, int]:
        """
        Returns a copy of the current allocated chip counts.
        """
        return self._allocated_chips.copy()

    def _validate_allocation_size(self, npu_type: str, num_chips: int) -> None:
        if npu_type not in self._allocated_chips:
            raise ValueError(
                f"NPU type {npu_type} is not supported by the cluster scheduler."
            )
        if num_chips <= 0:
            raise ValueError(f"num_chips must be positive, got {num_chips}.")

    def allocate(self, req: NPUPodAllocationRequest) -> bool:
        """
        Check if a pod allocation request can be satisfied.
        @return: True if allocation is successful. False otherwise.
        """
        # Always check capacity limit first, even when probabilistic result is cached
        if not self._has_capacity(req.npu_type, req.num_chips):
            if logging.level_debug():
                logging.debug(
                    "Allocation rejected for %s (%d chips): capacity limit reached (allocated: %d, max: %s)",
                    req.npu_type,
                    req.num_chips,
                    self._allocated_chips.get(req.npu_type, 0),
                    self.scheduler_config.max_chips_per_version.get(
                        req.npu_type, "unlimited"
                    )
                    if self.scheduler_config.max_chips_per_version
                    else "unlimited",
                )
            return False

        cached_outcome = self._allocation_outcomes.get(req.npu_type)
        if cached_outcome is not None:
            outcome, timestamp = cached_outcome
            elapsed = req.timestamp - timestamp
            if 0 <= elapsed <= self.allocation_update_interval_ns:
                self.last_allocation_success = outcome
                self.last_allocation_request_timestamp = timestamp
                self.last_allocation_npu_type = req.npu_type
                return outcome

        prob = self.get_allocation_satisfaction_probability(req.npu_type)
        alloc_success = self.random_generator.random() < prob
        self.last_allocation_success = alloc_success
        self.last_allocation_request_timestamp = req.timestamp
        self.last_allocation_npu_type = req.npu_type
        self._allocation_outcomes[req.npu_type] = (alloc_success, req.timestamp)
        return alloc_success

    def save_to_checkpoint(self, checkpoint_path: str):
        cpt = {
            "scheduler_config": self.scheduler_config,
            "chip_configs": self.chip_configs,
            "random_generator": self.random_generator,
            "_allocated_chips": self._allocated_chips,
            "last_allocation_success": self.last_allocation_success,
            "last_allocation_request_timestamp": self.last_allocation_request_timestamp,
            "last_allocation_npu_type": self.last_allocation_npu_type,
            "allocation_update_interval_ns": self.allocation_update_interval_ns,
            "_allocation_outcomes": self._allocation_outcomes,
        }
        with gzip.open(checkpoint_path, "wb") as f:
            pickle.dump(cpt, f)

    def load_from_checkpoint(self, checkpoint_path: str):
        with gzip.open(checkpoint_path, "rb") as f:
            cpt = pickle.load(f)
            self.scheduler_config = cpt["scheduler_config"]
            self._validate_scheduler_config()
            self.chip_configs = cpt["chip_configs"]
            self.random_generator = cpt["random_generator"]
            self._allocated_chips = cpt.get(
                "_allocated_chips", {v: 0 for v in self.scheduler_config.npu_types}
            )
            self.last_allocation_success = cpt.get("last_allocation_success")
            self.last_allocation_request_timestamp = cpt.get(
                "last_allocation_request_timestamp", 0
            )
            self.last_allocation_npu_type = cpt.get("last_allocation_npu_type")
            self.allocation_update_interval_ns = cpt.get(
                "allocation_update_interval_ns", int(1e9 * 60 * 10)
            )
            self._allocation_outcomes = cpt.get("_allocation_outcomes", {})

    ## TODO: new feature: add synthetic preemption/migration events
    ## TODO: new feature: add synthetic failure events
