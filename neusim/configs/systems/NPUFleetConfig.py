"""System configuration models for FleetSim."""

import math
import os
import uuid
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.models.ModelConfig import ModelConfig
from neusim.configs.systems.SystemConfig import SystemConfig
from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RESULTS_DIR = Path(
    os.environ.get("NEUSIM_RESULTS_DIR", Path.cwd() / "results" / "fleetsim")
).expanduser()


def _default_chip_config_path() -> str:
    """Resolve the chip JSON directory, honoring runtime environment overrides."""
    configs_dir = Path(
        os.environ.get("NEUSIM_CONFIGS_DIR") or _REPO_ROOT / "configs"
    ).expanduser()
    return str(configs_dir / "chips")


def num_chips_to_shape_3D(num_chips: int) -> list[int]:
    """Return a factorization ``[x, y, z]`` that is as cube-like as possible."""
    if not isinstance(num_chips, int) or isinstance(num_chips, bool) or num_chips <= 0:
        raise ValueError(f"num_chips must be a positive integer, got {num_chips!r}")

    dim_z = max(1, int(math.cbrt(num_chips)))
    while num_chips % dim_z:
        dim_z -= 1
    remaining = num_chips // dim_z
    dim_y = max(1, math.isqrt(remaining))
    while remaining % dim_y:
        dim_y -= 1
    dim_x = remaining // dim_y
    return [dim_x, dim_y, dim_z]


class ICITopology(str, Enum):
    """Inter-chip interconnect topology."""

    TORUS_2D = "2D-torus"
    TORUS_3D = "3D-torus"
    FULLY_CONNECTED = "fully-connected"


class VirtualSliceConfig(BaseModel):
    """Shape and root physical-NPU coordinate of a virtual slice."""

    name: str = Field(default_factory=lambda: uuid.uuid4().hex)
    shape: list[int] = Field(default_factory=lambda: [1, 1, 1])
    root: list[int] = Field(default_factory=lambda: [0, 0, 0])

    @field_validator("shape", "root")
    @classmethod
    def _require_three_dimensions(cls, value: list[int], info):
        if len(value) != 3:
            raise ValueError(f"{info.field_name} must contain exactly three dimensions")
        if info.field_name == "shape" and any(dim <= 0 for dim in value):
            raise ValueError("shape dimensions must be positive")
        return value

    @property
    def dimX(self) -> int:
        return self.shape[0]

    @property
    def dimY(self) -> int:
        return self.shape[1]

    @property
    def dimZ(self) -> int:
        return self.shape[2]

    @property
    def rootX(self) -> int:
        return self.root[0]

    @property
    def rootY(self) -> int:
        return self.root[1]

    @property
    def rootZ(self) -> int:
        return self.root[2]


class PhysicalCubeConfig(BaseModel):
    """Shape, topology, and chip definition for a physical NPU cube."""

    shape: tuple[int, int, int]
    topology: ICITopology = ICITopology.TORUS_3D
    chip_config: ChipConfig = Field(default_factory=ChipConfig)

    @field_validator("shape")
    @classmethod
    def _require_positive_shape(
        cls, value: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        if any(dim <= 0 for dim in value):
            raise ValueError("shape dimensions must be positive")
        return value

    @property
    def dimX(self) -> int:
        return self.shape[0]

    @property
    def dimY(self) -> int:
        return self.shape[1]

    @property
    def dimZ(self) -> int:
        return self.shape[2]

    @property
    def num_chips(self) -> int:
        return math.prod(self.shape)

    @classmethod
    def from_ModelConfig(cls, model_config: ModelConfig) -> "PhysicalCubeConfig":
        """Construct a physical cube from the model's chip and pod settings."""
        return cls(
            shape=tuple(num_chips_to_shape_3D(model_config.num_chips)),
            topology=ICITopology.TORUS_3D,
            chip_config=ChipConfig.model_validate(model_config.model_dump()),
        )


class NPUClusterSchedulerConfig(BaseModel):
    """Availability and capacity settings for the cluster scheduler."""

    npu_types: list[str] = Field(default_factory=lambda: ["4", "5e", "5p", "6e"])
    chip_config_path: str = Field(default_factory=_default_chip_config_path)
    satisfaction_probability: list[float] = Field(default_factory=lambda: [1.0])
    max_chips_per_version: dict[str, int] | None = None
    prefill_npu_types: list[str] | None = None
    decode_npu_types: list[str] | None = None


class NPUFleetConfig(BaseModel):
    """Top-level configuration for a fleet of NPU pods."""

    cluster_scheduler_config: NPUClusterSchedulerConfig = Field(
        default_factory=NPUClusterSchedulerConfig
    )
    workload_config: LLMInferenceWorkloadConfig = Field(
        default_factory=LLMInferenceWorkloadConfig
    )
    system_config: SystemConfig = Field(default_factory=SystemConfig)
    output_dir: str = str(_DEFAULT_RESULTS_DIR)
    npusim_backend_cache_dir: str = str(
        Path(
            os.environ.get(
                "NEUSIM_BACKEND_CACHE_DIR",
                _DEFAULT_RESULTS_DIR / ".cache" / "npusim_backend",
            )
        ).expanduser()
    )
    npusim_backend_cache_use_mmap: bool = False
    tqdm: bool = False
    enable_profile: bool = False
