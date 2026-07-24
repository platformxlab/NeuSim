"""System configuration models for static FleetSim deployments."""

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

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


class NPUClusterSchedulerConfig(BaseModel):
    """Static cluster assets needed to materialize the configured vPods."""

    model_config = ConfigDict(extra="forbid")

    chip_config_path: str = Field(default_factory=_default_chip_config_path)


class NPUFleetConfig(BaseModel):
    """Top-level configuration for one fixed prefill/decode deployment."""

    model_config = ConfigDict(extra="forbid")

    workload_config: LLMInferenceWorkloadConfig
    cluster_scheduler_config: NPUClusterSchedulerConfig = Field(
        default_factory=NPUClusterSchedulerConfig
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

    @property
    def npu_types(self) -> tuple[str, ...]:
        """Chip versions derived from the required static allocation."""
        return self.workload_config.static_vpod_allocation.npu_types
