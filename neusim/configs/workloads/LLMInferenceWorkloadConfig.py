"""Configuration models for FleetSim LLM-inference workloads."""

import os
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig

_NEUSIM_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RESULTS_DIR = Path(
    os.environ.get("NEUSIM_RESULTS_DIR", Path.cwd() / "results" / "fleetsim")
).expanduser()


class StaticVPodEntry(BaseModel):
    """One homogeneous group in a static virtual-pod allocation."""

    count: int = Field(gt=0)
    npu_type: str = Field(min_length=1)
    num_chips: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    dp: int = Field(default=1, gt=0)
    tp: int = Field(default=1, gt=0)
    pp: int = Field(default=1, gt=0)
    ep: int = Field(default=1, gt=0)


class StaticVPodAllocation(BaseModel):
    """Static allocation for the prefill and decode phases."""

    prefill: StaticVPodEntry
    decode: StaticVPodEntry


class RequestPatternType(str, Enum):
    """Supported request-arrival patterns."""

    TRACE = "trace"
    SYNTHETIC = "synthetic"


class LLMInferenceWorkloadConfig(BaseModel):
    """FleetSim workload and autoscaling settings.

    Defaults are deliberately local to the NeuSim checkout. Production experiments
    should normally override ``trace_file_path`` and ``request_results_cache_dir``.
    """

    request_pattern: RequestPatternType = RequestPatternType.TRACE
    trace_file_path: str = str(
        _NEUSIM_PACKAGE_ROOT
        / "fleetsim"
        / "tests"
        / "data"
        / "traces"
        / "AzureLLMInferenceTrace_code_test.csv"
    )
    max_timestamp: int = -1
    max_num_requests: int = -1
    max_decode_batch_size: int = -1
    request_rate: float = Field(default=1.0, gt=0)

    synthetic_num_requests: int = Field(default=500, gt=0)
    synthetic_request_rate: float = Field(default=10.0, gt=0)
    synthetic_input_len: int = Field(default=512, gt=0)
    synthetic_input_len_std: int = Field(default=0, ge=0)
    synthetic_output_len: int = Field(default=128, ge=2)
    synthetic_output_len_std: int = Field(default=0, ge=0)
    synthetic_seed: int = 42

    model_name: str = "llama3-70b"
    llm_config: SerializeAsAny[LLMConfig | DeepSeekConfig] = Field(
        default_factory=LLMConfig
    )
    autoscaler_type: Literal[
        "HorizontalAutoScaler",
        "IdealAutoScaler",
        "NeuScaleAutoScaler",
        "VerticalAutoScaler",
        "MultiPoolAutoScaler",
        "StaticAutoScaler",
    ] = "HorizontalAutoScaler"
    hs_initial_alloc_sample_criteria: Literal["max", "average"] = "max"

    hs_interval_minutes: float = Field(default=1.0, gt=0)
    vs_interval_minutes: float = Field(default=1.0, gt=0)
    hs_window_minutes: float = Field(default=10.0, gt=0)
    vs_window_minutes: float = Field(default=15.0, gt=0)
    instance_startup_delay_sec: int = Field(default=1, ge=0)

    input_seqlen_padding_factors: list[int] = Field(
        default_factory=lambda: [
            32,
            64,
            128,
            512,
            1024,
            4096,
            8192,
            16384,
            32768,
        ]
    )
    input_seqlen_padding_steps: list[int] = Field(
        default_factory=lambda: [
            128,
            512,
            1024,
            8192,
            16384,
            65536,
            131072,
            262144,
        ]
    )
    output_seqlen_padding_factors: list[int] = Field(
        default_factory=lambda: [4, 16, 32, 64, 128, 256, 512, 1024]
    )
    output_seqlen_padding_steps: list[int] = Field(
        default_factory=lambda: [32, 64, 128, 512, 1024, 2048, 8192]
    )
    pad_seqlen_loadgen: bool = True
    use_ideal_batch_size: bool = True
    min_decode_schedule_num_iterations: int = Field(default=4, gt=0)
    max_decode_schedule_num_iterations: int = Field(default=256, gt=0)

    request_results_cache_dir: str = str(
        Path(
            os.environ.get(
                "NEUSIM_REQUEST_CACHE_DIR",
                _DEFAULT_RESULTS_DIR / "request_lookup_cache",
            )
        ).expanduser()
    )
    optimization_goal: Literal["energy", "monetary"] = "energy"
    num_pools: int = Field(default=3, gt=0)
    ewma_alpha: float = Field(default=0.6, gt=0, le=1)
    ewma_interval_seconds: float = Field(default=10.0, gt=0)
    scaling_headroom_factor: float = Field(default=1.4, gt=0)
    queue_drain_target_seconds: float = Field(default=60.0, ge=0)
    coalesce_nl_threshold: float = Field(default=0.5, ge=0)
    decode_batch_seqlen_ratio_threshold: float = Field(default=2.0, ge=1)
    decode_batch_seqlen_min_threshold: int = Field(default=256, ge=0)
    decode_pool_single_config: bool = False
    output_prediction_accuracy: float = Field(default=1.0, ge=0.0, le=1.0)
    output_prediction_seed: int = 42
    mpa_min_interval_seconds: float = Field(default=10.0, ge=0)
    static_vpod_allocation: StaticVPodAllocation | None = None

    @field_validator("llm_config", mode="before")
    @classmethod
    def _preserve_deepseek_config(cls, value):
        """Select DeepSeekConfig when a plain mapping contains MLA fields."""
        if isinstance(value, dict) and {
            "kv_lora_rank",
            "q_lora_rank",
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "v_head_dim",
        }.issubset(value):
            return DeepSeekConfig(**value)
        return value

    @model_validator(mode="after")
    def _validate_related_settings(self):
        if self.max_timestamp < -1:
            raise ValueError("max_timestamp must be -1 or non-negative")
        if self.max_num_requests < -1:
            raise ValueError("max_num_requests must be -1 or non-negative")
        if self.max_decode_batch_size < -1 or self.max_decode_batch_size == 0:
            raise ValueError("max_decode_batch_size must be -1 or positive")
        if (
            self.min_decode_schedule_num_iterations
            > self.max_decode_schedule_num_iterations
        ):
            raise ValueError(
                "min_decode_schedule_num_iterations cannot exceed "
                "max_decode_schedule_num_iterations"
            )
        self._validate_padding_schedule(
            "input",
            self.input_seqlen_padding_factors,
            self.input_seqlen_padding_steps,
        )
        self._validate_padding_schedule(
            "output",
            self.output_seqlen_padding_factors,
            self.output_seqlen_padding_steps,
        )
        if (
            self.autoscaler_type == "StaticAutoScaler"
            and self.static_vpod_allocation is None
        ):
            raise ValueError("static_vpod_allocation is required for StaticAutoScaler")
        return self

    @staticmethod
    def _validate_padding_schedule(
        name: str, factors: list[int], steps: list[int]
    ) -> None:
        if len(factors) != len(steps) + 1:
            raise ValueError(
                f"{name}_seqlen_padding_factors must contain exactly one more "
                f"entry than {name}_seqlen_padding_steps"
            )
        if any(value <= 0 for value in factors + steps):
            raise ValueError(f"{name} padding values must be positive")
        if factors != sorted(factors) or steps != sorted(steps):
            raise ValueError(f"{name} padding values must be non-decreasing")
