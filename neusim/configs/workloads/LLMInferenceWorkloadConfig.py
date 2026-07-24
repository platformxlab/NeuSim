"""Configuration models for static-vPod FleetSim workloads."""

from enum import Enum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
    model_validator,
)

from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig


class StaticVPodEntry(BaseModel):
    """One homogeneous group in a static virtual-pod allocation."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(gt=0)
    npu_type: str = Field(min_length=1)
    num_chips: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    dp: int = Field(default=1, gt=0)
    tp: int = Field(default=1, gt=0)
    pp: int = Field(default=1, gt=0)
    ep: int = Field(default=1, gt=0)


class StaticVPodAllocation(BaseModel):
    """Fixed prefill and decode deployments used for the full simulation."""

    model_config = ConfigDict(extra="forbid")

    prefill: StaticVPodEntry
    decode: StaticVPodEntry

    @property
    def npu_types(self) -> tuple[str, ...]:
        """Return the chip versions referenced by this allocation."""
        return tuple(sorted({self.prefill.npu_type, self.decode.npu_type}))


class RequestPatternType(str, Enum):
    """Supported request-arrival patterns."""

    TRACE = "trace"
    SYNTHETIC = "synthetic"


class LLMInferenceWorkloadConfig(BaseModel):
    """Static LLM-inference workload configuration.

    FleetSim in this artifact intentionally models only the fixed vPod
    deployments used by the NPU DVFS paper. A deployment must therefore be
    supplied explicitly; there is no autoscaler or configuration-search path.
    """

    model_config = ConfigDict(extra="forbid")

    static_vpod_allocation: StaticVPodAllocation

    request_pattern: RequestPatternType = RequestPatternType.TRACE
    trace_file_path: str = ""
    max_timestamp: int = -1
    max_num_requests: int = -1
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

    max_decode_batch_size: int = -1
    min_decode_schedule_num_iterations: int = Field(default=4, gt=0)
    max_decode_schedule_num_iterations: int = Field(default=256, gt=0)
    decode_batch_seqlen_ratio_threshold: float = Field(default=2.0, ge=1)
    decode_batch_seqlen_min_threshold: int = Field(default=256, ge=0)

    # Static service-level DVFS. These knobs restore the paper's request
    # scheduler without reintroducing NeuScale or any allocation changes.
    enable_dvfs_power_model: bool = False
    enable_dvfs: bool = False
    dvfs_policy: str = "Custom"
    dvfs_max_perf_degrad: float = Field(default=1.0, ge=0)
    dvfs_safeguard_window_minutes: float = Field(default=5.0, gt=0)
    dvfs_safeguard_violation_threshold: float = Field(default=0.01, ge=0, le=1)
    dvfs_lookup_cache_dir: str = ""
    dvfs_require_cache_hit: bool = False
    slo_json_path: str = ""
    slo_multiplier: str = "5x"

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
        if self.enable_dvfs and not self.slo_json_path:
            raise ValueError("slo_json_path is required when enable_dvfs is true")
        if self.enable_dvfs and not self.enable_dvfs_power_model:
            raise ValueError("enable_dvfs requires enable_dvfs_power_model to be true")
        if self.dvfs_require_cache_hit and not self.enable_dvfs:
            raise ValueError("dvfs_require_cache_hit requires enable_dvfs to be true")
        if self.dvfs_require_cache_hit and not self.dvfs_lookup_cache_dir:
            raise ValueError(
                "dvfs_lookup_cache_dir is required when "
                "dvfs_require_cache_hit is true"
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
