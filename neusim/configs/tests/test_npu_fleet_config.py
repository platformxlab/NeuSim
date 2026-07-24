from pathlib import Path

import pytest
from pydantic import ValidationError

from neusim.configs.models.LLMConfig import DeepSeekConfig
from neusim.configs.systems.NPUFleetConfig import (
    NPUClusterSchedulerConfig,
    NPUFleetConfig,
)
from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
)


def allocation() -> dict:
    return {
        "prefill": {
            "count": 2,
            "npu_type": "5p",
            "num_chips": 4,
            "batch_size": 8,
        },
        "decode": {
            "count": 1,
            "npu_type": "6e",
            "num_chips": 8,
            "batch_size": 16,
            "tp": 2,
        },
    }


def workload(**updates) -> LLMInferenceWorkloadConfig:
    return LLMInferenceWorkloadConfig(static_vpod_allocation=allocation(), **updates)


def test_static_vpod_allocation_is_required_and_derives_chip_types() -> None:
    config = workload()
    assert config.static_vpod_allocation.prefill.count == 2
    assert config.static_vpod_allocation.decode.tp == 2
    assert config.static_vpod_allocation.npu_types == ("5p", "6e")

    with pytest.raises(ValidationError, match="static_vpod_allocation"):
        LLMInferenceWorkloadConfig()
    with pytest.raises(ValidationError, match="workload_config"):
        NPUFleetConfig()


def test_removed_dynamic_configuration_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="autoscaler_type"):
        workload(autoscaler_type="HorizontalAutoScaler")
    with pytest.raises(ValidationError, match="satisfaction_probability"):
        NPUClusterSchedulerConfig(satisfaction_probability=[1.0])


def test_nested_defaults_are_independent_and_portable() -> None:
    first = NPUFleetConfig(workload_config=workload())
    second = NPUFleetConfig(workload_config=workload())

    first.workload_config.input_seqlen_padding_factors.append(65536)
    first.workload_config.llm_config.input_seqlen = 17

    assert 65536 not in second.workload_config.input_seqlen_padding_factors
    assert second.workload_config.llm_config.input_seqlen != 17
    assert second.npu_types == ("5p", "6e")

    assert second.workload_config.trace_file_path == ""
    paths = [
        second.cluster_scheduler_config.chip_config_path,
        second.output_dir,
        second.npusim_backend_cache_dir,
    ]
    assert all("/mnt/spr8nfs" not in path for path in paths)
    assert Path(
        second.cluster_scheduler_config.chip_config_path, "tpuv5p.json"
    ).is_file()


def test_chip_config_default_honors_configs_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEUSIM_CONFIGS_DIR", str(tmp_path))
    config = NPUClusterSchedulerConfig()
    assert Path(config.chip_config_path) == tmp_path / "chips"


def test_deepseek_mapping_selects_deepseek_union_member() -> None:
    deepseek = {
        "d_ff": 7168,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
    }
    config = workload(llm_config=deepseek)
    assert isinstance(config.llm_config, DeepSeekConfig)
    assert config.llm_config.use_flash_attention is False

    round_tripped = LLMInferenceWorkloadConfig.model_validate_json(
        config.model_dump_json()
    )
    assert isinstance(round_tripped.llm_config, DeepSeekConfig)
    assert round_tripped.llm_config.kv_lora_rank == 512


def test_workload_rejects_inconsistent_padding_and_decode_schedules() -> None:
    with pytest.raises(ValidationError, match="exactly one more"):
        workload(input_seqlen_padding_factors=[4], input_seqlen_padding_steps=[8])
    with pytest.raises(ValidationError, match="cannot exceed"):
        workload(
            min_decode_schedule_num_iterations=8,
            max_decode_schedule_num_iterations=4,
        )


def test_static_dvfs_requires_explicit_slo_and_strict_cache_paths() -> None:
    assert workload().enable_dvfs_power_model is False
    assert workload(enable_dvfs_power_model=True).enable_dvfs is False
    with pytest.raises(ValidationError, match="slo_json_path"):
        workload(enable_dvfs=True)
    with pytest.raises(ValidationError, match="enable_dvfs_power_model"):
        workload(enable_dvfs=True, slo_json_path="slo.json")
    with pytest.raises(ValidationError, match="requires enable_dvfs"):
        workload(dvfs_require_cache_hit=True)
    with pytest.raises(ValidationError, match="dvfs_lookup_cache_dir"):
        workload(
            enable_dvfs_power_model=True,
            enable_dvfs=True,
            slo_json_path="slo.json",
            dvfs_require_cache_hit=True,
        )


def test_json_assets_live_only_in_repository_configs() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_json_files = sorted((repo_root / "neusim" / "configs").rglob("*.json"))
    assert package_json_files == []
