import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig
from neusim.configs.systems.NPUFleetConfig import (
    NPUClusterSchedulerConfig,
    NPUFleetConfig,
    PhysicalCubeConfig,
    VirtualSliceConfig,
    num_chips_to_shape_3D,
)
from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
)


def test_num_chips_to_shape_3d_and_invalid_values() -> None:
    assert num_chips_to_shape_3D(1) == [1, 1, 1]
    assert num_chips_to_shape_3D(4) == [2, 2, 1]
    assert num_chips_to_shape_3D(64) == [4, 4, 4]
    with pytest.raises(ValueError, match="positive integer"):
        num_chips_to_shape_3D(0)
    with pytest.raises(ValueError, match="positive integer"):
        num_chips_to_shape_3D(True)


def test_nested_defaults_are_independent_and_portable() -> None:
    first = NPUFleetConfig()
    second = NPUFleetConfig()

    first.cluster_scheduler_config.npu_types.append("custom")
    first.workload_config.input_seqlen_padding_factors.append(65536)
    first.workload_config.llm_config.input_seqlen = 17

    assert "custom" not in second.cluster_scheduler_config.npu_types
    assert 65536 not in second.workload_config.input_seqlen_padding_factors
    assert second.workload_config.llm_config.input_seqlen != 17

    paths = [
        second.cluster_scheduler_config.chip_config_path,
        second.workload_config.trace_file_path,
        second.workload_config.request_results_cache_dir,
        second.output_dir,
        second.npusim_backend_cache_dir,
    ]
    assert all("/mnt/spr8nfs" not in path for path in paths)
    assert Path(second.cluster_scheduler_config.chip_config_path, "tpuv5p.json").is_file()
    assert Path(second.workload_config.trace_file_path).is_file()


def test_chip_config_default_honors_configs_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NEUSIM_CONFIGS_DIR", str(tmp_path))

    config = NPUClusterSchedulerConfig()

    assert Path(config.chip_config_path) == tmp_path / "chips"


def test_static_vpod_allocation_parses_and_is_required_for_static_mode() -> None:
    allocation = {
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
    config = LLMInferenceWorkloadConfig(
        autoscaler_type="StaticAutoScaler", static_vpod_allocation=allocation
    )
    assert config.static_vpod_allocation is not None
    assert config.static_vpod_allocation.prefill.count == 2
    assert config.static_vpod_allocation.decode.tp == 2

    with pytest.raises(ValidationError, match="static_vpod_allocation"):
        LLMInferenceWorkloadConfig(autoscaler_type="StaticAutoScaler")


def test_deepseek_mapping_selects_deepseek_union_member() -> None:
    deepseek = {
        "d_ff": 7168,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "qk_nope_head_dim": 128,
        "v_head_dim": 128,
    }
    workload = LLMInferenceWorkloadConfig(llm_config=deepseek)
    assert isinstance(workload.llm_config, DeepSeekConfig)
    assert workload.llm_config.use_flash_attention is False

    round_tripped = LLMInferenceWorkloadConfig.model_validate_json(
        workload.model_dump_json()
    )
    assert isinstance(round_tripped.llm_config, DeepSeekConfig)
    assert round_tripped.llm_config.kv_lora_rank == 512
    assert round_tripped.llm_config.q_lora_rank == 1536


def test_physical_cube_validates_shape_and_extracts_chip_config() -> None:
    cube = PhysicalCubeConfig.from_ModelConfig(LLMConfig(num_chips=12, name="6e"))
    assert cube.shape == (3, 2, 2)
    assert cube.num_chips == 12
    assert cube.chip_config.name == "6e"

    with pytest.raises(ValidationError, match="shape"):
        PhysicalCubeConfig(shape=(1, 2))
    with pytest.raises(ValidationError, match="positive"):
        VirtualSliceConfig(shape=[1, 0, 1])


def test_workload_rejects_inconsistent_padding_schedule() -> None:
    with pytest.raises(ValidationError, match="exactly one more"):
        LLMInferenceWorkloadConfig(
            input_seqlen_padding_factors=[4],
            input_seqlen_padding_steps=[8],
        )


def test_json_assets_live_only_in_repository_configs() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_json_files = sorted((repo_root / "neusim" / "configs").rglob("*.json"))

    assert package_json_files == []


def test_qwen_model_in_repository_configs_preserves_model_identity() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    model_path = repo_root / "configs" / "models" / "llama-qwen3-32b.json"
    model = LLMConfig.model_validate(json.loads(model_path.read_text()))

    assert model.model_name == "llama-qwen3-32b"
