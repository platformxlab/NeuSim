"""Focused tests for the Stage-A lookup-cache generator."""

import json
from pathlib import Path
from typing import Any

import pytest

from neusim.run_scripts import run_sim_find_optimal as stage_a


def test_resolve_default_paths_is_repo_relative() -> None:
    paths = stage_a.resolve_default_paths({})

    assert paths["results_dir"] == Path.cwd() / "results" / "fleetsim"
    assert paths["traces_dir"] == Path.cwd() / "traces" / "inference"
    assert paths["configs_dir"] == stage_a.REPO_ROOT / "configs"
    assert (
        paths["request_cache_dir"]
        == Path.cwd() / "results" / "fleetsim" / "request_lookup_cache"
    )


def test_resolve_default_paths_honors_environment_overrides() -> None:
    paths = stage_a.resolve_default_paths(
        {
            "NEUSIM_RESULTS_DIR": "/tmp/custom-results",
            "NEUSIM_TRACES_DIR": "/tmp/custom-traces",
            "NEUSIM_CONFIGS_DIR": "/tmp/custom-configs",
            "NEUSIM_REQUEST_CACHE_DIR": "/tmp/custom-cache",
        }
    )

    assert paths == {
        "results_dir": Path("/tmp/custom-results"),
        "traces_dir": Path("/tmp/custom-traces"),
        "configs_dir": Path("/tmp/custom-configs"),
        "request_cache_dir": Path("/tmp/custom-cache"),
    }


def test_default_request_trace_files_are_rooted_under_override() -> None:
    traces = stage_a.default_request_trace_files("/datasets/inference")

    assert len(traces) == 4
    assert all(path.startswith("/datasets/inference/") for path in traces)
    assert traces[0].endswith(
        "AzurePublicDataset/data/AzureLLMInferenceTrace_code_1week_sampled.csv"
    )
    assert traces[-1].endswith("BurstGPT/data/synthetic_BurstGPT_trace.csv")


def test_trace_schema_is_detected_from_header_not_filename(tmp_path: Path) -> None:
    misleading_burst = tmp_path / "Azure_LVEval_trace.csv"
    misleading_burst.write_text(
        "Timestamp,Request tokens,Response tokens\n0,17,3\n",
        encoding="utf-8",
    )
    neutral_azure = tmp_path / "workload.csv"
    neutral_azure.write_text(
        "TIMESTAMP,ContextTokens,GeneratedTokens\n" "2026-01-01T00:00:00+00:00,31,7\n",
        encoding="utf-8",
    )

    assert stage_a.read_trace_seqlens(misleading_burst) == [(17, 3)]
    assert stage_a.read_trace_seqlens(neutral_azure) == [(31, 7)]


def test_trace_schema_rejects_unknown_or_empty_csv(tmp_path: Path) -> None:
    unknown = tmp_path / "Azure_unknown.csv"
    unknown.write_text("time,input,output\n0,1,2\n", encoding="utf-8")
    empty = tmp_path / "BurstGPT_empty.csv"
    empty.write_text("Timestamp,Request tokens,Response tokens\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported trace CSV schema"):
        stage_a.read_trace_seqlens(unknown)
    with pytest.raises(ValueError, match="contains no requests"):
        stage_a.read_trace_seqlens(empty)


def test_stage_a_ray_runtime_environment_is_portable() -> None:
    runtime_env_path = stage_a.REPO_ROOT / "neusim" / "run_scripts" / "runtime_env.json"
    runtime_env = json.loads(runtime_env_path.read_text())

    assert runtime_env == {}


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, 32), (32, 32), (33, 64), (128, 128), (129, 256), (513, 1024)],
)
def test_pad_seqlen_selects_factor_by_threshold(value: int, expected: int) -> None:
    assert stage_a.pad_seqlen(value, [32, 128, 512], [128, 512]) == expected


def test_pad_seqlen_rejects_mismatched_factor_and_step_counts() -> None:
    with pytest.raises(ValueError, match="one more element"):
        stage_a.pad_seqlen(10, [32], [128])


def test_build_run_configs_expands_every_dimension_without_mutating_generator_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        stage_a,
        "_get_ray",
        lambda: pytest.fail("pure config building must not initialize Ray"),
    )
    calls: list[tuple[int, str, dict[str, Any]]] = []
    generated_config = {
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": 8,
    }

    def fake_generator(
        num_chips: int, model: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        calls.append((num_chips, model, kwargs))
        return [generated_config]

    configs = stage_a.build_run_configs(
        "llama3-8b",
        [(128, 8), (1024, 64)],
        versions=["4", "6e"],
        num_chips_list=["8"],
        batch_sizes=["1", "4"],
        skip_exist=True,
        output_dir="/tmp/cache",
        workload="inference",
        max_pp=1,
        configs_path="/tmp/configs",
        parallelism_generator=fake_generator,
    )

    assert len(configs) == 8
    assert {
        (
            config[1],
            config[2]["input_seqlen"],
            config[2]["output_seqlen"],
            config[3],
        )
        for config in configs
    } == {
        (version, input_seqlen, output_seqlen, batch_size)
        for version in ("4", "6e")
        for input_seqlen, output_seqlen in ((128, 8), (1024, 64))
        for batch_size in (1, 4)
    }
    assert all(config[0] == "llama3-8b" for config in configs)
    assert all(
        config[4:] == (True, "/tmp/cache", "inference", "/tmp/configs")
        for config in configs
    )
    assert generated_config == {
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": 8,
    }
    assert calls == [
        (
            8,
            "llama3-8b",
            {
                "max_dp": 1,
                "max_num_dcn_pods": 1,
                "max_etp": 8,
                "max_pp": 1,
            },
        )
    ]


def test_build_run_configs_expands_large_chip_counts_over_dcn_sizes() -> None:
    calls: list[tuple[int, int]] = []

    def fake_generator(
        num_chips: int, model: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        del model
        calls.append((num_chips, kwargs["max_num_dcn_pods"]))
        return []

    configs = stage_a.build_run_configs(
        "deepseekv3-671b",
        [(128, 8)],
        versions=["6e"],
        num_chips_list=[512],
        batch_sizes=[1],
        skip_exist=False,
        output_dir="/tmp/cache",
        workload="inference",
        max_pp=2,
        parallelism_generator=fake_generator,
    )

    assert configs == []
    assert calls == [(512, 1), (256, 2), (128, 4)]


def test_generate_wrapper_builds_grouping_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefill_stats = {"sim_config": {"input_seqlen": 128}}
    decode_stats = {"sim_config": {"output_seqlen": 16}}
    generated = (
        "llama3-8b",
        "6e",
        ["prefill-op"],
        ["decode-op"],
        prefill_stats,
        decode_stats,
    )
    monkeypatch.setattr(stage_a, "generate", lambda *args: generated)

    result = stage_a.generate_wrapper({"item": ("unused",)})

    assert result["model_seqlen_config_key"] == "llama3-8b_128_16"
    assert result["model_seqlen_config"] == ("llama3-8b", (128, 16))
    assert result["item"] == generated


def test_optimizer_accepts_explicit_results_without_cache_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_a.OUTPUT_DIR = "/tmp/unused-cache"
    stage_a.VERSIONS = ["5p", "6e"]
    stage_a.GENERATE_OPT_RESULTS = False
    stage_a.GENERATE_TRACE = True
    monkeypatch.setattr(
        stage_a.gzip,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError(args)),
    )

    result = stage_a.find_cost_optimal_config_for_model(
        {"item": ("deepseekv3-671b", 32, 4)},
        result_list_override=[{"item": ("unused",)}],
    )

    assert result == {"item": None}


def test_generate_wrapper_preserves_filtered_row_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stage_a, "generate", lambda *args: None)

    assert stage_a.generate_wrapper({"item": ("unused",)}) == {
        "item": None,
        "model_seqlen_config_key": None,
        "model_seqlen_config": None,
    }


class _FakeOperator:
    def to_csv_dict(self) -> dict[str, str]:
        return {"name": "fake"}


def _optimizer_result(num_chips: int, efficiency: float) -> dict[str, Any]:
    sim_config = {"name": "5p", "num_chips": num_chips}
    prefill = {
        "sim_config": dict(sim_config),
        "out_of_memory": False,
        "TTFT_sec": 0.5,
        "avg_power_efficiency_tkn_per_joule": efficiency,
        "monetary_cost_tkn_per_dollar": efficiency,
    }
    decode = {
        "sim_config": dict(sim_config),
        "out_of_memory": False,
        "TPOT_ms_request": 0.5,
        "avg_power_efficiency_tkn_per_joule": efficiency,
        "monetary_cost_tkn_per_dollar": efficiency,
    }
    return {
        "item": (
            "deepseekv3-671b",
            "5p",
            [_FakeOperator()],
            [_FakeOperator()],
            prefill,
            decode,
        )
    }


@pytest.mark.parametrize(
    ("top_k", "expected_chips"),
    [(-1, [8, 4, 2, 1]), (2, [8, 4])],
)
def test_optimizer_writes_requested_rank_count_and_cleans_stale_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    top_k: int,
    expected_chips: list[int],
) -> None:
    monkeypatch.setattr(stage_a, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(stage_a, "VERSIONS", ["5p"])
    monkeypatch.setattr(stage_a, "GENERATE_OPT_RESULTS", True)
    monkeypatch.setattr(stage_a, "GENERATE_TRACE", False)
    monkeypatch.setattr(stage_a, "SLO_SCALE", 2.0)
    monkeypatch.setattr(stage_a, "SLO_BASELINE_VERSION", "5p")
    monkeypatch.setattr(stage_a, "OPTIMAL_TOP_K", top_k)
    stale_dir = tmp_path / "energy" / "deepseekv3-671b" / "32_4" / "5p" / "prefill"
    stale_dir.mkdir(parents=True)
    (stale_dir / "9.json").write_text("{}", encoding="utf-8")
    (stale_dir / "9.csv").write_text("stale", encoding="utf-8")
    (stale_dir / "notes.txt").write_text("keep", encoding="utf-8")

    result = stage_a.find_cost_optimal_config_for_model(
        {"item": ("deepseekv3-671b", 32, 4)},
        result_list_override=[
            _optimizer_result(chips, efficiency)
            for chips, efficiency in ((1, 10.0), (2, 20.0), (4, 30.0), (8, 40.0))
        ],
    )

    assert result == {"item": None}
    for goal in ("energy", "monetary"):
        for phase in ("prefill", "decode"):
            rank_dir = tmp_path / goal / "deepseekv3-671b" / "32_4" / "5p" / phase
            ranks = sorted(rank_dir.glob("*.json"))
            assert [path.name for path in ranks] == [
                f"{rank}.json" for rank in range(1, len(expected_chips) + 1)
            ]
            assert [
                json.loads(path.read_text(encoding="utf-8"))["sim_config"]["num_chips"]
                for path in ranks
            ] == expected_chips
    assert not (stale_dir / "9.csv").exists()
    assert (stale_dir / "notes.txt").read_text(encoding="utf-8") == "keep"


def test_rank_selector_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        stage_a.select_ranked_config_alternatives([], "efficiency", 0)
