#!/usr/bin/env python3
"""Build Stage-A request lookup caches for heterogeneous NPU fleets.

Run 'python -m neusim.run_scripts.run_sim_find_optimal --help' for usage.
"""

import csv
import gzip
import heapq
import json
import math
import os
import pickle
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
from absl import app, flags, logging

import neusim.npusim.frontend.query_results_helper_lib as results_lib
import neusim.npusim.frontend.run_sim_lib as run_sim_lib
from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig
from neusim.npusim.frontend.dit_ops_generator import DiTOpsGenerator
from neusim.npusim.frontend.dlrm_ops_generator import DLRMOpsGenerator
from neusim.npusim.frontend.gligen_ops_generator import GLIGENOpsGenerator
from neusim.npusim.frontend.llm_ops_generator import (
    DeepSeekOpsGenerator,
    LLMOpsGeneratorBase,
    LLMOpsGeneratorInference,
    LLMOpsGeneratorTraining,
)
from neusim.npusim.frontend.Operator import Operator

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_default_paths(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Path]:
    """Resolve portable Stage-A paths, honoring NeuSim environment overrides."""
    env = os.environ if environ is None else environ
    results_dir = Path(
        env.get("NEUSIM_RESULTS_DIR") or Path.cwd() / "results" / "fleetsim"
    ).expanduser()
    traces_dir = Path(
        env.get("NEUSIM_TRACES_DIR") or Path.cwd() / "traces" / "inference"
    ).expanduser()
    configs_dir = Path(
        env.get("NEUSIM_CONFIGS_DIR") or REPO_ROOT / "configs"
    ).expanduser()
    request_cache_dir = Path(
        env.get("NEUSIM_REQUEST_CACHE_DIR") or results_dir / "request_lookup_cache"
    ).expanduser()
    return {
        "results_dir": results_dir,
        "traces_dir": traces_dir,
        "configs_dir": configs_dir,
        "request_cache_dir": request_cache_dir,
    }


def default_request_trace_files(traces_dir: str | os.PathLike[str]) -> list[str]:
    """Return the standard lookup-cache input traces under traces_dir."""
    root = Path(traces_dir)
    return [
        os.fspath(
            root
            / "AzurePublicDataset"
            / "data"
            / "AzureLLMInferenceTrace_code_1week_sampled.csv"
        ),
        os.fspath(root / "BurstGPT" / "data" / "Azure_LVEval_traces.csv"),
        os.fspath(root / "BurstGPT" / "data" / "Azure_OpenThoughts_traces.csv"),
        os.fspath(root / "BurstGPT" / "data" / "synthetic_BurstGPT_trace.csv"),
    ]


def pad_to(value: int, multiple: int = 128) -> int:
    """Round value up to the next multiple."""
    return (value + multiple - 1) // multiple * multiple


def pad_seqlen(
    value: int, padding_factors: Sequence[int], padding_steps: Sequence[int]
) -> int:
    """Pad a sequence length using the factor selected by threshold."""
    if len(padding_factors) != len(padding_steps) + 1:
        raise ValueError(
            "padding_factors must have one more element than padding_steps"
        )
    for index, step in enumerate(padding_steps):
        if value <= step:
            return pad_to(value, padding_factors[index])
    return pad_to(value, padding_factors[-1])


def _get_ray() -> Any:
    """Import Ray only when a distributed operation is requested."""
    import ray

    return ray


_DEFAULT_PATHS = resolve_default_paths()
_STAGE_A_FLAGS = flags.FlagValues()


__REQUEST_TRACE_FILE = flags.DEFINE_multi_string(
    "request_trace_file",
    default_request_trace_files(_DEFAULT_PATHS["traces_dir"]),
    "Path to the request trace file used for generating sequence lengths",
    flag_values=_STAGE_A_FLAGS,
)
__INPUT_SEQLEN_PADDING_FACTORS = flags.DEFINE_list(
    "input_seqlen_padding_factors",
    [str(x) for x in [32, 64, 128, 512, 1024, 4096, 8192, 16384, 32768]],
    "List of input sequence length padding factors",
    flag_values=_STAGE_A_FLAGS,
)
__INPUT_SEQLEN_PADDING_STEPS = flags.DEFINE_list(
    "input_seqlen_padding_steps",
    [str(x) for x in [128, 512, 1024, 8192, 16384, 65536, 131072, 262144]],
    "List of input sequence length padding steps",
    flag_values=_STAGE_A_FLAGS,
)
__OUTPUT_SEQLEN_PADDING_FACTORS = flags.DEFINE_list(
    "output_seqlen_padding_factors",
    [str(x) for x in [4, 16, 32, 64, 128, 256, 512, 1024]],
    "List of output sequence length padding factors",
    flag_values=_STAGE_A_FLAGS,
)
__OUTPUT_SEQLEN_PADDING_STEPS = flags.DEFINE_list(
    "output_seqlen_padding_steps",
    [str(x) for x in [32, 64, 128, 512, 1024, 2048, 8192]],
    "List of output sequence length padding steps",
    flag_values=_STAGE_A_FLAGS,
)
__MODELS = flags.DEFINE_list(
    "models",
    ["llama3-8b", "llama3-70b", "llama3_1-405b", "deepseekv2-236b", "deepseekv3-671b"],
    "List of models to analyze",
    flag_values=_STAGE_A_FLAGS,
)
MODELS: list[str] | None = None
__VERSIONS = flags.DEFINE_list(
    "versions",
    ["5p", "6e"],
    "List of NPU versions to analyze",
    flag_values=_STAGE_A_FLAGS,
)
VERSIONS: list[str] | None = None
__NUM_CHIPS_LIST = flags.DEFINE_list(
    "num_chips",
    [str(x) for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]],
    "List of number of chips to analyze",
    flag_values=_STAGE_A_FLAGS,
)
NUM_CHIPS_LIST: list[str] | None = None
__INFERENCE_BATCH_SIZES = flags.DEFINE_list(
    "inference_batch_sizes",
    [str(x) for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]],
    "List of batch sizes to analyze for inference",
    flag_values=_STAGE_A_FLAGS,
)
INFERENCE_BATCH_SIZES: list[str] | None = None
__TRAINING_BATCH_SIZES = flags.DEFINE_list(
    "training_batch_sizes",
    [str(x) for x in [128]],
    "List of batch sizes to analyze for training",
    flag_values=_STAGE_A_FLAGS,
)
TRAINING_BATCH_SIZES: list[str] | None = None
__WORKLOAD = flags.DEFINE_string(
    "workload",
    "inference",
    "Workload type: training or inference",
    flag_values=_STAGE_A_FLAGS,
)
WORKLOAD: str | None = None
__OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    os.fspath(_DEFAULT_PATHS["request_cache_dir"]),
    "Output directory for trace files",
    flag_values=_STAGE_A_FLAGS,
)
OUTPUT_DIR: str | None = None
__CONFIGS_PATH = flags.DEFINE_string(
    "configs_path",
    os.fspath(_DEFAULT_PATHS["configs_dir"]),
    "Path to the configs directory",
    flag_values=_STAGE_A_FLAGS,
)
CONFIGS_PATH: str | None = None
__SKIP_EXIST = flags.DEFINE_boolean(
    "skip_exist",
    False,
    "Skip existing trace files",
    flag_values=_STAGE_A_FLAGS,
)
SKIP_EXIST: bool | None = None
__DEBUG = flags.DEFINE_boolean(
    "debug",
    False,
    "Debug mode. Disables parallel processing.",
    flag_values=_STAGE_A_FLAGS,
)
__GENERATE_TRACE = flags.DEFINE_boolean(
    "generate_trace",
    True,
    "Generate trace files",
    flag_values=_STAGE_A_FLAGS,
)
__GENERATE_OPT_RESULTS = flags.DEFINE_boolean(
    "generate_opt_results",
    True,
    "Generate optimization results",
    flag_values=_STAGE_A_FLAGS,
)
GENERATE_TRACE: bool | None = None
GENERATE_OPT_RESULTS: bool | None = None
__SLO_SCALE = flags.DEFINE_float(
    "slo_scale", 5.0, "SLO scale factor", flag_values=_STAGE_A_FLAGS
)
SLO_SCALE: float | None = None
__SLO_BASELINE_VERSION = flags.DEFINE_string(
    "slo_baseline_version",
    "5p",
    "NPU version used as the SLO latency baseline. The SLO target is "
    "slo_scale x the latency of the fastest minimal-chip-count deployment of "
    "this version (computed per seqlen pair). Default '5p' preserves the "
    "original behavior.",
    flag_values=_STAGE_A_FLAGS,
)
SLO_BASELINE_VERSION: str | None = None
__MAX_PP = flags.DEFINE_integer(
    "max_pp",
    -1,
    "Maximum pipeline parallelism degree to consider in the parallelism sweep. "
    "-1 = unrestricted (default). Set to 1 to disable pipeline parallelism.",
    flag_values=_STAGE_A_FLAGS,
)
MAX_PP: int | None = None
__OPTIMAL_TOP_K = flags.DEFINE_integer(
    "optimal_top_k",
    5,
    "Number of ranked configurations retained per NPU version and phase. "
    "Use -1 to retain every feasible chip-count alternative.",
    flag_values=_STAGE_A_FLAGS,
)
OPTIMAL_TOP_K: int | None = None


def dump_stats_llm(
    v: str,
    base_config: dict[str, Any],
    ops_gen: LLMOpsGeneratorBase,
    workload: str = "inference",
    dump_to_file: bool = True,
    prefill_ops: list[Operator] | None = None,
    decode_ops: list[Operator] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]] | dict[str, Any]:
    output_file: str = base_config["output_file_path"]
    global_batch_size: int = base_config["global_batch_size"]
    microbatch_size_ici: int = base_config["microbatch_size_ici"]
    pp_dcn: int = base_config["pipeline_parallel_degree_dcn"]
    dp_dcn: int = base_config["data_parallel_degree_dcn"]
    pp: int = base_config["pipeline_parallelism_degree"]

    num_pods = dp_dcn * pp_dcn
    batch_size_per_pod = ceil(global_batch_size / num_pods)
    base_config["num_pods"] = num_pods
    base_config["batch_size_per_pod"] = batch_size_per_pod

    total_pp = pp * pp_dcn
    layers_per_pp_stage = ceil(base_config["num_layers"] / total_pp)
    base_config["layers_per_pp_stage"] = layers_per_pp_stage

    if workload == "inference":
        if prefill_ops and decode_ops:
            prefill_stats = run_sim_lib.get_statistics_from_trace_file(prefill_ops)
            decode_stats = run_sim_lib.get_statistics_from_trace_file(decode_ops)
        else:
            prefill_stats = run_sim_lib.get_statistics_from_trace_file(
                output_file.replace(".csv", "_prefill.csv")
            )
            decode_stats = run_sim_lib.get_statistics_from_trace_file(
                output_file.replace(".csv", "_decode.csv")
            )

        input_seqlen = base_config["input_seqlen"]
        output_seqlen = base_config["output_seqlen"]

        # update prefill stats (throughput:tokens/sec and TTFT:sec)
        prefill_pp_stage_time_ns = ceil(prefill_stats["total_execution_time_chip_ns"])
        prefill_stats["throughput_tokens_per_sec"] = (
            microbatch_size_ici * input_seqlen * 1e9 / prefill_pp_stage_time_ns
        )
        prefill_pod_latency_ns = (
            prefill_stats["total_execution_time_non_pp_ns"]
            + prefill_stats["total_pp_ici_time_ns"]
        ) * pp
        prefill_tot_latency_ns = (
            prefill_pod_latency_ns + prefill_stats["total_pp_dcn_time_ns"]
        ) * pp_dcn
        prefill_stats["TTFT_sec"] = prefill_tot_latency_ns / 1e9

        # update decode stats (throughput:tokens/sec, throughput:tokens/sec/request)
        # tokens/sec/request is the inverse of TPOT (time per output token per request)
        decode_pod_latency_ns = (
            (
                decode_stats["total_execution_time_non_pp_ns"]
                + decode_stats["total_pp_ici_time_ns"]
            )
            * pp
            / output_seqlen
        )
        decode_tot_latency_ns = (
            decode_pod_latency_ns + decode_stats["total_pp_dcn_time_ns"] / output_seqlen
        ) * pp_dcn
        decode_pp_stage_time_ns = ceil(
            decode_stats["total_execution_time_chip_ns"] / output_seqlen
        )
        decode_stats["TPOT_ms_request"] = decode_tot_latency_ns / 1e6
        decode_stats["throughput_tokens_per_sec"] = (
            microbatch_size_ici * 1e9 / decode_pp_stage_time_ns
        )
        decode_stats["throughput_tokens_per_sec_request"] = (
            1e3 / decode_stats["TPOT_ms_request"]
        )
        # decode_stats["throughput_tokens_per_sec"] = microbatch_size_ici * 1e9 / decode_pp_stage_time_ns
        # decode_stats["throughput_tokens_per_sec_request"] = decode_stats["throughput_tokens_per_sec"] / microbatch_size_ici
        # decode_stats["TPOT_ms_request"] = 1e3 / (decode_stats["throughput_tokens_per_sec_request"])

        assert isinstance(
            ops_gen, LLMOpsGeneratorInference | DeepSeekOpsGenerator
        ), f"ops_gen should be LLMOpsGeneratorInference or DeepSeekOpsGenerator, but got {type(ops_gen)}"
        prefill_stats["mem_footprint_GB"] = (
            ops_gen.compute_memory_footprint_bytes("prefill") / 1024 / 1024 / 1024
        )
        prefill_stats["out_of_memory"] = (
            base_config["hbm_size_GB"] < prefill_stats["mem_footprint_GB"]
        )
        decode_stats["mem_footprint_GB"] = (
            ops_gen.compute_memory_footprint_bytes("decode") / 1024 / 1024 / 1024
        )
        decode_stats["out_of_memory"] = (
            base_config["hbm_size_GB"] < decode_stats["mem_footprint_GB"]
        )

        # merge sim configs into stats
        prefill_stats["sim_config"] = base_config
        decode_stats["sim_config"] = base_config

        if dump_to_file:
            json.dump(
                prefill_stats,
                open(output_file.replace(".csv", "_prefill.json"), "w"),
                indent=4,
            )
            json.dump(
                decode_stats,
                open(output_file.replace(".csv", "_decode.json"), "w"),
                indent=4,
            )

        return prefill_stats, decode_stats

    elif workload == "training":
        stats = run_sim_lib.get_statistics_from_trace_file(output_file)

        ## First, process the stats for each pod
        # all exe times should be multiplied by the number of microbatches + initiation interval (II) of the pipeline
        # TODO: currently using num microbatches equal to num pipeline stages. Should change this in the future.
        # II_ICI_PP = pp - 1
        # pp_time_multiplier = num_microbatches_per_pod + II_ICI_PP
        pp_time_multiplier = pp
        stats["total_execution_time_pod_ns"] = (
            stats["total_execution_time_chip_ns"] * pp_time_multiplier
        )
        stats["compute_only_time_pod_ns"] = (
            stats["compute_only_time_chip_ns"] * pp_time_multiplier
        )
        stats["memory_only_time_pod_ns"] = (
            stats["memory_only_time_chip_ns"] * pp_time_multiplier
        )
        stats["ici_bound_time_pod_ns"] = (
            stats["ici_bound_time_chip_ns"] * pp_time_multiplier
        )

        ## Second, process the stats for multiple pods over DCN
        # determine if PP_DCN is communication bound or per-pod computation bound
        total_pp_dcn_time = stats["total_pp_dcn_time_ns"]
        total_pod_time = stats["total_execution_time_pod_ns"]
        if total_pp_dcn_time > total_pod_time:
            # PP_DCN is communication bound
            stats["bounded_by_pp_dcn"] = True
            stats["total_execution_time_ns"] = total_pp_dcn_time
        else:
            # PP_DCN is computation bound
            stats["bounded_by_pp_dcn"] = False
            stats["total_execution_time_ns"] = total_pod_time

        # all exe times should be multiplied by the number of microbatches + initiation interval (II) of the pipeline
        # TODO: currently using num microbatches equal to num pipeline stages. Should change this in the future.
        # II_DCN_PP = pp_dcn - 1
        # pp_dcn_time_multiplier = num_microbatches + II_DCN_PP
        pp_dcn_time_multiplier = pp_dcn
        stats["total_execution_time_ns"] *= pp_dcn_time_multiplier
        # TODO: breakdown (maybe this is not necessary)

        assert isinstance(
            ops_gen, LLMOpsGeneratorTraining
        ), f"ops_gen should be LLMOpsGeneratorTraining, but got {type(ops_gen)}"
        stats["mem_footprint_GB"] = (
            ops_gen.compute_memory_footprint_bytes() / 1024 / 1024 / 1024
        )
        stats["out_of_memory"] = (
            base_config["num_chips"] * base_config["hbm_size_GB"]
            < stats["mem_footprint_GB"]
        )

        # merge sim configs into stats
        stats["sim_config"] = base_config

        if dump_to_file:
            json.dump(stats, open(output_file.replace(".csv", ".json"), "w"), indent=4)

        return stats
    else:
        raise ValueError(f"Invalid workload: {workload}")


def dump_stats_dlrm(
    v: str,
    base_config: dict[str, Any],
    ops_gen: DLRMOpsGenerator,
    workload: str = "inference",
):
    output_file: str = base_config["output_file_path"]
    global_batch_size: int = base_config["global_batch_size"]
    pp_dcn: int = base_config["pipeline_parallel_degree_dcn"]
    dp_dcn: int = base_config["data_parallel_degree_dcn"]
    num_chips: int = base_config["num_chips"]

    num_pods = dp_dcn * pp_dcn
    assert (
        num_pods == 1
    ), f"DLRM only supports one pod for now, but get num_pods={num_pods}"
    batch_size_per_pod = ceil(global_batch_size / num_pods)
    base_config["num_pods"] = num_pods
    base_config["batch_size_per_pod"] = batch_size_per_pod

    if workload == "inference":
        # process stats for individual chips
        for chip_id in range(num_chips):
            chip_output_file = output_file.replace(".csv", f"_chip{chip_id}.csv")
            stats = run_sim_lib.get_statistics_from_trace_file(chip_output_file)

            stats["throughput_requests_per_sec"] = (
                global_batch_size / stats["total_execution_time_chip_ns"] * 1e9
            )
            stats["latency_ns"] = stats["total_execution_time_chip_ns"]

            # merge sim configs into stats
            stats["sim_config"] = base_config

            json.dump(
                stats, open(chip_output_file.replace(".csv", ".json"), "w"), indent=4
            )

        # process overall performance (use the straggler chip's perf. as the overall perf.)
        stats = run_sim_lib.get_statistics_from_trace_file(output_file)
        stats["throughput_requests_per_sec"] = (
            global_batch_size / stats["total_execution_time_chip_ns"] * 1e9
        )
        stats["latency_ns"] = stats["total_execution_time_chip_ns"]
        stats["mem_footprint_GB"] = (
            ops_gen.compute_memory_footprint_bytes() / 1024 / 1024 / 1024
        )
        stats["out_of_memory"] = (
            base_config["num_chips"] * base_config["hbm_size_GB"]
            < stats["mem_footprint_GB"]
        )

        stats["sim_config"] = base_config
        json.dump(stats, open(output_file.replace(".csv", ".json"), "w"), indent=4)

    elif workload == "training":
        raise NotImplementedError("DLRM training is not supported yet")
    else:
        raise ValueError(f"Invalid workload: {workload}")


def dump_stats_stable_diffusion(
    v: str,
    base_config: dict[str, Any],
    ops_gen: DiTOpsGenerator | GLIGENOpsGenerator,
    workload: str = "inference",
):
    if workload == "inference":
        global_batch_size: int = base_config["global_batch_size"]
        output_file: str = base_config["output_file_path"]
        num_diffusion_steps: int = base_config["num_diffusion_steps"]
        stats = run_sim_lib.get_statistics_from_trace_file(output_file)
        if base_config["model_type"] == "gligen":
            total_num_steps: int = base_config["total_num_diffusion_steps"]
        elif base_config["model_type"] == "dit":
            total_num_steps = 1
        else:
            raise ValueError(f"Invalid model type: {base_config['model_type']}")
        stats["throughput_requests_per_sec"] = global_batch_size / (
            stats["total_execution_time_chip_ns"] / 1e9 * total_num_steps
        )
        stats["throughput_step_per_sec_per_request"] = num_diffusion_steps / (
            stats["total_execution_time_chip_ns"] / 1e9
        )
        stats["latency_sec"] = (
            stats["total_execution_time_chip_ns"] / 1e9 * total_num_steps
        )
        stats["latency_step_sec"] = (
            stats["total_execution_time_chip_ns"]
            / num_diffusion_steps
            / 1e9
            / total_num_steps
        )
        stats["mem_footprint_GB"] = (
            ops_gen.compute_memory_footprint_bytes() / 1024 / 1024 / 1024
        )
        stats["out_of_memory"] = (
            base_config["num_chips"] * base_config["hbm_size_GB"]
            < stats["mem_footprint_GB"]
        )

        stats["sim_config"] = base_config
        json.dump(stats, open(output_file.replace(".csv", ".json"), "w"), indent=4)
    elif workload == "training":
        raise NotImplementedError("Stable Diffusion training is not supported yet")
    else:
        raise ValueError(f"Invalid workload: {workload}")


def dump_stats(
    model: str,
    v: str,
    base_config: dict[str, Any],
    ops_gen,
    workload: str = "inference",
    dump_to_file: bool = True,
    prefill_ops: list[Operator] | None = None,
    decode_ops: list[Operator] | None = None,
):
    if "llama" in model.lower() or "deepseek" in model.lower():
        return dump_stats_llm(
            v, base_config, ops_gen, workload, dump_to_file, prefill_ops, decode_ops
        )
    elif "dlrm" in model.lower():
        return dump_stats_dlrm(v, base_config, ops_gen, workload)
    elif "gligen" in model.lower() or "dit" in model.lower():
        return dump_stats_stable_diffusion(v, base_config, ops_gen, workload)
    else:
        raise ValueError(f"Invalid model: {model}")


def analyze_operator_energy(
    ops: list[Operator],
    energy_stats: dict[str, Any],
):
    # get overall energy stats
    energy_stats["total_energy_J"] = sum(
        [op.stats.total_energy_J * op.stats.count for op in ops]
    )
    energy_stats["sa_energy_J"] = sum(
        [
            (op.stats.static_energy_sa_J + op.stats.dynamic_energy_sa_J)
            * op.stats.count
            for op in ops
        ]
    )
    energy_stats["vu_energy_J"] = sum(
        [
            (op.stats.static_energy_vu_J + op.stats.dynamic_energy_vu_J)
            * op.stats.count
            for op in ops
        ]
    )
    energy_stats["sram_energy_J"] = sum(
        [
            (op.stats.static_energy_sram_J + op.stats.dynamic_energy_sram_J)
            * op.stats.count
            for op in ops
        ]
    )
    energy_stats["ici_energy_J"] = sum(
        [
            (op.stats.static_energy_ici_J + op.stats.dynamic_energy_ici_J)
            * op.stats.count
            for op in ops
        ]
    )
    energy_stats["hbm_energy_J"] = sum(
        [
            (op.stats.static_energy_hbm_J + op.stats.dynamic_energy_hbm_J)
            * op.stats.count
            for op in ops
        ]
    )
    energy_stats["other_energy_J"] = sum(
        [
            (op.stats.static_energy_other_J + op.stats.dynamic_energy_other_J)
            * op.stats.count
            for op in ops
        ]
    )

    energy_stats["total_static_energy_J"] = sum(
        [op.stats.static_energy_J * op.stats.count for op in ops]
    )
    energy_stats["static_sa_energy_J"] = sum(
        [op.stats.static_energy_sa_J * op.stats.count for op in ops]
    )
    energy_stats["static_vu_energy_J"] = sum(
        [op.stats.static_energy_vu_J * op.stats.count for op in ops]
    )
    energy_stats["static_sram_energy_J"] = sum(
        [op.stats.static_energy_sram_J * op.stats.count for op in ops]
    )
    energy_stats["static_ici_energy_J"] = sum(
        [op.stats.static_energy_ici_J * op.stats.count for op in ops]
    )
    energy_stats["static_hbm_energy_J"] = sum(
        [op.stats.static_energy_hbm_J * op.stats.count for op in ops]
    )
    energy_stats["static_other_energy_J"] = sum(
        [op.stats.static_energy_other_J * op.stats.count for op in ops]
    )

    energy_stats["total_dynamic_energy_J"] = sum(
        [op.stats.dynamic_energy_J * op.stats.count for op in ops]
    )
    energy_stats["dynamic_sa_energy_J"] = sum(
        [op.stats.dynamic_energy_sa_J * op.stats.count for op in ops]
    )
    energy_stats["dynamic_vu_energy_J"] = sum(
        [op.stats.dynamic_energy_vu_J * op.stats.count for op in ops]
    )
    energy_stats["dynamic_sram_energy_J"] = sum(
        [op.stats.dynamic_energy_sram_J * op.stats.count for op in ops]
    )
    energy_stats["dynamic_ici_energy_J"] = sum(
        [op.stats.dynamic_energy_ici_J * op.stats.count for op in ops]
    )
    energy_stats["dynamic_hbm_energy_J"] = sum(
        [op.stats.dynamic_energy_hbm_J * op.stats.count for op in ops]
    )
    energy_stats["dynamic_other_energy_J"] = sum(
        [op.stats.dynamic_energy_other_J * op.stats.count for op in ops]
    )

    total_exe_time_s = (
        sum([op.stats.execution_time_ns * op.stats.count for op in ops]) / 1e9
    )
    energy_stats["peak_power_W"] = max([op.stats.total_power_W for op in ops])
    energy_stats["avg_power_W"] = energy_stats["total_energy_J"] / total_exe_time_s

    energy_stats["avg_static_power_W"] = (
        energy_stats["total_static_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_static_sa_power_W"] = (
        energy_stats["static_sa_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_static_vu_power_W"] = (
        energy_stats["static_vu_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_static_sram_power_W"] = (
        energy_stats["static_sram_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_static_ici_power_W"] = (
        energy_stats["static_ici_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_static_hbm_power_W"] = (
        energy_stats["static_hbm_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_static_other_power_W"] = (
        energy_stats["static_other_energy_J"] / total_exe_time_s
    )

    energy_stats["avg_dynamic_power_W"] = (
        energy_stats["total_dynamic_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_dynamic_sa_power_W"] = (
        energy_stats["dynamic_sa_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_dynamic_vu_power_W"] = (
        energy_stats["dynamic_vu_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_dynamic_sram_power_W"] = (
        energy_stats["dynamic_sram_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_dynamic_ici_power_W"] = (
        energy_stats["dynamic_ici_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_dynamic_hbm_power_W"] = (
        energy_stats["dynamic_hbm_energy_J"] / total_exe_time_s
    )
    energy_stats["avg_dynamic_other_power_W"] = (
        energy_stats["dynamic_other_energy_J"] / total_exe_time_s
    )


def generate(
    model: str,
    v: str,
    parallelism_config: dict[str, int],
    global_batch_size: int,
    skip_exist: bool,
    output_dir: str,
    workload: str,
    configs_path: str,
) -> (
    tuple[str, str, list[Operator], list[Operator], dict[str, Any], dict[str, Any]]
    | None
):
    options: dict[str, Any] = {**parallelism_config}
    dp = parallelism_config["data_parallelism_degree"]
    tp = parallelism_config["tensor_parallelism_degree"]
    pp = parallelism_config["pipeline_parallelism_degree"]
    dp_dcn = parallelism_config["data_parallel_degree_dcn"]
    tp_dcn = 1  # TODO: For now, only support tensor parallelism on ICI since TP has high communication traffic
    pp_dcn = parallelism_config["pipeline_parallel_degree_dcn"]

    if "expert_parallelism_degree" in parallelism_config:
        ep = parallelism_config["expert_parallelism_degree"]
        pstr = run_sim_lib.get_pstr_moe(
            dp, tp, pp, ep, dp_dcn, tp_dcn, pp_dcn, 1, global_batch_size
        )
    else:
        ep = 1
        pstr = run_sim_lib.get_pstr_llm(
            dp, tp, pp, dp_dcn, tp_dcn, pp_dcn, global_batch_size
        )

    # read base model and NPU configs
    base_model_config_path = os.path.join(configs_path, f"models/{model}.json")
    base_npu_config_path = os.path.join(configs_path, f"chips/tpuv{v}.json")
    base_model_config = json.load(open(base_model_config_path))
    base_npu_config = json.load(open(base_npu_config_path))
    base_sys_config = json.load(
        open(os.path.join(configs_path, "systems/system_config.json"))
    )

    # create config for the ops generator
    base_config = {**base_model_config, **base_npu_config, **base_sys_config}
    base_config.update(options)

    ## Determine microbatch sizes for ICI and DCN. Determine # of NPU pods and batch size per pod.
    microbatch_size_ici = ceil(global_batch_size / dp_dcn / pp / pp_dcn)
    microbatch_size_dcn = ceil(global_batch_size / pp_dcn)

    base_config["global_batch_size"] = global_batch_size
    base_config["microbatch_size_ici"] = microbatch_size_ici
    base_config["microbatch_size_dcn"] = microbatch_size_dcn

    ## set output file path
    base_config["output_file_path"] = os.path.join(
        output_dir,
        f"{model}/{pstr}/{workload}-v{v}.csv",
    )
    if skip_exist and os.path.exists(base_config["output_file_path"]):
        logging.info(f"Skipping {model} v{v} {pstr} since the output file exists")
        return

    prefill_valid = run_sim_lib.validate_parallelism_config(
        model, v, base_config, workload, prefill_or_decode="prefill"
    )
    decode_valid = run_sim_lib.validate_parallelism_config(
        model, v, base_config, workload, prefill_or_decode="decode"
    )
    if not prefill_valid and not decode_valid:
        # run them if at least one of them is valid
        # both valid flags will be the same if workload == training.
        logging.info(
            f"Skipping {model} v{v} {pstr} since the parallelism config is invalid"
        )
        return

    ## Map parallelism degrees to physical ICI axes
    axes_mappings = run_sim_lib.map_parallelism_to_ici_axes(
        model, v, parallelism_config
    )

    if "deepseek" in model.lower():
        assert (
            len(axes_mappings) == 4
        ), f"DeepSeek model {model} v{v} should have 4 axes mappings (dp, tp, pp, ep), but got {len(axes_mappings)}"
        dp_axes, tp_axes, pp_axes, ep_axes = axes_mappings
    else:
        assert (
            len(axes_mappings) == 3
        ), f"Model {model} v{v} should have 3 axes mappings (dp, tp, pp), but got {len(axes_mappings)}"
        dp_axes, tp_axes, pp_axes = axes_mappings
        ep_axes = 0

    # update parallelism axes in the config
    base_config["num_data_parallel_axes"] = dp_axes
    base_config["num_tensor_parallel_axes"] = tp_axes
    base_config["num_pipeline_parallel_axes"] = pp_axes
    if "deepseek" in model.lower():
        base_config["num_expert_parallel_axes"] = ep_axes

    ## create output dir
    # os.makedirs(os.path.dirname(base_config["output_file_path"]), exist_ok=True)

    ## run the ops generator
    if "llama" in model.lower():
        base_config = LLMConfig.model_validate(base_config)
        base_config.enable_swap_kv_cache = True
        if workload == "training":
            # TODO: flashattention is not supported for training yet
            base_config.use_flash_attention = False
            ops_generator = LLMOpsGeneratorTraining(base_config)
        elif workload == "inference":
            # enable flash attention
            # base_config["use_flash_attention"] = True
            ops_generator = LLMOpsGeneratorInference(base_config)
        else:
            raise ValueError(f"Invalid workload: {workload}")
    elif "deepseek" in model.lower():
        base_config = DeepSeekConfig.model_validate(base_config)
        base_config.enable_swap_kv_cache = True
        if workload == "training":
            raise NotImplementedError("DeepSeek training is not supported yet")
        elif workload == "inference":
            ops_generator = DeepSeekOpsGenerator(base_config)
        else:
            raise ValueError(f"Invalid workload: {workload}")
    elif "dlrm" in model.lower():
        if workload == "training":
            raise NotImplementedError("DLRM training is not supported yet")
        elif workload == "inference":
            ops_generator = DLRMOpsGenerator(base_config)
        else:
            raise ValueError(f"Invalid workload: {workload}")
    elif "gligen" in model.lower():
        if workload == "training":
            raise NotImplementedError("Stable Diffusion training is not supported yet")
        elif workload == "inference":
            ops_generator = GLIGENOpsGenerator(base_config)
        else:
            raise ValueError(f"Invalid workload: {workload}")
    elif "dit" in model.lower():
        if workload == "training":
            raise NotImplementedError("Stable Diffusion training is not supported yet")
        elif workload == "inference":
            ops_generator = DiTOpsGenerator(base_config)
        else:
            raise ValueError(f"Invalid workload: {workload}")
    else:
        raise ValueError(f"Invalid model: {model}")
    logging.info(f"Generating {model} {workload} v{v} {pstr}...")
    # TODO: support other models in the future
    assert isinstance(
        ops_generator, LLMOpsGeneratorInference | DeepSeekOpsGenerator
    ), f"Expected LLMOpsGeneratorInference or DeepSeekOpsGenerator, but got {type(ops_generator)}"
    ops, prefill_ops, decode_ops = ops_generator.generate(
        dump_to_file=False, separate_prefill_decode=True, analyze_energy=True
    )
    assert isinstance(prefill_ops, list)
    assert isinstance(decode_ops, list)

    # dump config
    # os.makedirs(os.path.join(CONFIGS_PATH, "generated", model, pstr), exist_ok=True)
    # with open(os.path.join(CONFIGS_PATH, "generated", model, pstr, f"{workload}-v{v}.json"), "w") as f:
    #     json.dump(ops_generator.config.model_dump(mode="json"), f, indent=4)  # type: ignore

    ## do not dump, only get stats
    prefill_stats, decode_stats = dump_stats(
        model,
        v,
        ops_generator.config.model_dump(mode="json"),
        ops_generator,
        workload,
        dump_to_file=False,
        prefill_ops=prefill_ops,
        decode_ops=decode_ops,
    )  # type: ignore
    assert isinstance(prefill_stats, dict)
    assert isinstance(decode_stats, dict)

    prefill_stats["energy_stats"] = {}
    decode_stats["energy_stats"] = {}
    analyze_operator_energy(prefill_ops, prefill_stats["energy_stats"])
    analyze_operator_energy(decode_ops, decode_stats["energy_stats"])

    def analyze_carbon_and_energy_stats(stats: dict[str, Any]):
        energy_stats = stats["energy_stats"]
        avg_power_W = energy_stats["avg_power_W"] * ops_generator.config.num_chips
        goodput = stats["throughput_tokens_per_sec"]
        eff = goodput / (ops_generator.config.PUE * avg_power_W)
        stats["avg_power_efficiency_tkn_per_joule"] = eff

    def analyze_monetary_cost_stats(stats: dict[str, Any]):
        goodput = stats["throughput_tokens_per_sec"]
        cost_per_chip_hour = results_lib.VERSION_TO_COST[v]  # chip*hour
        cost_per_sec = (
            cost_per_chip_hour / 3600 * ops_generator.config.num_chips
        )  # $/(chip*hour)/(3600sec/hour)*chip => $/sec
        eff = goodput / cost_per_sec  # tkn/$
        stats["monetary_cost_tkn_per_dollar"] = eff

    analyze_carbon_and_energy_stats(prefill_stats)
    analyze_carbon_and_energy_stats(decode_stats)
    analyze_monetary_cost_stats(prefill_stats)
    analyze_monetary_cost_stats(decode_stats)

    return (model, v, prefill_ops, decode_ops, prefill_stats, decode_stats)


def generate_wrapper(row: dict[str, Any]):
    result = generate(*row["item"])
    if result:
        model, v, prefill_ops, decode_ops, prefill_stats, decode_stats = result
        # return {
        #     "model": model,
        #     "v": v,
        #     "prefill_ops": prefill_ops,
        #     "decode_ops": decode_ops,
        #     "prefill_stats": prefill_stats,
        #     "decode_stats": decode_stats,
        # }
        mscfg_str = f"{model}_{prefill_stats['sim_config']['input_seqlen']}_{decode_stats['sim_config']['output_seqlen']}"
        return {
            "item": (model, v, prefill_ops, decode_ops, prefill_stats, decode_stats),
            "model_seqlen_config_key": mscfg_str,
            "model_seqlen_config": (
                model,
                (
                    prefill_stats["sim_config"]["input_seqlen"],
                    decode_stats["sim_config"]["output_seqlen"],
                ),
            ),
        }
    else:
        # return {"model": None}
        return {
            "item": None,
            "model_seqlen_config_key": None,
            "model_seqlen_config": None,
        }


RunConfig = tuple[str, str, dict[str, Any], int, bool, str, str, str]


def build_run_configs(
    model: str,
    seq_lens: Sequence[tuple[int, int]],
    *,
    versions: Sequence[str],
    num_chips_list: Sequence[str | int],
    batch_sizes: Sequence[str | int],
    skip_exist: bool,
    output_dir: str,
    workload: str,
    max_pp: int,
    configs_path: str | os.PathLike[str] = REPO_ROOT / "configs",
    parallelism_generator: Callable[..., Sequence[dict[str, Any]]] | None = None,
) -> list[RunConfig]:
    """Expand chip, sequence-length, version, and batch-size sweep dimensions."""
    generator = parallelism_generator or run_sim_lib.generate_parallelism_configs
    parallelism_configs: list[dict[str, Any]] = []
    for num_chips_value in num_chips_list:
        num_chips = int(num_chips_value)
        dcn_counts = (1, 2, 4) if num_chips > 256 else (1,)
        for dcn_count in dcn_counts:
            parallelism_configs.extend(
                generator(
                    num_chips // dcn_count,
                    model,
                    max_dp=1,
                    max_num_dcn_pods=dcn_count,
                    max_etp=8,
                    max_pp=max_pp,
                )
            )

    logging.info(
        "# parallelism configs (model=%s): %s", model, len(parallelism_configs)
    )
    all_configs: list[RunConfig] = []
    for input_seqlen, output_seqlen in seq_lens:
        for parallelism_config in parallelism_configs:
            config = deepcopy(parallelism_config)
            config["input_seqlen"] = input_seqlen
            config["output_seqlen"] = output_seqlen
            for version in versions:
                for batch_size in batch_sizes:
                    all_configs.append(
                        (
                            model,
                            version,
                            config,
                            int(batch_size),
                            skip_exist,
                            output_dir,
                            workload,
                            os.fspath(configs_path),
                        )
                    )
    return all_configs


def generate_all_run_configs(
    model: str, seq_lens: Sequence[tuple[int, int]]
) -> list[RunConfig]:
    """Generate the configured Stage-A sweep for model."""
    assert OUTPUT_DIR
    assert WORKLOAD
    assert MODELS
    assert VERSIONS
    assert NUM_CHIPS_LIST
    assert INFERENCE_BATCH_SIZES
    assert TRAINING_BATCH_SIZES
    assert SKIP_EXIST is not None
    assert MAX_PP is not None

    batch_sizes = (
        TRAINING_BATCH_SIZES if WORKLOAD == "training" else INFERENCE_BATCH_SIZES
    )
    return build_run_configs(
        model,
        seq_lens,
        versions=VERSIONS,
        num_chips_list=NUM_CHIPS_LIST,
        batch_sizes=batch_sizes,
        skip_exist=SKIP_EXIST,
        output_dir=OUTPUT_DIR,
        workload=WORKLOAD,
        max_pp=MAX_PP,
        configs_path=CONFIGS_PATH or REPO_ROOT / "configs",
    )


def select_ranked_config_alternatives(
    configs: Sequence[tuple[list[Operator], dict[str, Any]]],
    cost_eff_key: str,
    top_k: int,
) -> list[tuple[list[Operator], dict[str, Any]]]:
    """Rank configurations by efficiency, optionally retaining every one.

    Stage-A has already reduced the raw sweep to the best feasible configuration
    for each chip count. ``top_k=-1`` keeps all of those chip-count alternatives,
    allowing FleetSim to choose smaller configurations under a fleet chip cap.
    """
    if top_k == -1:
        return sorted(
            configs,
            key=lambda item: item[1][cost_eff_key],
            reverse=True,
        )
    if top_k <= 0:
        raise ValueError("top_k must be -1 or a positive integer")
    return heapq.nlargest(top_k, configs, key=lambda item: item[1][cost_eff_key])


def reset_ranked_output_directory(directory: Path) -> None:
    """Remove only numbered rank files before publishing a new ranking."""
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and path.suffix in {".json", ".csv"} and path.stem.isdigit():
            path.unlink()


# @ray.remote
def find_cost_optimal_config_for_model(
    row: dict[str, Any],
    *,
    result_list_override: list[dict[str, Any]] | None = None,
):
    global OUTPUT_DIR
    global VERSIONS
    assert OUTPUT_DIR
    assert VERSIONS
    model, input_seqlen, output_seqlen = row["item"]

    # read from cached pickle file
    cache_dir = os.path.join(OUTPUT_DIR, ".cache")
    cached_file = os.path.join(
        cache_dir, f"{model}_{input_seqlen}_{output_seqlen}.pkl.gz"
    )
    os.makedirs(cache_dir, exist_ok=True)

    def generate_simulation_results():
        params = generate_all_run_configs(model, [(input_seqlen, output_seqlen)])
        logging.info("model: %s, # of traces: %d", model, len(params))
        ray = _get_ray()
        param_ds = ray.data.from_items(params)
        result_ds = param_ds.map(generate_wrapper)
        result_ds = result_ds.filter(lambda x: x["item"] is not None)

        # take out the list from result_ds and dump to a gzipped pickle file
        result_list = result_ds.take_all()
        assert len(
            result_list
        ), f"No valid results found for {model}, {input_seqlen}, {output_seqlen}"
        temporary_file = f"{cached_file}.tmp.{os.getpid()}"
        try:
            with gzip.open(temporary_file, "wb") as f:
                pickle.dump(result_list, f)
            os.replace(temporary_file, cached_file)
            logging.info("Dumped results to %s", cached_file)
        finally:
            if os.path.exists(temporary_file):
                os.unlink(temporary_file)
        return result_list

    if result_list_override is not None:
        if not result_list_override:
            raise ValueError("result_list_override must contain at least one result")
        result_list = result_list_override
    else:
        try:
            # raise FileNotFoundError()  # TODO: temp override
            if os.path.exists(cached_file):
                with gzip.open(cached_file, "rb") as f:
                    result_list = pickle.load(f)
                    logging.info("Loaded results from %s", f.name)
            else:
                if GENERATE_OPT_RESULTS and not GENERATE_TRACE:
                    logging.warning("No cached results found for %s", cached_file)
                    return {"item": None}
                result_list = generate_simulation_results()
        except (FileNotFoundError, EOFError, zlib.error, gzip.BadGzipFile):
            if GENERATE_OPT_RESULTS and not GENERATE_TRACE:
                logging.warning("No valid cached results found for %s", cached_file)
                return {"item": None}
            result_list = generate_simulation_results()

    if not GENERATE_OPT_RESULTS:
        return {"item": None}  # generate optimization results first

    # find slo config (5x latency of batch size 1 with min num chips of v5p)
    def find_slo_config():
        """Return ((prefill ops, prefill stats), (decode ops, decode stats)) for the SLO configs."""
        # SLO baseline version (default "5p"; configurable via --slo_baseline_version).
        slo_baseline_version = SLO_BASELINE_VERSION
        prefill_configs = []
        decode_configs = []
        for row in result_list:
            _, v, prefill_ops, decode_ops, prefill_stats, decode_stats = row["item"]
            if v == slo_baseline_version:
                if not prefill_stats["out_of_memory"]:
                    # add a hard constraint for SLO to prevent too large SLO
                    # max 10min for prefill (in case we have very long seqlen)
                    # if prefill_stats["TTFT_sec"] <= 10 * 60:
                    prefill_configs.append((prefill_ops, prefill_stats))
                if not decode_stats["out_of_memory"]:
                    # add a hard constraint for SLO to prevent too large SLO
                    # max 200ms per token (5tkn/sec should be sufficient for deep research tasks)
                    # if decode_stats["TPOT_ms_request"] <= 200:
                    decode_configs.append((decode_ops, decode_stats))
        if not prefill_configs or not decode_configs:
            # No non-OOM baseline config (e.g. long context that does not fit when
            # pipeline parallelism is restricted). Skip this seqlen pair gracefully
            # instead of asserting, which would abort the whole Ray Data job.
            return None
        prefill_min_num_chips = min(
            [x[1]["sim_config"]["num_chips"] for x in prefill_configs]
        )
        decode_min_num_chips = min(
            [x[1]["sim_config"]["num_chips"] for x in decode_configs]
        )
        prefill_configs = filter(
            lambda x: x[1]["sim_config"]["num_chips"] == prefill_min_num_chips,
            prefill_configs,
        )
        decode_configs = filter(
            lambda x: x[1]["sim_config"]["num_chips"] == decode_min_num_chips,
            decode_configs,
        )

        # get prefill with best TTFT_sec
        min_prefill_config = min(prefill_configs, key=lambda x: x[1]["TTFT_sec"])
        # get decode with best TPOT_ms_request
        min_decode_config = min(decode_configs, key=lambda x: x[1]["TPOT_ms_request"])

        return min_prefill_config, min_decode_config

    slo_configs = find_slo_config()
    if slo_configs is None:
        logging.warning(
            "No valid %s SLO-baseline config for %s %d_%d (likely all OOM); skipping pair.",
            SLO_BASELINE_VERSION,
            model,
            input_seqlen,
            output_seqlen,
        )
        return {"item": None}
    slo_prefill_config, slo_decode_config = slo_configs
    prefill_slo = slo_prefill_config[1]["TTFT_sec"] * SLO_SCALE
    decode_slo = slo_decode_config[1]["TPOT_ms_request"] * SLO_SCALE

    # add a hard constraint for SLO to prevent too large SLO
    # prefill_slo = min(prefill_slo, 10 * 60)  # max 10min for prefill (in case we have very long seqlen)
    # decode_slo = min(decode_slo, 200)  # max 200ms per token (5tkn/sec should be sufficient for deep research tasks)

    def find_optimal_config(cost_eff_key: str):
        """
        Return the configured number of ranked configs for each NPU version.
        @return: tuple of map v -> list of optimal configs [(best_ops, best_stats), (2nd best ops, 2nd best stats), ...]
        """
        global VERSIONS
        global OPTIMAL_TOP_K
        assert VERSIONS
        assert OPTIMAL_TOP_K is not None
        prefill_configs = {v: {} for v in VERSIONS}
        decode_configs = {v: {} for v in VERSIONS}
        # Track the fastest config per version as fallback when no config meets SLO.
        fastest_prefill: dict[str, tuple | None] = {v: None for v in VERSIONS}
        fastest_decode: dict[str, tuple | None] = {v: None for v in VERSIONS}

        for row in result_list:
            model, v, prefill_ops, decode_ops, prefill_stats, decode_stats = row["item"]

            # Skip versions not requested (cache may contain all versions).
            if v not in VERSIONS:
                continue

            prefill_stats["slo_TTFT_sec"] = prefill_slo
            prefill_stats["slo_scale"] = SLO_SCALE
            decode_stats["slo_TPOT_ms_request"] = decode_slo
            decode_stats["slo_scale"] = SLO_SCALE

            prefill_latency = prefill_stats["TTFT_sec"]
            decode_latency = decode_stats["TPOT_ms_request"]

            prefill_num_chips = prefill_stats["sim_config"]["num_chips"]
            decode_num_chips = decode_stats["sim_config"]["num_chips"]

            # Track fastest regardless of SLO (for fallback).
            if not prefill_stats["out_of_memory"]:
                if (
                    fastest_prefill[v] is None
                    or prefill_latency < fastest_prefill[v][1]["TTFT_sec"]
                ):  # type: ignore
                    fastest_prefill[v] = (prefill_ops, prefill_stats)
            if not decode_stats["out_of_memory"]:
                if (
                    fastest_decode[v] is None
                    or decode_latency < fastest_decode[v][1]["TPOT_ms_request"]
                ):  # type: ignore
                    fastest_decode[v] = (decode_ops, decode_stats)

            # only keep the most efficient one for each num_chips
            if prefill_latency <= prefill_slo:
                if prefill_num_chips not in prefill_configs[v]:
                    prefill_configs[v][prefill_num_chips] = (prefill_ops, prefill_stats)
                prefill_eff = prefill_stats[cost_eff_key]
                if prefill_eff > prefill_configs[v][prefill_num_chips][1][cost_eff_key]:
                    prefill_configs[v][prefill_num_chips] = (prefill_ops, prefill_stats)

            if decode_latency <= decode_slo:
                if decode_num_chips not in decode_configs[v]:
                    decode_configs[v][decode_num_chips] = (decode_ops, decode_stats)
                decode_eff = decode_stats[cost_eff_key]
                if decode_eff > decode_configs[v][decode_num_chips][1][cost_eff_key]:
                    decode_configs[v][decode_num_chips] = (decode_ops, decode_stats)

        # flatten the prefill/decode configs dict
        prefill_optimal_configs = {
            v: list(prefill_configs[v].values()) for v in VERSIONS
        }
        decode_optimal_configs = {v: list(decode_configs[v].values()) for v in VERSIONS}

        # Fallback: if no config meets SLO for a version, use the fastest one.
        for v in VERSIONS:
            if not prefill_optimal_configs[v] and fastest_prefill[v] is not None:
                logging.warning(
                    "No prefill config meets SLO (%.3fs) for %s/%s/%s_%s v%s [%s]. "
                    "Falling back to fastest (TTFT=%.3fs).",
                    prefill_slo,
                    cost_eff_key,
                    model,
                    input_seqlen,
                    output_seqlen,
                    v,
                    fastest_prefill[v][1]["TTFT_sec"],  # type: ignore
                )
                prefill_optimal_configs[v] = [fastest_prefill[v]]
            if not decode_optimal_configs[v] and fastest_decode[v] is not None:
                logging.warning(
                    "No decode config meets SLO (%.3fms) for %s/%s/%s_%s v%s [%s]. "
                    "Falling back to fastest (TPOT=%.3fms).",
                    decode_slo,
                    cost_eff_key,
                    model,
                    input_seqlen,
                    output_seqlen,
                    v,
                    fastest_decode[v][1]["TPOT_ms_request"],  # type: ignore
                )
                decode_optimal_configs[v] = [fastest_decode[v]]

        # Retain enough chip-count alternatives for capped fleets.
        for v in VERSIONS:
            prefill_optimal_configs[v] = select_ranked_config_alternatives(
                prefill_optimal_configs[v], cost_eff_key, OPTIMAL_TOP_K
            )
            decode_optimal_configs[v] = select_ranked_config_alternatives(
                decode_optimal_configs[v], cost_eff_key, OPTIMAL_TOP_K
            )

        return prefill_optimal_configs, decode_optimal_configs

    (
        energy_prefill_optimal_configs,
        energy_decode_optimal_configs,
    ) = find_optimal_config("avg_power_efficiency_tkn_per_joule")
    (
        monetary_cost_prefill_optimal_configs,
        monetary_cost_decode_optimal_configs,
    ) = find_optimal_config("monetary_cost_tkn_per_dollar")

    # dump these ops and stats to files to OUTPUT_DIR/{energy,monetary}/model/v/{prefill,decode}/{1,2,...,k}.{json,csv}
    for v in VERSIONS:
        for goal in ("energy", "monetary"):
            for phase in ("prefill", "decode"):
                reset_ranked_output_directory(
                    Path(OUTPUT_DIR)
                    / goal
                    / model
                    / f"{input_seqlen}_{output_seqlen}"
                    / v
                    / phase
                )
        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                "energy",
                model,
                f"{input_seqlen}_{output_seqlen}",
                v,
                "prefill",
            ),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                "energy",
                model,
                f"{input_seqlen}_{output_seqlen}",
                v,
                "decode",
            ),
            exist_ok=True,
        )
        for i, (ops, stats) in enumerate(energy_prefill_optimal_configs[v]):
            with open(
                os.path.join(
                    OUTPUT_DIR,
                    "energy",
                    model,
                    f"{input_seqlen}_{output_seqlen}",
                    v,
                    "prefill",
                    f"{i+1}.json",
                ),
                "w",
            ) as f:
                json.dump(stats, f, indent=4)
            with open(
                os.path.join(
                    OUTPUT_DIR,
                    "energy",
                    model,
                    f"{input_seqlen}_{output_seqlen}",
                    v,
                    "prefill",
                    f"{i+1}.csv",
                ),
                "w",
            ) as f:
                ops_dict = [op.to_csv_dict() for op in ops]
                writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(ops_dict)
        for i, (ops, stats) in enumerate(energy_decode_optimal_configs[v]):
            with open(
                os.path.join(
                    OUTPUT_DIR,
                    "energy",
                    model,
                    f"{input_seqlen}_{output_seqlen}",
                    v,
                    "decode",
                    f"{i+1}.json",
                ),
                "w",
            ) as f:
                json.dump(stats, f, indent=4)
            with open(
                os.path.join(
                    OUTPUT_DIR,
                    "energy",
                    model,
                    f"{input_seqlen}_{output_seqlen}",
                    v,
                    "decode",
                    f"{i+1}.csv",
                ),
                "w",
            ) as f:
                ops_dict = [op.to_csv_dict() for op in ops]
                writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(ops_dict)
        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                "monetary",
                model,
                f"{input_seqlen}_{output_seqlen}",
                v,
                "prefill",
            ),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                "monetary",
                model,
                f"{input_seqlen}_{output_seqlen}",
                v,
                "decode",
            ),
            exist_ok=True,
        )
        for i, (ops, stats) in enumerate(monetary_cost_prefill_optimal_configs[v]):
            with open(
                os.path.join(
                    OUTPUT_DIR,
                    "monetary",
                    model,
                    f"{input_seqlen}_{output_seqlen}",
                    v,
                    "prefill",
                    f"{i+1}.json",
                ),
                "w",
            ) as f:
                json.dump(stats, f, indent=4)
            with open(
                os.path.join(
                    OUTPUT_DIR,
                    "monetary",
                    model,
                    f"{input_seqlen}_{output_seqlen}",
                    v,
                    "prefill",
                    f"{i+1}.csv",
                ),
                "w",
            ) as f:
                ops_dict = [op.to_csv_dict() for op in ops]
                writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(ops_dict)
        for i, (ops, stats) in enumerate(monetary_cost_decode_optimal_configs[v]):
            with open(
                os.path.join(
                    OUTPUT_DIR,
                    "monetary",
                    model,
                    f"{input_seqlen}_{output_seqlen}",
                    v,
                    "decode",
                    f"{i+1}.json",
                ),
                "w",
            ) as f:
                json.dump(stats, f, indent=4)
            with open(
                os.path.join(
                    OUTPUT_DIR,
                    "monetary",
                    model,
                    f"{input_seqlen}_{output_seqlen}",
                    v,
                    "decode",
                    f"{i+1}.csv",
                ),
                "w",
            ) as f:
                ops_dict = [op.to_csv_dict() for op in ops]
                writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(ops_dict)

    # return (prefill_best_ops, decode_best_ops, prefill_best_stats, decode_best_stats)
    return {"item": None}


def init_cmd_args():
    """Initialize command line arguments since ray remote function cannot serialize absl flags variables."""
    global CONFIGS_PATH
    global OUTPUT_DIR
    global WORKLOAD
    global MODELS
    global VERSIONS
    global NUM_CHIPS_LIST
    global INFERENCE_BATCH_SIZES
    global TRAINING_BATCH_SIZES
    global SKIP_EXIST
    global GENERATE_TRACE
    global GENERATE_OPT_RESULTS
    global SLO_SCALE
    global SLO_BASELINE_VERSION
    global MAX_PP
    global OPTIMAL_TOP_K
    CONFIGS_PATH = __CONFIGS_PATH.value
    OUTPUT_DIR = __OUTPUT_DIR.value
    WORKLOAD = __WORKLOAD.value
    MODELS = __MODELS.value
    VERSIONS = __VERSIONS.value
    NUM_CHIPS_LIST = __NUM_CHIPS_LIST.value
    INFERENCE_BATCH_SIZES = __INFERENCE_BATCH_SIZES.value
    TRAINING_BATCH_SIZES = __TRAINING_BATCH_SIZES.value
    SKIP_EXIST = __SKIP_EXIST.value
    GENERATE_TRACE = __GENERATE_TRACE.value
    GENERATE_OPT_RESULTS = __GENERATE_OPT_RESULTS.value
    SLO_SCALE = __SLO_SCALE.value
    SLO_BASELINE_VERSION = __SLO_BASELINE_VERSION.value
    MAX_PP = __MAX_PP.value
    OPTIMAL_TOP_K = __OPTIMAL_TOP_K.value
    if OPTIMAL_TOP_K == 0 or OPTIMAL_TOP_K < -1:
        raise ValueError("--optimal_top_k must be -1 or a positive integer")

    assert (
        GENERATE_TRACE or GENERATE_OPT_RESULTS
    ), "At least one of GENERATE_TRACE or GENERATE_OPT_RESULTS must be enabled."


def read_trace_seqlens(
    trace_file: str | os.PathLike[str],
) -> list[tuple[int, int]]:
    """Read raw input/output token counts using the CSV header, not its path."""
    with open(trace_file, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        azure_columns = {"TIMESTAMP", "ContextTokens", "GeneratedTokens"}
        burst_columns = {"Timestamp", "Request tokens", "Response tokens"}
        if azure_columns <= fields:
            input_column, output_column, schema = (
                "ContextTokens",
                "GeneratedTokens",
                "Azure",
            )
        elif burst_columns <= fields:
            input_column, output_column, schema = (
                "Request tokens",
                "Response tokens",
                "BurstGPT",
            )
        else:
            raise ValueError(
                f"Unsupported trace CSV schema in {trace_file}: {sorted(fields)}"
            )
        rows = [(int(row[input_column]), int(row[output_column])) for row in reader]
    if not rows:
        raise ValueError(f"Trace file contains no requests: {trace_file}")
    print(f"Processing {schema}-schema trace file {trace_file}")
    return rows


def read_input_output_seqlen_from_trace_file(
    use_cache: bool = True,
) -> set[tuple[int, int]]:
    """@return: [(input_seqlen, output_seqlen)]"""
    cache_dir = os.path.join(__OUTPUT_DIR.value, ".cache")
    if use_cache:
        if os.path.exists(os.path.join(cache_dir, "input_output_seqlens.pkl")):
            with open(os.path.join(cache_dir, "input_output_seqlens.pkl"), "rb") as f:
                all_seqlens = pickle.load(f)
                print(f"Loaded {len(all_seqlens)} seq lens from cache")
                return all_seqlens

    all_seqlens = set()

    def pad(seqlen: int, input_output: bool) -> int:
        """@input_output: True for input, False for output."""
        return pad_seqlen(
            seqlen,
            [
                int(x)
                for x in (
                    __INPUT_SEQLEN_PADDING_FACTORS.value
                    if input_output
                    else __OUTPUT_SEQLEN_PADDING_FACTORS.value
                )
            ],
            [
                int(x)
                for x in (
                    __INPUT_SEQLEN_PADDING_STEPS.value
                    if input_output
                    else __OUTPUT_SEQLEN_PADDING_STEPS.value
                )
            ],
        )

    for trace_file in __REQUEST_TRACE_FILE.value:
        seqlens = [
            (pad(input_seqlen, True), pad(output_seqlen, False))
            for input_seqlen, output_seqlen in tqdm.tqdm(read_trace_seqlens(trace_file))
        ]

        # add average prefill and decode seqlen pair
        avg_prefill = math.ceil(sum([x[0] for x in seqlens]) / len(seqlens))
        avg_decode = math.ceil(sum([x[1] for x in seqlens]) / len(seqlens))
        avg_prefill = pad(avg_prefill, True)
        avg_decode = pad(avg_decode, False)
        seqlens.append((avg_prefill, avg_decode))

        # add mean+0.25std seqlen pair
        input_seqlen_std = np.std([x[0] for x in seqlens])
        output_seqlen_std = np.std([x[1] for x in seqlens])
        avg_input_seqlen = math.ceil(
            np.mean([x[0] for x in seqlens]) + 0.25 * input_seqlen_std
        )
        avg_output_seqlen = math.ceil(
            np.mean([x[1] for x in seqlens]) + 0.25 * output_seqlen_std
        )
        avg_input_seqlen = pad(avg_input_seqlen, True)
        avg_output_seqlen = pad(avg_output_seqlen, False)
        seqlens.append((avg_input_seqlen, avg_output_seqlen))

        all_seqlens = all_seqlens.union(seqlens)

    # # manually 4x the seqlen to cover long seq len for now
    # all_seqlens = all_seqlens.union({
    #     (x[0] * 4, x[1] * 4) for x in all_seqlens
    # })
    # manually //4 for LVEval for now
    # all_seqlens = {(x[0] // 4, 4) for x in all_seqlens}

    # # manually try 64K, 128K, 256K, 512K seqlen
    # all_seqlens = all_seqlens.union({
    #     (64 * 1024, 32),
    #     (128 * 1024, 32),
    #     (256 * 1024, 32),
    #     (512 * 1024, 32),
    # })

    # add output seq len == 4 for every prefill seq len.
    all_seqlens = all_seqlens.union(set([(x[0], 4) for x in all_seqlens]))

    # dump the seq lens to a cache pickle file
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "input_output_seqlens.pkl"), "wb") as f:
        pickle.dump(all_seqlens, f)

    return all_seqlens


def verify_trace_from_cached_file(
    model: str, input_seqlen: int, output_seqlen: int
) -> bool:
    """Read the trace from cached pickle file."""
    global OUTPUT_DIR
    assert OUTPUT_DIR

    cache_dir = os.path.join(OUTPUT_DIR, ".cache")
    cached_file = os.path.join(
        cache_dir, f"{model}_{input_seqlen}_{output_seqlen}.pkl.gz"
    )
    try:
        if os.path.exists(cached_file):
            with gzip.open(cached_file, "rb") as f:
                pickle.load(f)
                logging.info("Loaded results from %s", f.name)
                return True
        else:
            logging.warning("No cached results found for %s", cached_file)
            return False
    except (FileNotFoundError, EOFError, zlib.error, gzip.BadGzipFile):
        logging.warning("No valid cached results found for %s", cached_file)
        return False


def verify_trace_from_cached_file_wrapper(row: dict[str, Any]) -> dict[str, Any]:
    model, (input_seqlen, output_seqlen) = row["item"]
    result_list_exist = verify_trace_from_cached_file(
        model, input_seqlen, output_seqlen
    )
    return {"item": result_list_exist, "params": row["item"]}


def dump_trace_to_pickle_files(grouped_data: dict[str, np.ndarray]):
    """
    - grouped_data: {
        "model_seqlen_config_key": This field should be the same for all entries in the same group.
        "item": The actual trace data. See generate_wrapper().
    }
    """

    global OUTPUT_DIR
    assert OUTPUT_DIR

    model_seqlen_config_key = grouped_data["model_seqlen_config_key"][0]
    cached_file = os.path.join(
        OUTPUT_DIR, ".cache", f"{model_seqlen_config_key}.pkl.gz"
    )
    os.makedirs(os.path.dirname(cached_file), exist_ok=True)

    # merge all data into a list[dict[str, Any]]
    traces = [{"item": item} for item in grouped_data["item"]]

    with gzip.open(cached_file, "wb") as f:
        pickle.dump(traces, f)
        logging.info("Dumped traces to %s", f.name)

    return {"output_file": np.array([cached_file])}


def generate_all_sim_traces(seqlen_pairs: Iterable[tuple[int, int]]):
    global OUTPUT_DIR
    global MODELS
    assert OUTPUT_DIR
    assert MODELS

    ray = _get_ray()
    all_params = []
    for model in MODELS:
        for input_seqlen, output_seqlen in seqlen_pairs:
            all_params.append((model, (input_seqlen, output_seqlen)))
    logging.info("models: %s\nexpected # of pickle files: %d", MODELS, len(all_params))

    # First, read from cached pickle files to check which configs already exist and do not need to be run again.
    param_ds = ray.data.from_items(all_params)
    to_generate_ds = param_ds.map(verify_trace_from_cached_file_wrapper).filter(
        lambda x: x["item"] is False
    )
    to_generate_list = to_generate_ds.take_all()
    to_generate_seqlens = [x["params"] for x in to_generate_list]
    logging.info("# of seqlens to generate: %d", len(to_generate_seqlens))

    # Next, generate those that do not yet exist.
    to_generate_params = []
    for model, (input_seqlen, output_seqlen) in to_generate_seqlens:
        to_generate_params += generate_all_run_configs(
            model, [(input_seqlen, output_seqlen)]
        )
    logging.info("# of traces to generate: %d", len(to_generate_params))
    to_generate_params_ds = ray.data.from_items(to_generate_params)
    result_ds = to_generate_params_ds.map(generate_wrapper).filter(
        lambda x: x["item"] is not None
    )

    # from IPython import embed; embed()

    # Next, group the dataset by (model, (input_seqlen, output_seqlen))
    grouped_ds = result_ds.materialize().groupby("model_seqlen_config_key")
    grouped_ds.map_groups(dump_trace_to_pickle_files).materialize()


def main(argv: Sequence[str]):
    del argv  # Unused.

    init_cmd_args()

    global MODELS
    assert MODELS
    models = []
    seqlen_pairs = read_input_output_seqlen_from_trace_file(False)

    # generate (model, inputseqlen, outputseqlen) tuples
    for model in MODELS:
        for input_seqlen, output_seqlen in seqlen_pairs:
            models.append((model, input_seqlen, output_seqlen))

    # from IPython import embed; embed()

    # model_ds = ray.data.from_items(models)
    # result_ds = model_ds.map(find_cost_optimal_config_for_model)
    # result_ds.materialize()
    # currently, we need to apply a workaround
    # since ray dataset map API being called from a remote function
    # causes failures.
    if GENERATE_TRACE:
        generate_all_sim_traces(seqlen_pairs)
        # run the function sequentially since the function will invoke ray dataset map
        # for i, model in enumerate(models):
        #     find_cost_optimal_config_for_model({"item": model})
        #     logging.info(f"Finished {i+1}/{len(models)}: {model}")
        # return
    if not GENERATE_TRACE:
        # run the function in parallel since it will not invoke ray dataset map
        ray = _get_ray()
        model_ds = ray.data.from_items(models)
        result_ds = model_ds.map(find_cost_optimal_config_for_model)
        result_ds.materialize()
        return
    # PARALLELISM_DEGREE = 8
    # # split models array into PARALLELISM_DEGREE partitions
    # model_partitions = [models[i::PARALLELISM_DEGREE] for i in range(PARALLELISM_DEGREE)]
    # counter = 0
    # for model_list in model_partitions:
    #     futures = []
    #     for model in model_list:
    #         futures.append(find_cost_optimal_config_for_model.remote({"item": model}))
    #         counter += 1
    #     ray.get(futures)
    #     logging.info(f"Finished {counter}/{len(models)} models")


def _parse_stage_a_flags(argv: list[str]) -> list[str]:
    """Parse private Stage-A flags, then let absl consume its global flags."""
    if any(arg in {"--help", "--helpshort", "--helpfull"} for arg in argv[1:]):
        print(_STAGE_A_FLAGS.get_help())
        raise SystemExit(0)
    remaining = _STAGE_A_FLAGS(argv, known_only=True)
    return flags.FLAGS(remaining)


if __name__ == "__main__":
    app.run(main, flags_parser=_parse_stage_a_flags)
