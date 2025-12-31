#!/usr/bin/env python3

### Run simulations to sweep different models and NPU configurations.
### Run "python run_sim.py --help" for more information on how to use this script.

from math import ceil
import os
from typing import Any, Sequence
import json

import ray

from absl import app, flags, logging

import neusim.npusim.frontend.run_sim_lib as run_sim_lib
from neusim.npusim.frontend.gligen_ops_generator import GLIGENOpsGenerator
from neusim.npusim.frontend.dit_ops_generator import DiTOpsGenerator
from neusim.npusim.frontend.dlrm_ops_generator import DLRMOpsGenerator
from neusim.npusim.frontend.llm_ops_generator import LLMOpsGeneratorBase, LLMOpsGeneratorTraining
from neusim.npusim.frontend.llm_ops_generator import LLMOpsGeneratorInference
from neusim.npusim.frontend.llm_ops_generator import DeepSeekOpsGenerator


models = flags.DEFINE_list(
    "models",
    ["llama3-8b", "llama2-13b", "llama3-70b", "llama3_1-405b"],
    "List of models to analyze",
)
versions = flags.DEFINE_list(
    "versions",
    ["2", "3", "4", "5p", "6p", "5e", "6e"],
    "List of NPU versions to analyze",
)
num_chips_list = flags.DEFINE_list(
    "num_chips",
    [str(x) for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]],
    "List of number of chips to analyze",
)
inference_batch_sizes = flags.DEFINE_list(
    "inference_batch_sizes",
    [str(x) for x in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]],
    "List of batch sizes to analyze for inference",
)
training_batch_sizes = flags.DEFINE_list(
    "training_batch_sizes",
    [str(x) for x in [32]],
    "List of batch sizes to analyze for training",
)
workload = flags.DEFINE_string(
    "workload", "training", "Workload type: training or inference"
)
seqlen = flags.DEFINE_string(
    "seqlen", "4096_512", "Sequence length override: input_output."
)
SEQLEN_OVERRIDE: tuple[int, int] | None = None
__OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", "../../results/raw", "Output directory for trace files"
)
OUTPUT_DIR: str | None = None
__CONFIGS_PATH = flags.DEFINE_string(
    "configs_path", "../../configs", "Path to the configs directory"
)
CONFIGS_PATH: str | None = None
skip_exist = flags.DEFINE_boolean(
    "skip_exist", False, "Skip existing trace files"
)
__DEBUG = flags.DEFINE_boolean(
    "debug", False, "Debug mode. Disables parallel processing."
)


def dump_stats_llm(v: str, base_config: dict[str, Any], ops_gen: LLMOpsGeneratorBase, workload: str = "inference"):
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
        ### temp fix: skip if the file already exists and is valid
        prefill_csv = output_file.replace(".csv", "_prefill.csv")
        prefill_json = output_file.replace(".csv", "_prefill.json")
        decode_csv = output_file.replace(".csv", "_decode.csv")
        decode_json = output_file.replace(".csv", "_decode.json")
        try:
            if os.path.exists(prefill_csv) and os.path.exists(prefill_json) and \
            os.path.exists(decode_csv) and os.path.exists(decode_json):
                # check all files can be loaded
                _ = json.load(open(prefill_json, "r"))
                _ = json.load(open(decode_json, "r"))
                logging.info(f"Skipping stats dump for {output_file} since the output files exist")
                return
        except Exception as e:
            logging.info(f"Existing stats files are corrupted, re-generating them: {e}")

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
        assert prefill_pp_stage_time_ns > 0, \
            f"csv files: {output_file.replace('.csv', '_prefill.csv')}, {output_file.replace('.csv', '_decode.csv')}"
        prefill_stats["throughput_tokens_per_sec"] = microbatch_size_ici * input_seqlen * 1e9 / prefill_pp_stage_time_ns
        prefill_pod_latency_ns = (prefill_stats["total_execution_time_non_pp_ns"] + prefill_stats["total_pp_ici_time_ns"]) * pp
        prefill_tot_latency_ns = (prefill_pod_latency_ns + prefill_stats["total_pp_dcn_time_ns"]) * pp_dcn
        prefill_stats["TTFT_sec"] = prefill_tot_latency_ns / 1e9

        # update decode stats (throughput:tokens/sec, throughput:tokens/sec/request)
        # tokens/sec/request is the inverse of TPOT (time per output token per request)
        decode_pod_latency_ns = (decode_stats["total_execution_time_non_pp_ns"] + decode_stats["total_pp_ici_time_ns"]) * pp / output_seqlen
        decode_tot_latency_ns = (decode_pod_latency_ns + decode_stats["total_pp_dcn_time_ns"] / output_seqlen) * pp_dcn
        decode_pp_stage_time_ns = ceil(decode_stats["total_execution_time_chip_ns"] / output_seqlen)
        assert decode_stats["total_execution_time_chip_ns"] > 0, \
            f"csv files: {output_file.replace('.csv', '_prefill.csv')}, {output_file.replace('.csv', '_decode.csv')}, stats: {decode_stats}"
        decode_stats["TPOT_ms_request"] = decode_tot_latency_ns / 1e6
        decode_stats["throughput_tokens_per_sec"] = microbatch_size_ici * 1e9 / decode_pp_stage_time_ns
        decode_stats["throughput_tokens_per_sec_request"] = 1e3 / decode_stats["TPOT_ms_request"]

        assert isinstance(ops_gen, (LLMOpsGeneratorInference, DeepSeekOpsGenerator)), \
            f"ops_gen should be LLMOpsGeneratorInference or DeepSeekOpsGenerator, but got {type(ops_gen)}"
        prefill_stats["mem_footprint_GB"] = ops_gen.compute_memory_footprint_bytes("prefill") / 1024 / 1024 / 1024
        prefill_stats["out_of_memory"] = (base_config["hbm_size_GB"] < prefill_stats["mem_footprint_GB"])
        decode_stats["mem_footprint_GB"] = ops_gen.compute_memory_footprint_bytes("decode") / 1024 / 1024 / 1024
        decode_stats["out_of_memory"] = (base_config["hbm_size_GB"] < decode_stats["mem_footprint_GB"])

        # merge sim configs into stats
        prefill_stats["sim_config"] = base_config
        decode_stats["sim_config"] = base_config

        json.dump(prefill_stats, open(output_file.replace(".csv", "_prefill.json"), "w"), indent=4)
        json.dump(decode_stats, open(output_file.replace(".csv", "_decode.json"), "w"), indent=4)
    elif workload == "training":
        stats = run_sim_lib.get_statistics_from_trace_file(output_file)

        ## First, process the stats for each pod
        # all exe times should be multiplied by the number of microbatches + initiation interval (II) of the pipeline
        # TODO: currently using num microbatches equal to num pipeline stages. Should change this in the future.
        # II_ICI_PP = pp - 1
        # pp_time_multiplier = num_microbatches_per_pod + II_ICI_PP
        pp_time_multiplier = pp
        stats["total_execution_time_pod_ns"] = stats["total_execution_time_chip_ns"] * pp_time_multiplier
        stats["compute_only_time_pod_ns"] = stats["compute_only_time_chip_ns"] * pp_time_multiplier
        stats["memory_only_time_pod_ns"] = stats["memory_only_time_chip_ns"] * pp_time_multiplier
        stats["ici_bound_time_pod_ns"] = stats["ici_bound_time_chip_ns"] * pp_time_multiplier

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

        assert isinstance(ops_gen, LLMOpsGeneratorTraining), \
            f"ops_gen should be LLMOpsGeneratorTraining, but got {type(ops_gen)}"
        stats["mem_footprint_GB"] = ops_gen.compute_memory_footprint_bytes() / 1024 / 1024 / 1024
        stats["out_of_memory"] = (base_config["num_chips"] * base_config["hbm_size_GB"] < stats["mem_footprint_GB"])

        # merge sim configs into stats
        stats["sim_config"] = base_config

        json.dump(stats, open(output_file.replace(".csv", ".json"), "w"), indent=4)
    else:
        raise ValueError(f"Invalid workload: {workload}")


def dump_stats_dlrm(v: str, base_config: dict[str, Any], ops_gen: DLRMOpsGenerator, workload: str = "inference"):
    output_file: str = base_config["output_file_path"]
    global_batch_size: int = base_config["global_batch_size"]
    pp_dcn: int = base_config["pipeline_parallel_degree_dcn"]
    dp_dcn: int = base_config["data_parallel_degree_dcn"]
    num_chips: int = base_config["num_chips"]

    num_pods = dp_dcn * pp_dcn
    assert num_pods == 1, f"DLRM only supports one pod for now, but get num_pods={num_pods}"
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

            json.dump(stats, open(chip_output_file.replace(".csv", ".json"), "w"), indent=4)

        # process overall performance (use the straggler chip's perf. as the overall perf.)
        stats = run_sim_lib.get_statistics_from_trace_file(output_file)
        stats["throughput_requests_per_sec"] = (
            global_batch_size / stats["total_execution_time_chip_ns"] * 1e9
        )
        stats["latency_ns"] = stats["total_execution_time_chip_ns"]
        stats["mem_footprint_GB"] = ops_gen.compute_memory_footprint_bytes() / 1024 / 1024 / 1024
        stats["out_of_memory"] = (base_config["num_chips"] * base_config["hbm_size_GB"] < stats["mem_footprint_GB"])

        stats["sim_config"] = base_config
        json.dump(stats, open(output_file.replace(".csv", ".json"), "w"), indent=4)

    elif workload == "training":
        raise NotImplementedError("DLRM training is not supported yet")
    else:
        raise ValueError(f"Invalid workload: {workload}")


def dump_stats_stable_diffusion(v: str, base_config: dict[str, Any], ops_gen: DiTOpsGenerator | GLIGENOpsGenerator, workload: str = "inference"):
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
        stats["throughput_requests_per_sec"] = (
            global_batch_size / (stats["total_execution_time_chip_ns"] / 1e9 * total_num_steps)
        )
        stats["throughput_step_per_sec_per_request"] = (
            num_diffusion_steps / (stats["total_execution_time_chip_ns"] / 1e9)
        )
        stats["latency_sec"] = stats["total_execution_time_chip_ns"] / 1e9 * total_num_steps
        stats["latency_step_sec"] = stats["total_execution_time_chip_ns"] / num_diffusion_steps / 1e9 / total_num_steps
        stats["mem_footprint_GB"] = ops_gen.compute_memory_footprint_bytes() / 1024 / 1024 / 1024
        stats["out_of_memory"] = (base_config["num_chips"] * base_config["hbm_size_GB"] < stats["mem_footprint_GB"])

        stats["sim_config"] = base_config
        json.dump(stats, open(output_file.replace(".csv", ".json"), "w"), indent=4)
    elif workload == "training":
        raise NotImplementedError("Stable Diffusion training is not supported yet")
    else:
        raise ValueError(f"Invalid workload: {workload}")


def dump_stats(model: str, v: str, base_config: dict[str, Any], ops_gen, workload: str = "inference"):
    if "llama" in model.lower() or "deepseek" in model.lower():
        dump_stats_llm(v, base_config, ops_gen, workload)
    elif "dlrm" in model.lower():
        dump_stats_dlrm(v, base_config, ops_gen, workload)
    elif "gligen" in model.lower() or "dit" in model.lower():
        dump_stats_stable_diffusion(v, base_config, ops_gen, workload)
    else:
        raise ValueError(f"Invalid model: {model}")


# @ray.remote
def generate(
    model: str,
    v: str,
    parallelism_config: dict[str, int],
    global_batch_size: int,
    skip_exist: bool,
    output_dir: str,
    workload: str,
) -> None:
    global CONFIGS_PATH
    global SEQLEN_OVERRIDE
    global OUTPUT_DIR
    assert OUTPUT_DIR
    assert CONFIGS_PATH

    options: dict[str, Any] = {
        **parallelism_config
    }
    dp = parallelism_config["data_parallelism_degree"]
    tp = parallelism_config["tensor_parallelism_degree"]
    pp = parallelism_config["pipeline_parallelism_degree"]
    dp_dcn = parallelism_config["data_parallel_degree_dcn"]
    tp_dcn = 1  # TODO: For now, only support tensor parallelism on ICI since TP has high communication traffic
    pp_dcn = parallelism_config["pipeline_parallel_degree_dcn"]

    if "expert_parallelism_degree" in parallelism_config:
        ep = parallelism_config["expert_parallelism_degree"]
        pstr = run_sim_lib.get_pstr_moe(dp, tp, pp, ep, dp_dcn, tp_dcn, pp_dcn, 1, global_batch_size)
    else:
        ep = 1
        pstr = run_sim_lib.get_pstr_llm(dp, tp, pp, dp_dcn, tp_dcn, pp_dcn, global_batch_size)

    # read base model and NPU configs
    base_model_config_path = os.path.join(CONFIGS_PATH, f"models/{model}.json")
    base_npu_config_path = os.path.join(CONFIGS_PATH, f"chips/tpuv{v}.json")
    base_model_config = json.load(open(base_model_config_path, "r"))
    base_npu_config = json.load(open(base_npu_config_path, "r"))
    base_sys_config = json.load(open(os.path.join(CONFIGS_PATH, "systems/system_config.json"), "r"))

    # create config for the ops generator
    base_config = {
        **base_model_config, **base_npu_config, **base_sys_config
    }
    base_config.update(options)

    ## Determine microbatch sizes for ICI and DCN. Determine # of NPU pods and batch size per pod.
    microbatch_size_ici = ceil(global_batch_size / dp_dcn / pp / pp_dcn)
    microbatch_size_dcn = ceil(global_batch_size / pp_dcn)

    base_config["global_batch_size"] = global_batch_size
    base_config["microbatch_size_ici"] = microbatch_size_ici
    base_config["microbatch_size_dcn"] = microbatch_size_dcn

    if SEQLEN_OVERRIDE:
        input_seqlen, output_seqlen = SEQLEN_OVERRIDE
        base_config["input_seqlen"] = input_seqlen
        base_config["output_seqlen"] = output_seqlen
        base_config["output_file_path"] = (
            os.path.join(
                output_dir,
                f"{model}_{input_seqlen}_{output_seqlen}/{pstr}/{workload}-v{v}.csv",
            )
        )
    else:
        ## set output file path
        base_config["output_file_path"] = (
            os.path.join(
                output_dir,
                f"{model}/{pstr}/{workload}-v{v}.csv",
            )
        )
    if skip_exist and os.path.exists(base_config["output_file_path"]):
        logging.info(f"Skipping {model} v{v} {pstr} since the output file exists")
        return

    prefill_valid = run_sim_lib.validate_parallelism_config(
        model, v, base_config, workload, allow_oom=1.5, prefill_or_decode="prefill"
    )
    decode_valid = run_sim_lib.validate_parallelism_config(
        model, v, base_config, workload, allow_oom=1.5, prefill_or_decode="decode"
    )
    if not prefill_valid and not decode_valid:
        # run them if at least one of them is valid
        # both valid flags will be the same if workload == training.
        logging.info(
            f"Skipping {model} v{v} {pstr} since the parallelism config is invalid"
        )
        return

    ## Map parallelism degrees to physical ICI axes
    axes_mappings = run_sim_lib.map_parallelism_to_ici_axes(model, v, parallelism_config)

    if "deepseek" in model.lower():
        assert len(axes_mappings) == 4, f"DeepSeek model {model} v{v} should have 4 axes mappings (dp, tp, pp, ep), but got {len(axes_mappings)}"
        dp_axes, tp_axes, pp_axes, ep_axes = axes_mappings
    else:
        assert len(axes_mappings) == 3, f"Model {model} v{v} should have 3 axes mappings (dp, tp, pp), but got {len(axes_mappings)}"
        dp_axes, tp_axes, pp_axes = axes_mappings
        ep_axes = 0

    # update parallelism axes in the config
    base_config["num_data_parallel_axes"] = dp_axes
    base_config["num_tensor_parallel_axes"] = tp_axes
    base_config["num_pipeline_parallel_axes"] = pp_axes
    if "deepseek" in model.lower():
        base_config["num_expert_parallel_axes"] = ep_axes

    ## create output dir
    os.makedirs(os.path.dirname(base_config["output_file_path"]), exist_ok=True)

    ## run the ops generator
    if "llama" in model.lower():
        if workload == "training":
            # TODO: flashattention is not supported for training yet
            base_config["use_flash_attention"] = False
            ops_generator = LLMOpsGeneratorTraining(base_config)
        elif workload == "inference":
            ops_generator = LLMOpsGeneratorInference(base_config)
        else:
            raise ValueError(f"Invalid workload: {workload}")
    elif "deepseek" in model.lower():
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
    ops_generator.generate(dump_to_file=True, analyze_energy=False)

    # dump config
    os.makedirs(os.path.join(OUTPUT_DIR, "generated_configs", model, pstr), exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "generated_configs", model, pstr, f"{workload}-v{v}.json"), "w") as f:
        json.dump(ops_generator.config.model_dump(mode="json"), f, indent=4)  # type: ignore

    ## dump stats
    dump_stats(model, v, ops_generator.config.model_dump(mode="json"), ops_generator, workload)


def generate_all_run_configs() -> list[tuple[str, str, dict[str, Any], int, bool, str, str]]:
    '''
    Generate all possible parallelism configs for all models, versions, and number of chips.
    '''
    batch_sizes = training_batch_sizes.value if workload.value == "training" else inference_batch_sizes.value
    max_dp = 1 if workload.value == "inference" else -1  # only use DP=1 for each instance during inference
    batch_sizes = [int(x) for x in batch_sizes]
    all_configs = []
    for model in models.value:
        parallelism_configs = []
        for num_chips in num_chips_list.value:
            num_chips = int(num_chips)
            if num_chips > 256:
                for dcn in [1, 2, 4]:  # number of NPU pods. Only try up to 4 for now. Only use multi-pod for more than 256 chips (>= limit of v5e/v6e).
                    parallelism_configs += run_sim_lib.generate_parallelism_configs(
                        num_chips // dcn, model, max_dp=max_dp, max_num_dcn_pods=dcn,
                        max_etp=8,  # DeepSeek experts are small, so do not need to use too much TP for experts
                    )
            else:
                parallelism_configs += run_sim_lib.generate_parallelism_configs(
                    num_chips, model, max_dp=max_dp, max_num_dcn_pods=1,
                    max_etp=8,  # DeepSeek experts are small, so do not need to use too much TP for experts
                )
            # parallelism_configs += run_sim_lib.generate_parallelism_configs(num_chips, model)
        logging.info("# parallelism configs (model=%s): %s", model, len(parallelism_configs))
        for v in versions.value:
            for parallelism_config in parallelism_configs:
                for bs in batch_sizes:
                    all_configs.append((model, v, parallelism_config, bs, skip_exist.value, __OUTPUT_DIR.value, workload.value))
    return all_configs


def generate_wrapper(row: dict[str, Any]):
    # set logging level to INFO
    logging.set_verbosity(logging.INFO)

    return {"item": generate(*row["item"])}


def init_cmd_args():
    '''Initialize command line arguments since ray remote function cannot serialize absl flags variables.'''
    global CONFIGS_PATH
    global SEQLEN_OVERRIDE
    global OUTPUT_DIR

    CONFIGS_PATH = __CONFIGS_PATH.value
    OUTPUT_DIR = __OUTPUT_DIR.value
    if seqlen.present:
        input_seqlen, output_seqlen = seqlen.value.split("_")
        SEQLEN_OVERRIDE = (int(input_seqlen), int(output_seqlen))
        logging.info(f"Using sequence length override: input_seqlen={input_seqlen}, output_seqlen={output_seqlen}")


def main(argv: Sequence[str]):
    del argv  # Unused.

    init_cmd_args()

    params = generate_all_run_configs()
    print("# of traces:", len(params))
    # with Pool(min(len(params), 44)) as p:
    #     p.starmap(generate, params)
    if __DEBUG.value:
        import tqdm
        for param in tqdm.tqdm(params):
            generate(*param)  # type: ignore
    else:
        # progress_starmap(generate, params, n_cpu=44)
        # ray.init(object_store_memory=60 * (2**30))
        # ray.init(address=os.getenv("RAY_ADDRESS", "auto"), namespace="default")
        param_ds = ray.data.from_items(params)
        result_ds = param_ds.map(generate_wrapper)
        result_ds.materialize()
        # futures = [generate.remote(*param) for param in params]
        # ray.get(futures)

    # for model, v, parallelism_config in params:
    #     generate(model, v, parallelism_config)


if __name__=="__main__":
    app.run(main)
