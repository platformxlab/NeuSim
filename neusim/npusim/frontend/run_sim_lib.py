### Library for launching simulations via XXX_ops_generator.py


import csv
from typing import Any

from absl import logging

from neusim.npusim.frontend.llm_ops_generator import DeepSeekOpsGenerator, LLMOpsGeneratorInference, LLMOpsGeneratorTraining
from neusim.npusim.frontend.Operator import Operator
import neusim.npusim.frontend.util as util


def get_pstr_llm(
    dp: int, tp: int, pp: int,
    dp_dcn: int, tp_dcn: int, pp_dcn: int,
    global_batch_size: int,
) -> str:
    return f"dp{dp}-tp{tp}-pp{pp}-dpdcn{dp_dcn}-tpdcn{tp_dcn}-ppdcn{pp_dcn}-bs{global_batch_size}"


def get_pstr_moe(
    dp: int, tp: int, pp: int, ep: int,
    dp_dcn: int, tp_dcn: int, pp_dcn: int, ep_dcn: int,
    global_batch_size: int,
) -> str:
    return f"dp{dp}-tp{tp}-pp{pp}-ep{ep}-dpdcn{dp_dcn}-tpdcn{tp_dcn}-ppdcn{pp_dcn}-epdcn{ep_dcn}-bs{global_batch_size}"


def generate_parallelism_configs_llm(
    num_chips: int, max_num_dcn_pods: int = 16,
    max_dp: int = -1, max_tp: int = -1, max_pp: int = -1,
    *args, **kwargs,
) -> list[dict[str, int]]:
    '''
    Generate all possible parallelism configs for a given number of chips for an LLM.
    Returns a list of possible combinations.
    Each combination is a dict with keys "num_chips", "data_parallelism_degree",
    "tensor_parallelism_degree", "pipeline_parallelism_degree", "data_parallel_degree_dcn".
    TODO: For now, only support data parallelism on DCN.
    '''
    num_chips_factors = util.get_factors(num_chips)
    # use at most max_num_dcn_pods pods
    # otherwise there are too many combinations
    num_chips_factors_dcn = [
        x for x in num_chips_factors if x <= max_num_dcn_pods
    ]
    max_dp = max_dp if max_dp != -1 else num_chips
    max_tp = max_tp if max_tp != -1 else num_chips
    max_pp = max_pp if max_pp != -1 else num_chips
    parallelism_configs: list[dict[str, int]] = []
    for data_parallelism_degree in [x for x in num_chips_factors if x <= max_dp]:
        for tensor_parallelism_degree in [x for x in num_chips_factors if x <= max_tp]:
            for pipeline_parallelism_degree in [x for x in num_chips_factors if x <= max_pp]:
                for dp_dcn in num_chips_factors_dcn:
                    for pp_dcn in num_chips_factors_dcn:
                        if dp_dcn * pp_dcn > max_num_dcn_pods:
                            # use at most max_num_dcn_pods pods
                            continue
                        if data_parallelism_degree * tensor_parallelism_degree * pipeline_parallelism_degree * dp_dcn * pp_dcn == num_chips:
                            parallelism_configs.append({
                                "num_chips": num_chips,
                                "data_parallelism_degree": data_parallelism_degree,
                                "tensor_parallelism_degree": tensor_parallelism_degree,
                                "pipeline_parallelism_degree": pipeline_parallelism_degree,
                                "data_parallel_degree_dcn": dp_dcn,
                                "pipeline_parallel_degree_dcn": pp_dcn,
                            })
    return parallelism_configs


def generate_parallelism_configs_llm_moe(
    num_chips: int, max_num_dcn_pods: int = 16,
    max_dp: int = -1, max_tp: int = -1, max_pp: int = -1, max_ep: int = -1, max_etp: int = -1,
    *args, **kwargs,
) -> list[dict[str, int]]:
    '''
    Generate all possible parallelism configs for a given number of chips for an MoE model.
    Compared to the dense model, it adds the expert parallelism degree.
    '''
    num_chips_factors = util.get_factors(num_chips)
    # use at most max_num_dcn_pods pods
    # otherwise there are too many combinations
    num_chips_factors_dcn = [
        x for x in num_chips_factors if x <= max_num_dcn_pods
    ]
    max_dp = max_dp if max_dp != -1 else num_chips
    max_tp = max_tp if max_tp != -1 else num_chips
    max_pp = max_pp if max_pp != -1 else num_chips
    max_ep = max_ep if max_ep != -1 else num_chips
    max_etp = max_etp if max_etp != -1 else num_chips
    parallelism_configs: list[dict[str, int]] = []
    for data_parallelism_degree in [x for x in num_chips_factors if x <= max_dp]:
        for tensor_parallelism_degree in [x for x in num_chips_factors if x <= max_tp]:
            for pipeline_parallelism_degree in [x for x in num_chips_factors if x <= max_pp]:
                for expert_parallelism_degree in [x for x in num_chips_factors if x <= max_ep]:
                    for dp_dcn in num_chips_factors_dcn:
                        for pp_dcn in num_chips_factors_dcn:
                            for ep_dcn in [1]:  # TODO: use ep_dcn = 1 for now to avoid DCN overhead
                                if dp_dcn * pp_dcn * ep_dcn > max_num_dcn_pods:
                                    # use at most max_num_dcn_pods pods
                                    continue
                                if expert_parallelism_degree > data_parallelism_degree * tensor_parallelism_degree:
                                    # ep should be less than or equal to dp*tp as we want to enforce ep*etp == dp*tp
                                    continue
                                etp = data_parallelism_degree * tensor_parallelism_degree // expert_parallelism_degree
                                if etp > max_etp:
                                    continue
                                if (
                                    data_parallelism_degree * tensor_parallelism_degree
                                    * pipeline_parallelism_degree * expert_parallelism_degree
                                    * dp_dcn * pp_dcn * ep_dcn
                                ) == num_chips:
                                    parallelism_configs.append({
                                        "num_chips": num_chips,
                                        "data_parallelism_degree": data_parallelism_degree,
                                        "tensor_parallelism_degree": tensor_parallelism_degree,
                                        "pipeline_parallelism_degree": pipeline_parallelism_degree,
                                        "expert_parallelism_degree": expert_parallelism_degree,
                                        "data_parallel_degree_dcn": dp_dcn,
                                        "pipeline_parallel_degree_dcn": pp_dcn,
                                        "expert_parallel_degree_dcn": ep_dcn,
                                    })
    return parallelism_configs


def generate_parallelism_configs_dlrm(num_chips: int, *args, **kwargs,) -> list[dict[str, int]]:
    '''
    Generate all possible parallelism configs for a given number of chips for DLRM.
    Returns a list of possible combinations.
    Each combination is a dict with keys "num_chips", "data_parallelism_degree",
    "model_parallelism_degree", "pipeline_parallelism_degree", "data_parallel_degree_dcn".
    Data parallelism is used for MLP layers. Model parallelism is used for embedding tables.
    TODO: For now, only sweep the MP degree from 1 to num_chips. DP uses all chips, and other
        parallelisms are set to 1.
    TODO: Model parallelism specified here is the total MP degree. The actual row/column/table
        sharding is decided in the DLRMOpsGenerator. Maybe expose these in the config as well.
    '''
    # num_chips_factors = get_factors(num_chips)
    # parallelism_configs: list[dict[str, int]] = []
    # for model_parallelism_degree in num_chips_factors:
    #     parallelism_configs.append({
    #         "num_chips": model_parallelism_degree,
    #         "data_parallelism_degree": model_parallelism_degree,
    #         "tensor_parallelism_degree": model_parallelism_degree,
    #         "pipeline_parallelism_degree": 1,
    #         "data_parallel_degree_dcn": 1,
    #         "pipeline_parallel_degree_dcn": 1,
    #     })
    model_parallelism_degree = num_chips
    parallelism_configs = [
        {
            "num_chips": model_parallelism_degree,
            "data_parallelism_degree": model_parallelism_degree,
            "tensor_parallelism_degree": model_parallelism_degree,
            "pipeline_parallelism_degree": 1,
            "data_parallel_degree_dcn": 1,
            "pipeline_parallel_degree_dcn": 1,
        }
    ]
    return parallelism_configs


def generate_parallelism_configs_stable_diffusion(num_chips: int, model="dit-xl", *args, **kwargs) -> list[dict[str, int]]:
    '''
    Generate all possible parallelism configs for a given number of chips for Stable Diffusion models.
    Returns a list of possible combinations.
    Each combination is a dict with keys "num_chips", "data_parallelism_degree",
    "tensor_parallelism_degree", "pipeline_parallelism_degree", "data_parallel_degree_dcn".
    TODO: Do not consider DCN parallelism for now.
    '''
    num_chips_factors = util.get_factors(num_chips)
    parallelism_configs: list[dict[str, int]] = []
    for data_parallelism_degree in num_chips_factors:
        for tensor_parallelism_degree in num_chips_factors:
            for pipeline_parallelism_degree in num_chips_factors:
                if "gligen" in model.lower():
                    # GLIGEN does not support pipeline parallelism
                    if pipeline_parallelism_degree > 1:
                        continue
                if data_parallelism_degree * tensor_parallelism_degree * pipeline_parallelism_degree == num_chips:
                    parallelism_configs.append({
                        "num_chips": num_chips,
                        "data_parallelism_degree": data_parallelism_degree,
                        "tensor_parallelism_degree": tensor_parallelism_degree,
                        "pipeline_parallelism_degree": pipeline_parallelism_degree,
                        "data_parallel_degree_dcn": 1,
                        "pipeline_parallel_degree_dcn": 1,
                    })
    # data_parallelism_degree = num_chips
    # parallelism_configs = [
    #     {
    #          "num_chips": data_parallelism_degree,
    #          "data_parallelism_degree": data_parallelism_degree,
    #          "tensor_parallelism_degree": 1,
    #          "pipeline_parallelism_degree": 1,
    #          "data_parallel_degree_dcn": 1,
    #          "pipeline_parallel_degree_dcn": 1,
    #     }
    # ]
    return parallelism_configs


def generate_parallelism_configs(num_chips: int, model: str, *args, **kwargs) -> list[dict[str, int]]:
    '''
    Generate all possible parallelism configs for a given @num_chips for the specified @model.
    Returns a list of possible combinations.
    '''

    if "llama" in model.lower():
        return generate_parallelism_configs_llm(num_chips, *args, **kwargs)
    elif "deepseek" in model.lower():
        return generate_parallelism_configs_llm_moe(num_chips, *args, **kwargs)
    elif "dlrm" in model.lower():
        return generate_parallelism_configs_dlrm(num_chips, *args, **kwargs)
    elif "gligen" in model.lower() or "dit" in model.lower():
        return generate_parallelism_configs_stable_diffusion(num_chips, model, *args, **kwargs)
    else:
        raise ValueError(f"Invalid model: {model}")


def get_statistics_from_trace_file(trace_file_or_ops_or_ops_dict: str | list[Operator] | list[dict[str, Any]]) -> dict[str, Any]:
    '''
    Get NPU chip-level statistics from a CSV trace file.
    The PP-related stats only considers a single pod with ICI (excluding DCN parallelisms).
    The function only computes the total PP ops time due to PP over DCN.
    Other DCN-related stats should be considered out of this function based on DCN parallel degrees.
    '''
    if isinstance(trace_file_or_ops_or_ops_dict, str):
        with open(trace_file_or_ops_or_ops_dict, "r") as f:
            ops: list[dict[str, Any]] = list(csv.DictReader(f))
    elif isinstance(trace_file_or_ops_or_ops_dict, list):
        if isinstance(trace_file_or_ops_or_ops_dict[0], Operator):
            ops: list[dict[str, Any]] = [op.to_csv_dict() for op in trace_file_or_ops_or_ops_dict]  # type: ignore
        elif isinstance(trace_file_or_ops_or_ops_dict[0], dict):
            ops: list[dict[str, Any]] = trace_file_or_ops_or_ops_dict  # type: ignore
        else:
            raise ValueError(f"Invalid element type in input list. Got {type(trace_file_or_ops_or_ops_dict[0])}")
    else:
        raise ValueError(f"Invalid input type. Got {type(trace_file_or_ops_or_ops_dict)}")

    stats = {}
    # communication ops due to PP
    ops_pp = [
        row for row in ops if "Pipeline" in row["Description"]
    ]
    # other non-PP ops
    ops_non_pp = [
        row for row in ops if "Pipeline" not in row["Description"]
    ]

    ops_pp_ici = [
        row for row in ops_pp if "ICI" in row["Description"]
    ]
    ops_pp_dcn = [
        row for row in ops_pp if "DCN" in row["Description"]
    ]

    ### Below are PP stats on a single chip
    # Total PP time
    stats["total_pp_time_ns"] = sum([
        int(row["Execution time"]) * int(row["Count"])
        for row in ops_pp
    ])
    # Total ICI PP time
    stats["total_pp_ici_time_ns"] = sum([
        int(row["Execution time"]) * int(row["Count"])
        for row in ops_pp_ici
    ])
    # Total DCN PP time
    stats["total_pp_dcn_time_ns"] = sum([
        int(row["Execution time"]) * int(row["Count"])
        for row in ops_pp_dcn
    ])

    ### Below are stats for non-PP ops on a single chip
    # total execution time
    stats["total_execution_time_non_pp_ns"] = sum([
        int(row["Execution time"]) * int(row["Count"])
        for row in ops_non_pp
    ])
    # overlapped time between compute and others
    stats["overlapped_compute_time_non_pp_ns"] = sum([
        min(
            int(row["Compute time"]),
            max(
                int(row["Memory time"]),
                int(row["ICI/NVLink time"]),
            )
        ) * int(row["Count"])
        for row in ops_non_pp
    ])
    # compute-only time
    stats["compute_only_time_non_pp_ns"] = sum([
        (int(row["Compute time"]) - max(int(row["Memory time"]), int(row["ICI/NVLink time"])))
        * int(row["Count"])
        for row in ops_non_pp
        if row["Bounded-by"] == "Compute"
    ])
    # memory-only time
    stats["memory_only_time_non_pp_ns"] = sum([
        (int(row["Memory time"]) - max(int(row["Compute time"]), int(row["ICI/NVLink time"])))
        * int(row["Count"])
        for row in ops_non_pp
        if row["Bounded-by"] == "Memory"
    ])
    # ICI/NVLink-only time
    stats["ici_bound_time_non_pp_ns"] = sum([
        abs(int(row["ICI/NVLink time"]) - int(row["Compute time"]))
        * int(row["Count"])
        for row in ops_non_pp
        if row["Bounded-by"] == "ICI/NVLink"
    ])

    ## Compute overall stats based on non-PP and PP stats
    # total execution time, determined based on whether PP_ICI is the bottleneck
    stats["total_execution_time_chip_ns"] = max(stats["total_execution_time_non_pp_ns"], stats["total_pp_ici_time_ns"])
    # overlapped compute time should only be treated and used as a single-chip stat independent from PP
    stats["overlapped_compute_time_chip_ns"] = stats["overlapped_compute_time_non_pp_ns"]
    stats["compute_only_time_chip_ns"] = stats["compute_only_time_non_pp_ns"]  # compute only time is not affected by PP
    stats["memory_only_time_chip_ns"] = stats["memory_only_time_non_pp_ns"]  # memory only time is not affected by PP
    # If PP is the bottleneck, ici bound time is the difference between total PP time and non-PP non-ICI-bound time
    # Otherwise, ici bound time is the same as non-PP ici bound time
    if stats["total_pp_time_ns"] > stats["total_execution_time_non_pp_ns"]:
        stats["ici_bound_time_chip_ns"] = stats["total_pp_time_ns"] - (
            stats["total_execution_time_non_pp_ns"] - stats["ici_bound_time_non_pp_ns"]
        )
        stats["bounded_by_pp_chip"] = True
    else:
        stats["ici_bound_time_chip_ns"] = stats["ici_bound_time_non_pp_ns"]
        stats["bounded_by_pp_chip"] = False
    return stats


def get_statistics_from_ops(ops: list[Operator]) -> dict[str, Any]:
    return get_statistics_from_trace_file(ops)


def map_parallelism_to_ici_axes_llm(model: str, v: str, parallelism_config: dict[str, Any]) -> tuple[int, ...]:
    '''
    Map parallelism degrees to physical ICI axes for LLMs.
    @return: # of (dp, tp, pp) axes.
    '''
    dp = parallelism_config["data_parallelism_degree"]
    tp = parallelism_config["tensor_parallelism_degree"]
    pp = parallelism_config["pipeline_parallelism_degree"]

    ## Map parallelism to physical ICI axes
    ## PP can take at most 1 axis (due to its P2P communication nature)
    ## Uses heuristic: prioritize TP > DP > PP for training, and TP > DP > PP for inference
    num_physical_axes = 2 if v in ["2", "3", "5e", "6e"] else 3
    tp_enabled, dp_enabled, pp_enabled = tp > 1, dp > 1, pp > 1
    tp_axes, dp_axes, pp_axes = int(tp_enabled), int(dp_enabled), int(pp_enabled)

    if tp_enabled + dp_enabled + pp_enabled == 1:
        # if only one parallelism is enabled, it uses all axes
        tp_axes = num_physical_axes if tp_enabled else 0
        dp_axes = num_physical_axes if dp_enabled else 0
        pp_axes = 1 if pp_enabled else 0  ## PP only uses 1 axis
    elif tp_enabled + dp_enabled + pp_enabled == 2:
        # if two parallelisms are enabled, follow the heuristic to assign axes
        if tp_enabled and (dp_enabled ^ pp_enabled):
            tp_axes = num_physical_axes - 1
            dp_axes = 1 if dp_enabled else 0
            pp_axes = 1 if pp_enabled else 0
        elif not tp_enabled and (dp_enabled and pp_enabled):
            dp_axes = num_physical_axes - 1
            pp_axes = 1
        else:
            raise ValueError("Invalid parallelism configuration")
    elif tp_enabled + dp_enabled + pp_enabled == 3:
        # if all parallelisms are enabled, each uses one axis
        tp_axes, dp_axes, pp_axes = 1, 1, 1

    return dp_axes, tp_axes, pp_axes


def map_parallelism_to_ici_axes_llm_moe(model: str, v: str, parallelism_config: dict[str, Any]) -> tuple[int, ...]:
    '''
    Map parallelism degrees to physical ICI axes for LLMs.
    @return: # of (dp, tp, pp, ep) axes.
    '''
    dp = parallelism_config["data_parallelism_degree"]
    tp = parallelism_config["tensor_parallelism_degree"]
    pp = parallelism_config["pipeline_parallelism_degree"]
    ep = parallelism_config["expert_parallelism_degree"]

    # first map dp/tp/pp axes
    dp_axes, tp_axes, pp_axes = map_parallelism_to_ici_axes_llm(model, v, parallelism_config)

    # then we map ep and etp:
    # ep+etp share the same axes as dp+tp;
    # our heuristic is that we prioritize more axes given to etp over ep
    # since tensor parallelism is more bandwidth hungry than expert parallelism.
    e_axes = dp_axes + tp_axes
    if e_axes == 0:
        # if no dp/tp axes, then no expert axes
        assert ep == 1, \
            f"If no dp/tp axes, ep should be 1. Got ep={ep}, dp={dp}, tp={tp}, pp={pp} with dp_axes={dp_axes}, tp_axes={tp_axes}, pp_axes={pp_axes} (v={v})"
        ep_axes = 0
    elif e_axes == 1:
        if ep > 1:
            # if only one dp/tp axis, and ep > 1, then we cannot have etp
            assert ep == dp * tp, \
                f"If only one axis and ep > 1, then ep should be dp*tp (i.e., no etp). Got ep={ep}, dp={dp}, tp={tp}, pp={pp} with dp_axes={dp_axes}, tp_axes={tp_axes}, pp_axes={pp_axes} (v={v})"
            ep_axes = 1
        else:  # ep == 1
            # if ep == 1, then no ep axes
            ep_axes = 0
    elif e_axes > 1:
        if ep < dp * tp:
            # if we have etp, then we given at most 1 axis to ep
            ep_axes = 1 if ep > 1 else 0
        else:  # ep == dp * tp
            assert ep == dp * tp, f"Must have ep == dp * tp. Got ep={ep}, dp={dp}, tp={tp}, pp={pp} with dp_axes={dp_axes}, tp_axes={tp_axes}, pp_axes={pp_axes} (v={v})"
            # if there is no etp, then we given all axes to ep
            ep_axes = e_axes
    else:
        raise ValueError(f"Invalid value for e_axes. Got {e_axes} for {model} v{v} with config {parallelism_config}")

    return dp_axes, tp_axes, pp_axes, ep_axes


def map_parallelism_to_ici_axes(model: str, v: str, parallelism_config: dict[str, Any]) -> tuple[int, ...]:
    '''
    Map parallelism degrees to physical ICI axes based on the model type.
    @return:
        LLM: # of (dp, tp, pp) axes.
        DeepSeek: # of (dp, tp, pp, ep) axes.
        DLRM: # of (dp, mp, pp) axes.
        SD: # of (dp, tp, pp) axes.
    TODO: For now, do no support PP for DLRM, TP/PP for SD.
    '''
    max_num_axes = 2 if v in ["2", "3"] else 3
    if "llama" in model.lower():
        return map_parallelism_to_ici_axes_llm(model, v, parallelism_config)
    elif "deepseek" in model.lower():
        return map_parallelism_to_ici_axes_llm_moe(model, v, parallelism_config)
    elif "dlrm" in model.lower():
        return max_num_axes, max_num_axes, 0
    elif "gligen" in model.lower() or "dit" in model.lower():
        return max_num_axes, 0, 0
    else:
        raise ValueError(f"Invalid model: {model}")


def validate_parallelism_config_llm(model: str, v: str, config: dict[str, Any], workload: str = "inference", allow_oom: bool | float | int = True, prefill_or_decode: str = "prefill") -> bool:
    '''
    @return False if we should skip the trace generation run for this config. True otherwise.
    '''
    dp_dcn = config["data_parallel_degree_dcn"]
    pp_dcn = config["pipeline_parallel_degree_dcn"]
    dp = config["data_parallelism_degree"]
    tp = config["tensor_parallelism_degree"]
    pp = config["pipeline_parallelism_degree"]
    pstr = get_pstr_llm(
        dp,
        tp,
        pp,
        dp_dcn,
        config["tensor_parallel_degree_dcn"],
        pp_dcn,
        config["global_batch_size"],
    )

    # skip dp_dcn for inference since all pods are the same
    if workload == "inference" and dp_dcn > 1:
        logging.info(f"Skipping {model} v{v} {pstr} since dp_dcn > 1 for inference")
        return False

    # skip this run if total PP > number of layers
    if pp * pp_dcn > config['num_layers']:
        logging.info(f"Skipping {model} v{v} {pstr} since total PP={pp * pp_dcn} > number of layers = {config['num_layers']}")
        return False

    # only two parallelisms can be used on ICI for TPUv2 and TPUv3 due to 2D torus
    if v in ["2", "3"] and (dp > 1 and tp > 1 and pp > 1):
        logging.info(f"Skipping {model} v{v} {pstr} since TPUv2 and TPUv3 only support 2D torus")
        return False

    # num_chips = parallelism_config["num_chips"]
    ici_pod_size = dp * tp * pp
    # TPUv2 Pod has a max of 256 chips
    if v in ["2", "5e", "6e"] and ici_pod_size > 256:
        logging.info(f"Skipping {model} v{v} {pstr} since TPUv2/v5e/v6e Pod has a max of 256 chips")
        return False
    # TPUv3 Pod has a max of 512 chips
    if v == "3" and ici_pod_size > 1024:
        logging.info(f"Skipping {model} v{v} {pstr} since TPUv3 Pod has a max of 512 chips")
        return False
    # TPUv4 Pod has a max of 4096 chips
    if v == "4" and ici_pod_size > 4096:
        logging.info(f"Skipping {model} v{v} {pstr} since TPUv4 Pod has a max of 4096 chips")
        return False
    # TPUv5 Pod has a max of 6144 chips
    if v == "5p" and ici_pod_size > 6144:
        logging.info(f"Skipping {model} v{v} {pstr} since TPUv5p Pod has a max of 6144 chips")
        return False

    # TPUv6 Pod has a max of 8192 chips
    if v == "6p" and ici_pod_size > 8192:
        logging.info(f"Skipping {model} v{v} {pstr} since TPUv6p Pod has a max of 8192 chips")
        return False

    ## By default, uses global batch size 128 with 4K sequence length
    ## The global batch size is first split across dp_dcn and then dp on ICI.
    ## Skip this run if dp * dp_dcn * pp * pp_dcn > global batch size
    if dp * dp_dcn * pp * pp_dcn > config["global_batch_size"]:
        logging.info(f"Skipping {model} v{v} {pstr} since dp * dp_dcn * pp * pp_dcn > global batch size")
        return False

    # check OOM
    if isinstance(allow_oom, (int, float)):
        oom_factor = allow_oom
    else:
        assert isinstance(allow_oom, bool)
        oom_factor = 1 if not allow_oom else 999999.0 # a large number
    # if not allow_oom:
    if "deepseek" in model:
        assert workload == "inference", "DeepSeek models are only supported for inference for now"
        ops_gen = DeepSeekOpsGenerator(config)
        if prefill_or_decode == "prefill":
            mem_footprint_GB = ops_gen.compute_memory_footprint_bytes("prefill") / 1024 / 1024 / 1024
        else:
            mem_footprint_GB = ops_gen.compute_memory_footprint_bytes("decode") / 1024 / 1024 / 1024
    else:
        if workload == "inference":
            ops_gen = LLMOpsGeneratorInference(config)
            if prefill_or_decode == "prefill":
                mem_footprint_GB = ops_gen.compute_memory_footprint_bytes("prefill") / 1024 / 1024 / 1024
            else:
                mem_footprint_GB = ops_gen.compute_memory_footprint_bytes("decode") / 1024 / 1024 / 1024
        else:
            ops_gen = LLMOpsGeneratorTraining(config)
            mem_footprint_GB = ops_gen.compute_memory_footprint_bytes() / 1024 / 1024 / 1024
    is_oom = mem_footprint_GB > config["hbm_size_GB"] * oom_factor
    if is_oom:
        logging.info(f"Skipping {model} v{v} {pstr} since it is OOM. Mem footprint = {mem_footprint_GB:.2f} GB, HBM size = {config['hbm_size_GB']} GB, oom_factor = {oom_factor}")
        return False

    return True


def validate_parallelism_config_llm_moe(model: str, v: str, config: dict[str, Any], workload: str = "inference", allow_oom: bool | float | int = True, prefill_or_decode: str = "prefill") -> bool:
    '''
    @return False if we should skip the trace generation run for this config. True otherwise.
    '''
    dp_dcn = config["data_parallel_degree_dcn"]
    pp_dcn = config["pipeline_parallel_degree_dcn"]
    ep_dcn = config["expert_parallel_degree_dcn"]
    dp = config["data_parallelism_degree"]
    tp = config["tensor_parallelism_degree"]
    pp = config["pipeline_parallelism_degree"]
    ep = config["expert_parallelism_degree"]
    pstr = get_pstr_moe(
        dp,
        tp,
        pp,
        ep,
        dp_dcn,
        config["tensor_parallel_degree_dcn"],
        pp_dcn,
        ep_dcn,
        config["global_batch_size"],
    )

    assert workload == "inference", "MoE models are not supported for training yet."

    ### first validate dp/tp/pp and their dcn axes
    if not validate_parallelism_config_llm(model, v, config, workload, allow_oom, prefill_or_decode):
        logging.info(f"Skipping {model} v{v} {pstr} since dp/tp/pp validation failed")
        return False

    ### then validate ep/etp axes

    # enforce ep_dcn == 1 for now to avoid DCN overhead
    # TODO: support ep_dcn > 1 for MoE models
    if ep_dcn > 1:
        logging.info(f"Skipping {model} v{v} {pstr} since ep_dcn > 1 for inference")
        return False

    # enforce we can have ep*etp == dp*tp
    if ep > dp * tp:
        logging.info(f"Skipping {model} v{v} {pstr} since ep > dp * tp")
        return False

    # if only one of dp or tp is enabled and it only uses one axis, then we can only have one of ep or etp,
    # i.e., ep must be either dp or tp.
    dp_axes, tp_axes, pp_axes = map_parallelism_to_ici_axes_llm(model, v, config)
    if (dp_axes + tp_axes) == 1:
        if ep != dp and ep != tp:  # one of dp or tp is 1.
            logging.info(f"Skipping {model} v{v} {pstr} since ep must be either dp or tp when we only have one axis for dp+tp")
            return False

    # enforce # of experts is more than ep
    if config["num_routed_experts"] < ep:
        logging.info(f"Skipping {model} v{v} {pstr} since num_routed_experts < ep")
        return False

    return True


def validate_parallelism_config(
    model: str,
    v: str,
    config: dict[str, Any],
    workload: str = "inference",
    allow_oom: bool | float | int = True,
    prefill_or_decode: str = "prefill",
) -> bool:
    '''
    @return False if we should skip the trace generation. True otherwise.
    @allow_oom: If True, allow OOM configs to be valid.
    TODO: Currently this is only needed for LLMs. Always returns True for other models.
    '''

    if "llama" in model.lower():
        return validate_parallelism_config_llm(model, v, config, workload, allow_oom, prefill_or_decode)
    elif "deepseek" in model.lower():
        return validate_parallelism_config_llm_moe(model, v, config, workload, allow_oom, prefill_or_decode)
    elif "dlrm" in model.lower():
        return True
    elif "gligen" in model.lower() or "dit" in model.lower():
        return True
    else:
        raise ValueError(f"Invalid model: {model}")
