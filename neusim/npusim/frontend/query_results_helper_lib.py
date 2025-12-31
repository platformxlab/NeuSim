### Library for reading and processing the results from the Ops Generator.


import json
import csv
import os
from typing import Callable, Sequence, Any


RESULTS_PATH = "../../../results/raw"


# dollar cost per chip hour
VERSION_TO_COST: dict[str, float] = {
    "2": 1.2375,
    "3": 2,
    "4": 3.22,
    "5p": 4.2,
    "6p": 9.45,
    "5e": 1.2,
    "6e": 2.7,
}
'''Dollar per chip-hour.'''

VERSION_TO_COST_3_YEAR_COMMIT: dict[str, float] = {
    "2": 0.556875,
    "3": 0.9,
    "4": 1.449,
    "5p": 1.89,
    "6p": 4.27,
    "5e": 0.54,
    "6e": 1.22,
}
'''Dollar per chip-hour with 3-year committment.'''


def set_results_path(results_path: str):
    '''
    Sets the path to the results directory.
    '''
    global RESULTS_PATH
    RESULTS_PATH = results_path


def is_model_llm(model: str) -> bool:
    model = model.lower()
    return ("llama" in model or "llm" in model) or is_model_llm_moe(model)


def is_model_llm_moe(model: str) -> bool:
    model = model.lower()
    if "deepseek" in model:
        if "236b" in model or "671b" in model:
            return True
    # TODO: also add qwen and more MoE LLM models here
    return False


def is_model_dlrm(model: str) -> bool:
    model = model.lower()
    return "dlrm" in model


def is_model_sd(model: str) -> bool:
    model = model.lower()
    return "gligen" in model or "dit" in model


def get_pstr_from_pconfig(
    model: str | None = None,
    **kwargs: int | None
) -> str:
    '''
    Returns the parallelism string for the given parallelism config.

    @kwargs: dp, tp, pp, ep, dp_dcn, tp_dcn, pp_dcn, ep_dcn, bs

    @model: If provided, use the model name to determine if it is an LLM MoE model.
    If not provided, use the presence of "ep" and "ep_dcn" in kwargs
    to determine if it is an LLM MoE model.

    If ep and ep_dcn are not provided, the model is assumed
    to be a dense model, and ep/ep_dcn will be ignored from the pstr.
    '''
    dp = kwargs["dp"]
    tp = kwargs["tp"]
    pp = kwargs["pp"]
    dp_dcn = kwargs["dp_dcn"]
    tp_dcn = kwargs["tp_dcn"]
    pp_dcn = kwargs["pp_dcn"]
    bs = kwargs["bs"]
    assert dp and tp and pp and dp_dcn and tp_dcn and pp_dcn and bs, \
        "dp, tp, pp, dp_dcn, tp_dcn, pp_dcn, and bs must be provided."

    model_llm_moe_check = (model and is_model_llm_moe(model)) or ((not model) and ("ep" in kwargs or "ep_dcn" in kwargs))
    if model_llm_moe_check:
        assert "ep" in kwargs and "ep_dcn" in kwargs, \
            "ep and ep_dcn must be provided together for MoE models."
        ep = kwargs["ep"]
        ep_dcn = kwargs["ep_dcn"]
        assert ep and ep_dcn, "ep and ep_dcn must be provided."
        pstr = f"dp{dp}-tp{tp}-pp{pp}-ep{ep}-dpdcn{dp_dcn}-tpdcn{tp_dcn}-ppdcn{pp_dcn}-epdcn{ep_dcn}-bs{bs}"
    else:
        pstr = f"dp{dp}-tp{tp}-pp{pp}-dpdcn{dp_dcn}-tpdcn{tp_dcn}-ppdcn{pp_dcn}-bs{bs}"
    return pstr


def get_pconfig_from_pstr(pstr: str) -> tuple[int, ...]:
    '''
    @pstr: Parallelism string in the format:
        "dp{dp}-tp{tp}-pp{pp}-ep{ep}-dpdcn{dp_dcn}-tpdcn{tp_dcn}-ppdcn{pp_dcn}-epdcn{ep_dcn}-bs{bs}"
    Returns a tuple of the parallelism config:
        (dp, tp, pp, ep, dp_dcn, tp_dcn, pp_dcn, ep_dcn, bs)
    If ep and ep_dcn are not present in the pstr,
    it is assumed to be a dense model, and ep/ep_dcn will be set to 1.
    '''
    pconfigstrs = pstr.split("-")

    map_to_int = lambda pf, x: int(x.removeprefix(pf))

    dp = map_to_int("dp", pconfigstrs[0])
    tp = map_to_int("tp", pconfigstrs[1])
    pp = map_to_int("pp", pconfigstrs[2])
    if len(pconfigstrs) == 7:
        # dense model
        ep = 1
        ep_dcn = 1
        dp_dcn = map_to_int("dpdcn", pconfigstrs[3])
        tp_dcn = map_to_int("tpdcn", pconfigstrs[4])
        pp_dcn = map_to_int("ppdcn", pconfigstrs[5])
        bs = map_to_int("bs", pconfigstrs[6])
    elif len(pconfigstrs) == 9:
        # MoE model
        ep = map_to_int("ep", pconfigstrs[3])
        dp_dcn = map_to_int("dpdcn", pconfigstrs[4])
        tp_dcn = map_to_int("tpdcn", pconfigstrs[5])
        pp_dcn = map_to_int("ppdcn", pconfigstrs[6])
        ep_dcn = map_to_int("epdcn", pconfigstrs[7])
        bs = map_to_int("bs", pconfigstrs[8])
    else:
        raise ValueError(
            f"Invalid pstr format: {pstr}. Expected 7 or 9 components, got {len(pconfigstrs)}."
        )

    return (
        dp, tp, pp, ep, dp_dcn, tp_dcn, pp_dcn, ep_dcn, bs
    )


def get_stats_filepath(
    model: str,
    version: str,
    workload: str,
    dp: int,
    tp: int,
    pp: int,
    dp_dcn: int,
    tp_dcn: int,
    pp_dcn: int,
    batch_size: int,
    ep: int | None = None,
    ep_dcn: int | None = None,
    prefill_or_decode: str = "",
    results_path: str | None = None,
) -> str:
    '''
    Returns the path to the json stats file for the given experiment.
    '''
    if results_path is None:
        results_path = RESULTS_PATH
    pstr = get_pstr_from_pconfig(
        model=model,
        dp=dp,
        tp=tp,
        pp=pp,
        dp_dcn=dp_dcn,
        tp_dcn=tp_dcn,
        pp_dcn=pp_dcn,
        bs=batch_size,
        ep=ep,
        ep_dcn=ep_dcn
    )
    if is_model_llm(model):
        if workload == "inference":
            prefill_or_decode = f"_{prefill_or_decode}"
        elif workload == "training":
            prefill_or_decode = ""
        else:
            raise ValueError(f"Invalid workload: {workload}")
        return f"{results_path}/{model}/{pstr}/{workload}-v{version}{prefill_or_decode}.json"
    else:
        if workload == "inference":
            prefill_or_decode = ""
        elif workload == "training":
            raise ValueError("Training workload is not supported for non-LLM models.")
        else:
            raise ValueError(f"Invalid workload: {workload}")
        return f"{results_path}/{model}/{pstr}/{workload}-v{version}.json"


def get_op_stats_filepath(
    model: str,
    version: str,
    workload: str,
    dp: int,
    tp: int,
    pp: int,
    dp_dcn: int,
    tp_dcn: int,
    pp_dcn: int,
    batch_size: int,
    ep: int | None = None,
    ep_dcn: int | None = None,
    prefill_or_decode: str = "",
    results_path: str | None = None,
) -> str:
    '''
    Returns the path to the csv op stats file for the given experiment.
    '''
    return get_stats_filepath(
        model,
        version,
        workload,
        dp,
        tp,
        pp,
        dp_dcn,
        tp_dcn,
        pp_dcn,
        batch_size,
        ep,
        ep_dcn,
        prefill_or_decode,
        results_path
    ).replace(".json", ".csv")


def get_stats(
    model: str,
    version: str,
    workload: str,
    dp: int,
    tp: int,
    pp: int,
    dp_dcn: int,
    tp_dcn: int,
    pp_dcn: int,
    batch_size: int,
    ep: int | None = None,
    ep_dcn: int | None = None,
    prefill_or_decode: str = "",
    results_path: str | None = None,
) -> dict[str, Any]:
    '''
    Returns the stats dictionary for the given experiment.
    '''
    stats_file = get_stats_filepath(
        model,
        version,
        workload,
        dp,
        tp,
        pp,
        dp_dcn,
        tp_dcn,
        pp_dcn,
        batch_size,
        ep,
        ep_dcn,
        prefill_or_decode,
        results_path
    )
    try:
        stats = json.load(open(stats_file, "r"))
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON from {stats_file}. Please check the file format.")
    return stats


def get_op_stats(
    model: str,
    version: str,
    workload: str,
    dp: int,
    tp: int,
    pp: int,
    dp_dcn: int,
    tp_dcn: int,
    pp_dcn: int,
    batch_size: int,
    ep: int | None = None,
    ep_dcn: int | None = None,
    prefill_or_decode: str = "",
    results_path: str | None = None,
) -> list[dict[str, Any]]:
    '''
    Returns the op stats (list of dictionary) for the given experiment.
    '''
    op_stats_file = get_op_stats_filepath(
        model,
        version,
        workload,
        dp,
        tp,
        pp,
        dp_dcn,
        tp_dcn,
        pp_dcn,
        batch_size,
        ep,
        ep_dcn,
        prefill_or_decode,
        results_path
    )
    with open(op_stats_file, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        op_stats = list(reader)
    return op_stats


def get_num_chips(
    model: str,
    pstr: str | None = None,
    dp: int | None = None,
    tp: int | None = None,
    pp: int | None = None,
    ep: int | None = None,
    dp_dcn: int | None = None,
    tp_dcn: int | None = None,
    pp_dcn: int | None = None,
    ep_dcn: int | None = None,
) -> int:
    '''
    Return # of chips for @model and @pstr or parallelism config.
    One of @pstr and parallelism config (@dp, ..., @pp_dcn) must be provided.
    '''
    assert pstr is not None or all(
        [
            # for all models, dp, tp, pp, dp_dcn, tp_dcn, pp_dcn must be provided
            dp is not None,
            tp is not None,
            pp is not None,
            dp_dcn is not None,
            tp_dcn is not None,
            pp_dcn is not None,

            # if model is MoE, then ep and ep_dcn must also be provided
            not is_model_llm_moe(model) or (ep is not None and ep_dcn is not None),
         ]
    ), "One of pstr and parallelism config must be provided."
    if pstr is not None:
        dp, tp, pp, ep, dp_dcn, tp_dcn, pp_dcn, ep_dcn, bs = get_pconfig_from_pstr(pstr)
    if is_model_dlrm(model):
        assert dp == tp, "DP and TP must be the same for DLRM."
        assert pp == 1, "PP must be 1 for DLRM."
        num_chips = dp * dp_dcn  # type: ignore
    else:
        num_chips = dp * tp * pp * dp_dcn * tp_dcn * pp_dcn * ep * ep_dcn  # type: ignore
    return num_chips


def get_all_stats_helper(
    model: str,
    version: str,
    workload: str,
    prefill_or_decode: str = "",
    max_num_chips: int = -1,
    batch_size: int = -1,
    read_csv: bool = False,
    read_json_with_csv: bool = False,
    results_path: str | None = None,
) -> dict[tuple[int, ...], dict[str, Any] | list[dict[str, Any]] | tuple[dict[str, Any], list[dict[str, Any]]]]:
    '''
    Returns all stats dictionaries (or per-op stats csv) for all experiments of the given model and version.
    Index the returned dict by @return[(dp, tp, pp, ep, dp_dcn, pp_dcn, batch_size)] (ignore ep_dcn for now due to its DCN overhead).
    Set @max_num_chips or @batch_size to -1 to ignore these filters.
    '''
    if results_path is None:
        results_path = RESULTS_PATH
    model_results_dir = f"{results_path}/{model}/"
    config_dirs = os.listdir(model_results_dir)
    all_stats = {}
    for config_dir in config_dirs:
        dp, tp, pp, ep, dp_dcn, tp_dcn, pp_dcn, ep_dcn, bs = get_pconfig_from_pstr(config_dir)

        if batch_size != -1 and batch_size != bs:
            continue

        num_chips = get_num_chips(
            model,
            tp=tp,
            dp=dp,
            pp=pp,
            ep=ep,
            dp_dcn=dp_dcn,
            tp_dcn=tp_dcn,
            pp_dcn=pp_dcn,
            ep_dcn=ep_dcn,
        )
        if max_num_chips > 0 and num_chips > max_num_chips:
            continue

        # skip if the file does not exist
        stats_file = get_stats_filepath(
            model, version, workload, dp, tp, pp, dp_dcn, tp_dcn, pp_dcn, bs, ep, ep_dcn, prefill_or_decode, results_path
        )
        if not os.path.exists(stats_file):
            continue

        if read_csv:
            csv_stats = get_op_stats(
                model, version, workload, dp, tp, pp, dp_dcn, tp_dcn, pp_dcn, bs, ep, ep_dcn, prefill_or_decode
            )
        if read_json_with_csv or not read_csv:
            stats = get_stats(
                model, version, workload, dp, tp, pp, dp_dcn, tp_dcn, pp_dcn, bs, ep, ep_dcn, prefill_or_decode
            )
        if read_csv:
            if read_json_with_csv:
                all_stats[(dp, tp, pp, ep, dp_dcn, pp_dcn, bs)] = (stats, csv_stats)  # type: ignore
            else:
                all_stats[(dp, tp, pp, ep, dp_dcn, pp_dcn, bs)] = csv_stats  # type: ignore
        else:
            all_stats[(dp, tp, pp, ep, dp_dcn, pp_dcn, bs)] = stats  # type: ignore
    return all_stats


def get_all_stats(
    model: str,
    version: str,
    workload: str,
    prefill_or_decode: str = "",
    max_num_chips: int = -1,
    batch_size: int = -1,
) -> dict[tuple[int, ...], dict[str, Any]]:
    '''
    Returns all stats dictionaries for all experiments of the given model and version.
    Index the returned dict by @return[(dp, tp, pp, ep, dp_dcn, pp_dcn, batch_size)].
    Set @max_num_chips or @batch_size to -1 to ignore these filters.
    '''
    all_stats = get_all_stats_helper(
        model, version, workload, prefill_or_decode, max_num_chips, batch_size, read_csv=False
    )
    return all_stats  # type: ignore


def get_all_op_stats(
    model: str,
    version: str,
    workload: str,
    prefill_or_decode: str = "",
    max_num_chips: int = -1,
    batch_size: int = -1,
    read_json_with_csv: bool = False,
    results_path: str | None = None,
) -> dict[tuple[int, ...], list[dict[str, Any]] | tuple[dict[str, Any], list[dict[str, Any]]]]:
    '''
    Returns all per-op stats csv for all experiments of the given model and version.
    Index the returned dict by @return[(dp, tp, pp, ep, dp_dcn, pp_dcn, batch_size)].
    Set @max_num_chips or @batch_size to -1 to ignore these filters.
    '''
    all_stats = get_all_stats_helper(
        model, version, workload, prefill_or_decode, max_num_chips, batch_size,
        read_csv=True, read_json_with_csv=read_json_with_csv,
        results_path=results_path,
    )
    return all_stats  # type: ignore


def get_pareto_frontier(
    all_stats: list[dict[str, Any]],
    metric_cmp_fns: list[Callable[[dict[str, Any], dict[str, Any]], bool]],
) -> list[dict[str, Any]]:
    '''
    Returns the pareto frontier of the given stats.
    @metric_cmp_fns is a list of metric functions to consider for the pareto frontier.
        Each metric function should return True if the first stat is better than the second stat,
        and False otherwise.
    '''
    pareto_frontier = []
    for i, stat in enumerate(all_stats):
        is_pareto = all(
            any(metric_cmp(stat, other_stat) for metric_cmp in metric_cmp_fns)
            for other_stat in all_stats if other_stat != stat
        )
        if is_pareto:
            pareto_frontier.append(stat)
    return pareto_frontier


def get_optimal_stats_for_max_num_chips(
    model: str,
    version: str,
    max_num_chips: int = 1024,
    workload: str = "inference",
    prefill_or_decode: str = "prefill",
    perf_metric: str | Callable[[dict[str, Any]], Any] = "total_execution_time_ns",
    min_or_max_metric: str = "min",
    batch_size: int = 128,
    all_stats: dict | None = None,
) -> dict[str, Any]:
    '''
    Returns the stats dictionary for the optimal configuration
    (e.g., parallelism config and batch size) for the given model and version.
    @perf_metric: one of [
            "total_execution_time_ns",
            "overlapped_compute_time_ns",
            "compute_only_time_ns",
            "memory_only_time_ns",
            "ici_bound_time_ns",
            "throughput_tokens_per_sec",
            "throughput_tokens_per_sec_request",
            "TPOT_ms_request",
    ].
    @min_or_max_metric: one of ["min", "max"]; whether to use the min or max value of @perf_metric as the optimal.
    '''
    if isinstance(perf_metric, str):
        perf_metric_fn = lambda x: x[perf_metric]
    else:
        perf_metric_fn = perf_metric
    if not all_stats:
        all_stats = get_all_stats(model, version, workload, prefill_or_decode, max_num_chips, batch_size)
    assert len(all_stats) > 0, \
        f"No stats found for {model} v{version} {workload} {prefill_or_decode} with max_num_chips={max_num_chips} and batch_size={batch_size}"
    if min_or_max_metric == "min":
        optimal_stats = min(all_stats.values(), key=perf_metric_fn)
    elif min_or_max_metric == "max":
        optimal_stats = max(all_stats.values(), key=perf_metric_fn)
    else:
        raise ValueError(f"Invalid min_or_max_metric: {min_or_max_metric}")

    return optimal_stats


def get_min_num_chips(
    model: str,
    version: str,
    workload: str = "inference",
    prefill_or_decode: str = "prefill",
    batch_size: int = 1,
    all_stats: dict | None = None,
) -> int:
    '''
    Returns the minimum number of chips that can run the workload on NPU @version with @batch_size.
    '''
    if not all_stats:
        all_stats = get_all_stats(model, version, workload, prefill_or_decode, -1, batch_size)

    # filter out OOM configs
    valid_stats = {
        k: v for k, v in all_stats.items()
        if v["out_of_memory"] is False
    }

    min_num_chips = min(
        int(stat["sim_config"]["num_chips"]) for stat in valid_stats.values()
    )
    return min_num_chips


def get_component_data_from_file(results_file: str, component: str) -> int:
    '''
    Returns the bounded-by time (exposed time) specified by @component.
    @component is one of ["Execution time", "Compute time", "Memory time", "ICI/NVLink time"].
    If @component is "Execution time", returns the overlapped time between compute and *memory or network*.
    If @component is "Compute time" or "Memory time", returns the compute-only or memory-only time.
    If @component is "ICI/NVLink time", returns the *sum of the overlapped network+memory (no compute) time and the network-only time*
    (i.e., the network time that is not overlapped with compute).
    The total execution time is the sum of get_component_data() for calling all types of components.
    '''

    with open(results_file, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        bounded_by_key = component.removesuffix(" time")
        if bounded_by_key == "Execution":
            data_point = sum([
                min(
                    int(row["Compute time"]),
                    max(
                        int(row["Memory time"]),
                        int(row["ICI/NVLink time"])
                    )
                ) * int(row["Count"])
                for row in reader
            ])
        elif bounded_by_key == "Compute":
            data_point = sum([
                (int(row["Compute time"]) - max(int(row["Memory time"]), int(row["ICI/NVLink time"])))
                * int(row["Count"])
                for row in reader
                if row["Bounded-by"] == bounded_by_key
            ])
        elif bounded_by_key == "Memory":
            data_point = sum([
                (int(row["Memory time"]) - max(int(row["Compute time"]), int(row["ICI/NVLink time"])))
                * int(row["Count"])
                for row in reader
                if row["Bounded-by"] == bounded_by_key
            ])
        elif bounded_by_key == "ICI/NVLink":
            data_point = sum([
                abs(int(row["ICI/NVLink time"]) - int(row["Compute time"]))
                * int(row["Count"])
                for row in reader
                if row["Bounded-by"] == bounded_by_key
            ])
        else:
            raise ValueError(f"Invalid bounded_by_key: {bounded_by_key}")
    return data_point


def get_total_execution_time_from_file(results_file: str) -> int:
    '''
    Returns the total execution time (sum of all components) from the results file.
    '''
    with open(results_file, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        data_point = sum([
            int(row["Execution time"]) * int(row["Count"])
            for row in reader
        ])
    return data_point


# def get_optimal_parallelism_config(
#     model: str,
#     version: str,
#     max_num_chips: int = 1024,
#     workload: str = "inference",
#     prefill_or_decode: str = "prefill",
# ) -> tuple[int, int, int, int]:
#     '''
#     ***Deprecated: Use get_optimal_stats_for_max_num_chips() instead.***

#     Returns the optimal (DP, TP, PP, exe time ns) configuration for the model and TPU version.
#     '''
#     model_results_dir = f"{RESULTS_PATH}/{model}/"
#     pconfig_dirs = os.listdir(model_results_dir)

#     if workload == "training":
#         prefill_or_decode = ""
#     elif workload == "inference":
#         prefill_or_decode = f"_{prefill_or_decode}"
#     else:
#         raise ValueError(f"Invalid workload: {workload}")

#     all_config_times = []
#     for pconfig_dir in pconfig_dirs:
#         dpstr, tpstr, ppstr = pconfig_dir.split("-")
#         dp = int(dpstr.removeprefix("dp"))
#         tp = int(tpstr.removeprefix("tp"))
#         pp = int(ppstr.removeprefix("pp"))
#         if dp * tp * pp > max_num_chips:
#             continue
#         results_file = os.path.join(
#             model_results_dir, pconfig_dir, f"{workload}-v{version}{prefill_or_decode}.csv"
#         )
#         if not os.path.exists(results_file):
#             continue
#         data_point = get_total_execution_time_from_file(results_file)
#         all_config_times.append((dp, tp, pp, data_point))

#     optimal_config = min(all_config_times, key=lambda x: x[3])

#     return optimal_config


# def get_all_optimal_parallelism_configs(
#     models: Sequence[str],
#     versions: Sequence[str],
#     max_num_chips: int = 1024,
#     dump_configs: bool = False,
#     workload: str = "inference",
#     prefill_or_decode: str = "prefill",
# ) -> dict[str, dict[str, tuple[int, int, int, int]]]:
#     '''
#     ***Deprecated: Use get_optimal_stats_for_max_num_chips() instead.***

#     Returns the optimal (DP, TP, PP, exe time ns) configuration for each model and TPU version.
#     @return[m][v] is the optimal (DP, TP, PP, exe time ns) configuration for model m and TPU version v.
#     '''
#     optimal_configs = {}
#     for model in models:
#         optimal_configs[model] = {}
#         for v in versions:
#             model_results_dir = f"{RESULTS_PATH}/{model}/"
#             optimal_config = get_optimal_parallelism_config(
#                 model, v, max_num_chips, workload, prefill_or_decode
#             )
#             optimal_configs[model][v] = optimal_config

#     if dump_configs:
#         # dump as json
#         with open(f"optimal_parallelism_configs_{workload}_{max_num_chips}.json", "w") as f:
#             json.dump(optimal_configs, f, indent=4)

#     return optimal_configs


def get_latency_metric_name_and_min_max(
    model: str,
    workload: str,
    prefill_or_decode: str = "",
) -> tuple[str, str]:
    if is_model_llm(model):
        if workload == "inference":
            if prefill_or_decode == "prefill":
                metric_name = "TTFT_sec"
                metric_min_max = "min"
            elif prefill_or_decode == "decode":
                metric_name = "TPOT_ms_request"
                metric_min_max = "min"
            else:
                raise ValueError("Invalid LLM inference stage")
        elif workload == "training":
            metric_name = "total_execution_time_ns"
            metric_min_max = "min"
        else:
            raise ValueError(f"Invalid workload: {workload}")
    elif is_model_sd(model):
        metric_name = "latency_step_sec"
        metric_min_max = "min"
    elif is_model_dlrm(model):
        metric_name = "latency_ns"
        metric_min_max = "min"
    else:
        raise ValueError(f"Unknown model: {model}")
    return metric_name, metric_min_max


def get_throughput_metric_name_and_min_max(
    model: str,
    workload: str,
    prefill_or_decode: str = "",
) -> tuple[str, str]:
    if is_model_llm(model):
        if workload == "inference":
            metric_name = "throughput_tokens_per_sec"
            metric_min_max = "max"
        elif workload == "training":
            # TODO: use the same metric for latency and throughput for now
            # as we assume fixed batch size.
            metric_name = "total_execution_time_ns"
            metric_min_max = "min"
        else:
            raise ValueError(f"Invalid workload: {workload}")
    elif is_model_sd(model):
        metric_name = "throughput_step_per_sec_per_request"
        metric_min_max = "max"
    elif is_model_dlrm(model):
        metric_name = "throughput_requests_per_sec"
        metric_min_max = "max"
    else:
        raise ValueError(f"Unknown model: {model}")
    return metric_name, metric_min_max


def get_energy_eff_metric_name_and_min_max(
    model: str,
    workload: str,
    prefill_or_decode: str = "",
) -> tuple[str, str]:
    if is_model_llm(model):
        if workload == "inference":
            metric_name = "avg_power_efficiency_tkn_per_joule"
            metric_min_max = "max"
        elif workload == "training":
            metric_name = "avg_power_efficiency_iteration_per_joule"
            metric_min_max = "max"
        else:
            raise ValueError(f"Invalid workload: {workload}")
    elif is_model_sd(model):
        metric_name = "avg_power_efficiency_req_per_joule"
        metric_min_max = "max"
    elif is_model_dlrm(model):
        metric_name = "avg_power_efficiency_req_per_joule"
        metric_min_max = "max"
    else:
        raise ValueError(f"Unknown model: {model}")
    return metric_name, metric_min_max


def get_carbon_eff_metric_name_and_min_max(
    model: str,
    workload: str,
    prefill_or_decode: str = "",
) -> tuple[str, str]:
    if is_model_llm(model):
        if workload == "inference":
            metric_name = "avg_total_carbon_efficiency_tkn_per_kgCO2e"
            metric_min_max = "max"
        elif workload == "training":
            metric_name = "avg_total_carbon_efficiency_iteration_per_kgCO2e"
            metric_min_max = "max"
        else:
            raise ValueError(f"Invalid workload: {workload}")
    elif is_model_sd(model):
        metric_name = "avg_total_carbon_efficiency_req_per_kgCO2e"
        metric_min_max = "max"
    elif is_model_dlrm(model):
        metric_name = "avg_total_carbon_efficiency_req_per_kgCO2e"
        metric_min_max = "max"
    else:
        raise ValueError(f"Unknown model: {model}")
    return metric_name, metric_min_max


def get_slo_stat(
    model: str,
    workload: str,
    prefill_or_decode: str,
    version: str,
    all_stats: dict | None = None,
) -> dict[str, Any]:
    '''
    Return the reference stats that defines the SLO.
    '''

    if not all_stats:
        all_stats = get_all_stats(
            model, version, workload, prefill_or_decode
        )
    # filter out OOM configs
    all_stats = {
        k: v for k, v in all_stats.items()
        if v["out_of_memory"] is False
    }

    min_num_chips = get_min_num_chips(model, version, workload, prefill_or_decode, all_stats=all_stats)
    all_stats_chip = {
        config: stat for config, stat in all_stats.items()
        if stat["sim_config"]["num_chips"] <= min_num_chips
    }

    metric_name, metric_min_max = get_latency_metric_name_and_min_max(
        model, workload, prefill_or_decode
    )

    ref_stats = get_optimal_stats_for_max_num_chips(
        model,
        version,
        min_num_chips,
        workload,
        prefill_or_decode,
        metric_name,
        metric_min_max,
        1,
        all_stats_chip
    )

    return ref_stats
