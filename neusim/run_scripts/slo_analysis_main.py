### LLM SLO Analysis Main Program
### Dependencies:
###     - Run launch_npu_reuse_trace_generation.py first to generate the necessary trace files.


import copy
from functools import lru_cache
import json
import os
from typing import Any
from absl import app, flags, logging

from parallelbar import progress_starmap


from neusim.npusim.frontend import query_results_helper_lib as results_lib


__RESULTS_PATH = flags.DEFINE_string(
    "results_path",
    "../../results/carbon_NoPG/CI0.0624/UTIL0.6",
    "Path to the results directory",
)
__OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    "../../results/slo",
    "Path to the output directory",
)
__MODELS = flags.DEFINE_list(
    "models",
    ["llama3-8b", "llama2-13b", "llama3-70b", "llama3_1-405b"],
    "List of models to analyze",
)
__NPU_VERSIONS = flags.DEFINE_list(
    "npu_versions",
    ["2", "3", "4", "5p", "6p", "5e", "6e"],
    "List of NPU versions to analyze",
)
__MAX_NUM_CHIPS = flags.DEFINE_integer(
    "max_num_chips",
    2147483647,
    "Maximum number of chips to analyze.",
)
__WORKLOAD = flags.DEFINE_string(
    "workload", "inference", "Workload type: training or inference"
)
__SLO_SCALES = flags.DEFINE_list(
    "slo_scales",
    [str(x) for x in [1, 2, 3, 4, 5, 6]],
    "List of SLO scales to analyze",
)
__DEFAULT_SLO_MULTIPLE = flags.DEFINE_integer(
    "default_slo_multiple",
    1,
    "Default SLO scale multiple: we will use this times the performance of the reference config as the 1x SLO.",
)
# __UTIL_FACTOR = flags.DEFINE_float(
#     "util_factor",
#     0.6,
#     "Utilization factor for the SLO analysis.",
# )
__SKIP_EXIST = flags.DEFINE_boolean(
    "skip_exist", False, "Skip existing trace files"
)
__DEBUG = flags.DEFINE_boolean(
    "debug", False, "Debug mode"
)


MIN_NUM_CHIPS_TRAINING: dict[str, dict[str, int]] = {
    "llama3-8b": {
        "2": 64,
        "3": 32,
        "4": 32,
        "5p": 8,
        "6p": 4,
        "5e": 64,
        "6e": 32,
    },
    "llama2-13b": {
        "2": 64,
        "3": 32,
        "4": 32,
        "5p": 16,
        "6p": 8,
        "5e": 64,
        "6e": 32,
    },
    "llama3-70b": {
        "2": 256,
        "3": 128,
        "4": 128,
        "5p": 64,
        "6p": 32,
        "5e": 256,
        "6e": 128,
    },
    "llama3_1-405b": {
        "2": 1024,
        "3": 512,
        "4": 512,
        "5p": 256,
        "6p": 128,
        "5e": 1024,
        "6e": 512,
    },
}


MIN_NUM_CHIPS_INFERENCE: dict[str, dict[str, int]] = {
    "llama3-8b": {
        "2": 2,
        "3": 1,
        "4": 1,
        "5p": 1,
        "6p": 1,
        "5e": 2,
        "6e": 1,
    },
    "llama2-13b": {
        "2": 2,
        "3": 1,
        "4": 1,
        "5p": 1,
        "6p": 1,
        "5e": 2,
        "6e": 1,
    },
    "llama3-70b": {
        "2": 16,
        "3": 8,
        "4": 8,
        "5p": 2,
        "6p": 1,
        "5e": 16,
        "6e": 8,
    },
    "llama3_1-405b": {
        "2": 64,
        "3": 32,
        "4": 32,
        "5p": 16,
        "6p": 8,
        "5e": 64,
        "6e": 32,
    },
    "dlrm-s": {
        "2": 2,
        "3": 1,
        "4": 1,
        "5p": 1,
        "6p": 1,
        "5e": 2,
        "6e": 1,
    },
    "dlrm-m": {
        "2": 4,
        "3": 2,
        "4": 2,
        "5p": 1,
        "6p": 1,
        "5e": 4,
        "6e": 2,
    },
    "dlrm-l": {
        "2": 8,
        "3": 4,
        "4": 4,
        "5p": 2,
        "6p": 1,
        "5e": 8,
        "6e": 4,
    },
    "dit-xl": {
        "2": 1,
        "3": 1,
        "4": 1,
        "5p": 1,
        "6p": 1,
        "5e": 1,
        "6e": 1,
    },
    "gligen": {
        "2": 1,
        "3": 1,
        "4": 1,
        "5p": 1,
        "6p": 1,
        "5e": 1,
        "6e": 1,
    }
}


VERSION_TO_COST = results_lib.VERSION_TO_COST

# SLO_METRIC_LLM_INFERENCE_PREFILL = "TTFT_sec"
# SLO_METRIC_LLM_INFERENCE_DECODE = "TPOT_ms_request"
# SLO_METRIC_LLM_TRAINING = "total_execution_time_ns"
# SLO_METRIC_DLRM_INFERENCE = "latency_ns"
# SLO_METRIC_SD_INFERENCE = "latency_step_sec"


# @lru_cache(maxsize=None)
# def get_llm_inference_slo_stats(model: str, prefill_or_decode: str) -> dict[str, Any]:
#     '''
#     Return the reference stats that defines the SLO.
#     '''
#     all_stats = results_lib.get_all_stats(
#         model, "5p", "inference", prefill_or_decode,
#         max_num_chips=MIN_NUM_CHIPS_INFERENCE[model]["5p"],
#         batch_size=-1,
#     )
#     all_stats_chip = {
#         config: stat for config, stat in all_stats.items()
#         if stat["sim_config"]["num_chips"] <= MIN_NUM_CHIPS_INFERENCE[model]["5p"]
#     }
#     if prefill_or_decode == "prefill":
#         metric = SLO_METRIC_LLM_INFERENCE_PREFILL
#     elif prefill_or_decode == "decode":
#         metric = SLO_METRIC_LLM_INFERENCE_DECODE
#     else:
#         raise ValueError(f"Unknown prefill_or_decode: {prefill_or_decode}")
#     ref_stats = results_lib.get_optimal_stats_for_max_num_chips(
#         model,
#         "5p",
#         MIN_NUM_CHIPS_INFERENCE[model]["5p"],
#         "inference",
#         prefill_or_decode,
#         metric,
#         "min",
#         -1,
#         all_stats_chip
#     )

#     return ref_stats


def get_min_num_chips(model: str, workload: str, v: str, prefill_or_decode: str, batch_size: int = 1, all_stats: dict | None = None) -> int:
    '''
    Returns the minimum number of chips for the given model and workload.
    '''
    return results_lib.get_min_num_chips(
        model, v, workload, prefill_or_decode, batch_size, all_stats
    )


@lru_cache(maxsize=None)
def get_slo_stats(model: str, workload: str, prefill_or_decode: str) -> dict[str, Any]:
    '''
    Return the reference stats that defines the SLO.
    '''
    if workload == "training":
        bs = 32
    elif workload == "inference":
        if results_lib.is_model_dlrm(model):
            bs = 1024
        else:
            bs = 1
    else:
        raise ValueError(f"Unknown workload: {workload}")
    all_stats = results_lib.get_all_stats(
        model, "5p", workload, prefill_or_decode,
        batch_size=bs,
    )
    return results_lib.get_slo_stat(
        model, workload, prefill_or_decode, "5p", all_stats
    )


def get_slo(model: str, workload: str, prefill_or_decode: str) -> float:
    '''
    Returns the SLO for the given model.
    For prefill, the SLO is TTFT_sec. For decode, the SLO is TPOT_ms_request.
    SLO is determined as the performance on TPUv5p with min # of chips.
    '''
    metric_name, metric_min_max = results_lib.get_latency_metric_name_and_min_max(
        model, workload, prefill_or_decode
    )
    ref_stats = get_slo_stats(model, workload, prefill_or_decode)
    return ref_stats[metric_name] * __DEFAULT_SLO_MULTIPLE.value


def get_slo_multiple(model: str, workload: str, prefill_or_decode: str, slo_scale: float | int) -> float:
    '''
    Returns the TTFT seconds SLO multipled by @slo_scale.
    '''
    return get_slo(model, workload, prefill_or_decode) * slo_scale


def analyze(
    model: str, workload: str, prefill_or_decode: str
):
    analysis_result_stats = {
        int(slo_scale): { "2": {}, "3": {}, "4": {}, "5p": {} }
        for slo_scale in __SLO_SCALES.value
    }
    batch_size = 32 if workload == "training" else -1
    # min_num_chips_map = (
    #     MIN_NUM_CHIPS_TRAINING if workload == "training" else MIN_NUM_CHIPS_INFERENCE
    # )

    slo_metric, slo_metric_min_max = results_lib.get_latency_metric_name_and_min_max(
        model, workload, prefill_or_decode
    )
    eff_metric, eff_metric_min_max = results_lib.get_throughput_metric_name_and_min_max(
        model, workload, prefill_or_decode
    )
    energy_metric, energy_metric_min_max = results_lib.get_energy_eff_metric_name_and_min_max(
        model, workload, prefill_or_decode
    )

    def get_eff_metric_for_stat(stat: dict[str, Any]) -> float:
        return stat[eff_metric] / stat["sim_config"]["num_chips"]

    def get_energy_metric_for_stat(stat: dict[str, Any]) -> float:
        '''Always use "min" metric for energy efficiency (e.g., Joule/Good).'''
        if energy_metric_min_max == "max":
            return 1 / stat["carbon_and_energy_stats"]["1"][energy_metric]
        elif energy_metric_min_max == "min":
            return stat["carbon_and_energy_stats"]["1"][energy_metric]
        else:
            raise ValueError(f"Unknown energy metric min_max: {energy_metric_min_max}")

    def get_cost_for_stat(stat: dict[str, Any], v: str) -> float:
        '''
        Returns the cost (good/$) for the given stat.
        Cost is calculated as:
            cost = num_chips * cost_per_chip_hour * total_execution_time_sec / 3600
        Only consider inference for now. TODO: add training cost.
        '''
        num_chips = stat["sim_config"]["num_chips"]
        cost_per_chip_hour = VERSION_TO_COST[v]  # chip*hour
        cost_per_sec = cost_per_chip_hour / 3600 * num_chips  # $/(chip*hour)/(3600sec/hour)*chip => $/sec
        tput = get_eff_metric_for_stat(stat) # good/sec
        return tput / cost_per_sec

    # TODO: define a better throughput metric for training to avoid this hack
    if workload == "training":
        eff_metric_display_name = f"eff_metric_{eff_metric}"
    else:
        eff_metric_display_name = eff_metric

    for slo_scale in __SLO_SCALES.value:
        slo_scale = int(slo_scale)
        slo = get_slo_multiple(model, workload, prefill_or_decode, slo_scale)

        for v in __NPU_VERSIONS.value:
            raw_all_stats = results_lib.get_all_stats(
                model, v, workload, prefill_or_decode,
                max_num_chips=__MAX_NUM_CHIPS.value,
                batch_size=batch_size,
            )
            # filter out config that exceeds memory capacity
            all_stats_by_mem_capacity = {
                config: stat for config, stat in raw_all_stats.items()
                if stat["out_of_memory"] is False
            }
            # filter out config that violates SLO
            all_stats = {
                config: stat for config, stat in all_stats_by_mem_capacity.items()
                if stat[slo_metric] <= slo
            }
            # all_stats = raw_all_stats
            if len(all_stats) == 0:
                # no configs can satisfy slo
                stats = [stat[slo_metric] for stat in all_stats_by_mem_capacity.values()]
                min_slo_metric = min(stats) if len(stats) > 0 else 0
                analysis_result_stats[slo_scale][v] = {
                    "min_num_chips": 0,
                    slo_metric: min_slo_metric,
                    eff_metric: 0,
                }
                continue

            # find min num of chips that satisfies SLO
            # min_num_chips = min(stat["sim_config"]["num_chips"] for stat in all_stats.values())
            # # enforce mem capacity limit
            # min_num_chips = max(
            #     get_min_num_chips(model, workload, v),
            #     min_num_chips,
            # )
            min_num_chips = get_min_num_chips(
                model, workload, v, prefill_or_decode, batch_size, all_stats
            )
            # min_num_chips = 1
            # if "dlrm" in model:
            #     min_num_chips = 8

            ### find min chip stats
            min_chip_stats = {
                config: stat for config, stat in all_stats.items()
                # "==" enforces using min_num_chips
                if stat["sim_config"]["num_chips"] == min_num_chips
            }
            min_chip_stat = results_lib.get_optimal_stats_for_max_num_chips(
                model,
                v,
                -1,
                workload,
                prefill_or_decode,
                get_eff_metric_for_stat,
                eff_metric_min_max,
                batch_size,
                min_chip_stats,
            )
            ###

            ### find optimal chip stats
            optimal_stats = copy.deepcopy(all_stats)
            # {
            #     config: stat for config, stat in all_stats.items()
            #     # ">=" finds the optimal num chips
            #     if stat["sim_config"]["num_chips"] >= min_num_chips
            # }
            optimal_stat = results_lib.get_optimal_stats_for_max_num_chips(
                model,
                v,
                -1,
                workload,
                prefill_or_decode,
                get_eff_metric_for_stat,
                eff_metric_min_max,
                batch_size,
                optimal_stats,
            )
            ###

            ### find optimal energy efficiency stats
            optimal_energy_eff_stats = copy.deepcopy(all_stats)
            # {
            #     config: stat for config, stat in all_stats.items()
            #     # ">=" finds the optimal num chips
            #     if stat["sim_config"]["num_chips"] >= min_num_chips
            # }
            optimal_energy_eff_stat = results_lib.get_optimal_stats_for_max_num_chips(
                model,
                v,
                -1,
                workload,
                prefill_or_decode,
                get_energy_metric_for_stat,
                "min",
                batch_size,
                optimal_energy_eff_stats,
            )
            # optimal_energy_eff_stat = {}
            ###

            ### find optimal economic cost stats
            # optimal_cost_stats = {
            #     config: stat for config, stat in all_stats_by_mem_capacity.items()
            #     # ">=" finds the optimal num chips
            #     if stat["sim_config"]["num_chips"] >= min_num_chips
            # }
            optimal_cost_stats = all_stats_by_mem_capacity
            optimal_cost_stat = results_lib.get_optimal_stats_for_max_num_chips(
                model,
                v,
                -1,
                workload,
                prefill_or_decode,
                lambda x: get_cost_for_stat(x, v),
                "max",
                batch_size,
                optimal_cost_stats,
            )
            ###

            optimal_stats_filepath = os.path.abspath(optimal_stat["sim_config"]["output_file_path"]).replace(".csv", ".json")
            min_chip_stats_filepath = os.path.abspath(min_chip_stat["sim_config"]["output_file_path"]).replace(".csv", ".json")
            optimal_energy_eff_filepath = os.path.abspath(optimal_energy_eff_stat["sim_config"]["output_file_path"]).replace(".csv", ".json")
            optimal_cost_filepath = os.path.abspath(optimal_cost_stat["sim_config"]["output_file_path"]).replace(".csv", ".json")
            if results_lib.is_model_llm(model):
                if workload == "inference":
                    optimal_stats_filepath = os.path.abspath(optimal_stat["sim_config"]["output_file_path"].replace(".csv", f"_{prefill_or_decode}.json"))
                    min_chip_stats_filepath = os.path.abspath(min_chip_stat["sim_config"]["output_file_path"].replace(".csv", f"_{prefill_or_decode}.json"))
                    optimal_energy_eff_filepath = os.path.abspath(optimal_energy_eff_stat["sim_config"]["output_file_path"].replace(".csv", f"_{prefill_or_decode}.json"))
                    optimal_cost_filepath = os.path.abspath(optimal_cost_stat["sim_config"]["output_file_path"].replace(".csv", f"_{prefill_or_decode}.json"))

            analysis_result_stats[slo_scale][v] = {
                "min_num_chips": min_num_chips,
                "optimal_num_chips": optimal_stat["sim_config"]["num_chips"],
                "optimal_energy_eff_num_chips": optimal_energy_eff_stat["sim_config"]["num_chips"],
                "optimal_cost_num_chips": optimal_cost_stat["sim_config"]["num_chips"],
                "optimal_cost_global_batch_size": optimal_cost_stat["sim_config"]["global_batch_size"],

                f"ref_{slo_metric}": slo,

                slo_metric: min_chip_stat[slo_metric],
                f"optimal_{slo_metric}": optimal_stat[slo_metric],
                f"optimal_energy_eff_{slo_metric}": optimal_energy_eff_stat[slo_metric],
                f"optimal_cost_{slo_metric}": optimal_cost_stat[slo_metric],

                eff_metric_display_name: min_chip_stat[eff_metric],
                f"optimal_{eff_metric_display_name}": optimal_stat[eff_metric],
                f"optimal_energy_eff_joule_per_good": 1 / optimal_energy_eff_stat["carbon_and_energy_stats"]["1"][energy_metric],

                "throughput_per_chip": min_chip_stat[eff_metric] / min_chip_stat["sim_config"]["num_chips"],
                "optimal_throughput_per_chip": optimal_stat[eff_metric] / optimal_stat["sim_config"]["num_chips"],
                "optimal_energy_eff_per_chip": 1 / optimal_energy_eff_stat["carbon_and_energy_stats"]["1"][energy_metric] / optimal_energy_eff_stat["sim_config"]["num_chips"],

                "optimal_cost_good_per_dollar": get_cost_for_stat(optimal_cost_stat, v),
                # "optimal_cost_mem_footprint_GB": optimal_cost_stat["sim_config"]["num_chips"] * optimal_cost_stat["mem_footprint_GB"],  # TODO: change this back
                "optimal_cost_total_memory_capacity_GB": optimal_cost_stat["sim_config"]["num_chips"] * optimal_cost_stat["sim_config"]["hbm_size_GB"],

                "optimal_energy_eff_watt": optimal_energy_eff_stat["avg_power_W"],
                "optimal_energy_eff_watt_per_chip": optimal_energy_eff_stat["avg_power_W"] / optimal_energy_eff_stat["sim_config"]["num_chips"],

                "stats_file": min_chip_stats_filepath,
                "optimal_stats_file": optimal_stats_filepath,
                "optimal_energy_eff_stats_file": optimal_energy_eff_filepath,
                "optimal_cost_stats_file": optimal_cost_filepath,
            }

    outpath = f"{__OUTPUT_PATH.value}/{model}"
    os.makedirs(outpath, exist_ok=True)
    outfile_path = f"{outpath}/{workload}.json"
    if results_lib.is_model_llm(model):
        if workload == "inference":
            outfile_path = f"{outpath}/{workload}-{prefill_or_decode}.json"
    with open(outfile_path, "w") as f:
        json.dump(analysis_result_stats, f, indent=4)


def main(argv):
    del argv  # Unused.

    results_lib.set_results_path(__RESULTS_PATH.value)

    # inference
    if __WORKLOAD.value == "inference":
        params = [
            (m, "inference", "prefill") for m in __MODELS.value
        ] + [
            (m, "inference", "decode") for m in __MODELS.value if results_lib.is_model_llm(m)
        ]
    elif __WORKLOAD.value == "training":
        params = [
            (m, "training", "") for m in __MODELS.value
        ]
    else:
        raise ValueError(f"Unknown workload: {__WORKLOAD.value}")

    if __DEBUG.value:
        for param in params:
            analyze(*param)
    else:
        progress_starmap(analyze, params)


if __name__ == "__main__":
    app.run(main)
