### Carbon Analysis Main Program
### Dependencies:
###     - Run llm_results_analysis_main.py first to generate the necessary stats files.
### Note: For single-chip, all stats are reported for a single LLM layer.


import itertools
import json
import os
import re
import stat
from typing import Any, Callable
from absl import app, flags, logging

# from parallelbar import progress_starmap
import ray
import tqdm

from neusim.configs.models.ModelConfig import ModelConfig
from neusim.npusim.frontend import query_results_helper_lib as results_lib
from neusim.npusim.frontend import energy_carbon_analysis_lib as carbon_lib


# ray.init()

__RESULTS_PATH = flags.DEFINE_string(
    "results_path",
    "../../results/raw_energy",
    "Path to the results directory",
)
RESULTS_PATH = None
__MODELS = flags.DEFINE_list(
    "models",
    [
        "llama3-8b", "llama2-13b", "llama3-70b", "llama3_1-405b",
        "dlrm-s", "dlrm-m", "dlrm-l",
        "dit-xl", "gligen",
    ],
    "List of models to analyze",
)
__NPU_VERSIONS = flags.DEFINE_list(
    "npu_versions",
    ["2", "3", "4", "5p", "6p", "5e", "6e"],
    # ["2", "3", "4"],  # for now, we only have public power and carbon data for v3 and v4
    "List of NPU versions to analyze",
)
__WORKLOAD = flags.DEFINE_string(
    "workload", "training", "Workload type: training or inference"
)
WORKLOAD = None
__LIFETIME = flags.DEFINE_list(
    "lifetime",
    [str(x) for x in [1]],
    "List of lifetime of the NPU chip in data center in years",
)
LIFETIME = None
__CARBON_INTENSITIES = flags.DEFINE_list(
    # 0.0717 0.0624 0.1012 0.1155 0.1352
    "carbon_intensity",
    [str(x) for x in [0.1352]],
    "Carbon intensity override (kgCO2e/kWh)"
)
CARBON_INTENSITIES = None
__UTILIZATION_FACTOR = flags.DEFINE_float(
    "utilization_factor", 0.6, "Utilization factor (i.e., duty cycle) for scaling power consumption"
)
UTILIZATION_FACTOR = None
__POWER_GATING_STRATEGY = flags.DEFINE_string(
    "power_gating_strategy", "disabled", "Power gating strategy"
)
POWER_GATING_STRATEGY = None
__DVFS_POLICY = flags.DEFINE_string(
    "dvfs_policy",
    "None",
    "DVFS policy mode. Options: Ideal, None",
)
DVFS_POLICY = None
__SKIP_EXIST = flags.DEFINE_boolean(
    "skip_exist", False, "Skip existing trace files"
)
__DEBUG = flags.DEFINE_boolean(
    "debug", False, "Debug mode"
)
DEBUG = False


def init_flags():
    global RESULTS_PATH
    # global OUTPUT_PATH
    global WORKLOAD
    global LIFETIME
    global CARBON_INTENSITIES
    global UTILIZATION_FACTOR
    global POWER_GATING_STRATEGY
    global DVFS_POLICY
    global DEBUG

    RESULTS_PATH = __RESULTS_PATH.value
    # OUTPUT_PATH = __OUTPUT_PATH.value
    WORKLOAD = __WORKLOAD.value
    LIFETIME = __LIFETIME.value
    CARBON_INTENSITIES = __CARBON_INTENSITIES.value
    UTILIZATION_FACTOR = __UTILIZATION_FACTOR.value
    POWER_GATING_STRATEGY = __POWER_GATING_STRATEGY.value
    DVFS_POLICY = __DVFS_POLICY.value
    DEBUG = __DEBUG.value


@ray.remote
def analyze_carbon_and_energy_for_stats(
    stats: dict[str, Any],
    unit_per_sec_fn: Callable[[dict[str, Any]], float] = lambda x: x["throughput_tokens_per_sec"],
    carbon_eff_unit: str = "unit_per_kgCO2e",
    power_eff_unit: str = "unit_per_joule",
    util_factor: float = 0.6,
    carbon_intensity: float = -1,
    dump_to_file_fn: Callable[[dict[str, Any]], Any] | None = None,
):
    '''
    Add "carbon_and_energy_stats" and "carbon_and_energy_stats_zero_embodied" to the @stats dictionary.
    Returns the updated stats dictionary.
    '''
    global LIFETIME
    global POWER_GATING_STRATEGY

    config = ModelConfig.model_validate(stats["sim_config"])
    num_chips = config.num_chips
    PUE = config.PUE
    idle_ratio = 1 - util_factor
    energy_stats = stats["energy_stats"][POWER_GATING_STRATEGY]

    if POWER_GATING_STRATEGY != "NoPG":
        idle_power_W = (
            config.static_power_hbm_W * 0.0897 +
            config.static_power_other_W
        ) * num_chips
    else:
        idle_power_W = config.idle_power_W * num_chips
    avg_active_power_W = energy_stats["avg_power_W"] * num_chips
    avg_power_W = idle_ratio * idle_power_W + util_factor * avg_active_power_W
    peak_power_W = energy_stats["peak_power_W"] * num_chips
    static_sa_ratio = energy_stats["static_sa_energy_J"] / energy_stats["total_energy_J"]
    static_vu_ratio = energy_stats["static_vu_energy_J"] / energy_stats["total_energy_J"]
    static_sram_ratio = energy_stats["static_sram_energy_J"] / energy_stats["total_energy_J"]
    static_ici_ratio = energy_stats["static_ici_energy_J"] / energy_stats["total_energy_J"]
    static_hbm_ratio = energy_stats["static_hbm_energy_J"] / energy_stats["total_energy_J"]
    static_other_ratio = energy_stats["static_other_energy_J"] / energy_stats["total_energy_J"]
    dynamic_sa_ratio = energy_stats["dynamic_sa_energy_J"] / energy_stats["total_energy_J"]
    dynamic_vu_ratio = energy_stats["dynamic_vu_energy_J"] / energy_stats["total_energy_J"]
    dynamic_sram_ratio = energy_stats["dynamic_sram_energy_J"] / energy_stats["total_energy_J"]
    dynamic_ici_ratio = energy_stats["dynamic_ici_energy_J"] / energy_stats["total_energy_J"]
    dynamic_hbm_ratio = energy_stats["dynamic_hbm_energy_J"] / energy_stats["total_energy_J"]
    dynamic_other_ratio = energy_stats["dynamic_other_energy_J"] / energy_stats["total_energy_J"]

    stats["idle_power_W"] = idle_power_W
    stats["avg_active_power_W"] = avg_active_power_W
    stats["avg_power_W"] = avg_power_W
    stats["peak_power_W"] = peak_power_W

    stats["idle_ratio"] = idle_ratio
    stats["active_ratio"] = util_factor
    stats["static_sa_ratio"] = static_sa_ratio
    stats["static_vu_ratio"] = static_vu_ratio
    stats["static_sram_ratio"] = static_sram_ratio
    stats["static_ici_ratio"] = static_ici_ratio
    stats["static_hbm_ratio"] = static_hbm_ratio
    stats["static_other_ratio"] = static_other_ratio
    stats["dynamic_sa_ratio"] = dynamic_sa_ratio
    stats["dynamic_vu_ratio"] = dynamic_vu_ratio
    stats["dynamic_sram_ratio"] = dynamic_sram_ratio
    stats["dynamic_ici_ratio"] = dynamic_ici_ratio
    stats["dynamic_hbm_ratio"] = dynamic_hbm_ratio
    stats["dynamic_other_ratio"] = dynamic_other_ratio

    if carbon_intensity == -1:
        carbon_intensity = config.carbon_intensity_kgCO2_per_kWh

    def get_carbon_and_energy_stats(embodied_carbon_zero: bool) -> dict[str, Any]:
        embodied_carbon = (
            config.embodied_carbon_kgCO2
            # divided by 4 below since we have 4 chips per server
            # + 15.36 * 2 / 4  # 2-socket AMD EPYC 7713 CPU
            # + 0.065 * 2048 / 4  # 2TB DRAM
            # + 0.01 * 2048 / 4  # 2TB SSD
        ) * num_chips if not embodied_carbon_zero else 0

        # get performance stats
        slowdown = stats["energy_stats"][POWER_GATING_STRATEGY]["component_stats"]["slowdown"]
        goodput = unit_per_sec_fn(stats) / slowdown

        carbon_stats = {}
        assert LIFETIME
        for lifetime in LIFETIME:
            lifetime = int(lifetime)
            avg_total_carbon_efficiency = carbon_lib.get_total_carbon_efficiency(
                goodput * util_factor, PUE, lifetime, carbon_intensity, avg_power_W, embodied_carbon
            )
            avg_power_efficiency = carbon_lib.get_power_efficiency(
                goodput * util_factor, PUE, avg_power_W
            )
            avg_total_carbon = carbon_lib.get_total_carbon_emission_kgCO2e(
                PUE, carbon_intensity, lifetime, avg_power_W, embodied_carbon
            )
            avg_total_energy_consumption = carbon_lib.get_operational_energy_consumption_kWh(
                PUE, lifetime, avg_power_W
            )
            avg_idle_energy_consumption = carbon_lib.get_operational_energy_consumption_kWh(
                PUE, lifetime * idle_ratio, idle_power_W
            )
            avg_active_energy_consumption = carbon_lib.get_operational_energy_consumption_kWh(
                PUE, lifetime * util_factor, avg_active_power_W
            )
            # below are all for active state energy consumption
            avg_static_sa_energy_consumption = avg_active_energy_consumption * static_sa_ratio
            avg_static_vu_energy_consumption = avg_active_energy_consumption * static_vu_ratio
            avg_static_sram_energy_consumption = avg_active_energy_consumption * static_sram_ratio
            avg_static_ici_energy_consumption = avg_active_energy_consumption * static_ici_ratio
            avg_static_hbm_energy_consumption = avg_active_energy_consumption * static_hbm_ratio
            avg_static_other_energy_consumption = avg_active_energy_consumption * static_other_ratio
            avg_static_energy_consumption = (
                avg_static_sa_energy_consumption +
                avg_static_vu_energy_consumption +
                avg_static_sram_energy_consumption +
                avg_static_ici_energy_consumption +
                avg_static_hbm_energy_consumption +
                avg_static_other_energy_consumption
            )
            avg_dynamic_sa_energy_consumption = avg_active_energy_consumption * dynamic_sa_ratio
            avg_dynamic_vu_energy_consumption = avg_active_energy_consumption * dynamic_vu_ratio
            avg_dynamic_sram_energy_consumption = avg_active_energy_consumption * dynamic_sram_ratio
            avg_dynamic_ici_energy_consumption = avg_active_energy_consumption * dynamic_ici_ratio
            avg_dynamic_hbm_energy_consumption = avg_active_energy_consumption * dynamic_hbm_ratio
            avg_dynamic_other_energy_consumption = avg_active_energy_consumption * dynamic_other_ratio
            avg_dynamic_energy_consumption = (
                avg_dynamic_sa_energy_consumption +
                avg_dynamic_vu_energy_consumption +
                avg_dynamic_sram_energy_consumption +
                avg_dynamic_ici_energy_consumption +
                avg_dynamic_hbm_energy_consumption +
                avg_dynamic_other_energy_consumption
            )
            avg_embodied_carbon_percentage = embodied_carbon / avg_total_carbon
            if lifetime not in carbon_stats:
                carbon_stats[lifetime] = {}
            carbon_stats[lifetime][f"avg_total_carbon_efficiency_{carbon_eff_unit}"] = 1 / avg_total_carbon_efficiency
            carbon_stats[lifetime]["avg_total_energy_consumption_kWh"] = avg_total_energy_consumption
            carbon_stats[lifetime]["avg_idle_energy_consumption_kWh"] = avg_idle_energy_consumption
            carbon_stats[lifetime]["avg_active_energy_consumption_kWh"] = avg_active_energy_consumption
            carbon_stats[lifetime]["avg_static_energy_consumption_kWh"] = avg_static_energy_consumption
            carbon_stats[lifetime]["avg_dynamic_energy_consumption_kWh"] = avg_dynamic_energy_consumption
            carbon_stats[lifetime]["avg_static_sa_energy_consumption_kWh"] = avg_static_sa_energy_consumption
            carbon_stats[lifetime]["avg_static_vu_energy_consumption_kWh"] = avg_static_vu_energy_consumption
            carbon_stats[lifetime]["avg_static_sram_energy_consumption_kWh"] = avg_static_sram_energy_consumption
            carbon_stats[lifetime]["avg_static_ici_energy_consumption_kWh"] = avg_static_ici_energy_consumption
            carbon_stats[lifetime]["avg_static_hbm_energy_consumption_kWh"] = avg_static_hbm_energy_consumption
            carbon_stats[lifetime]["avg_static_other_energy_consumption_kWh"] = avg_static_other_energy_consumption
            carbon_stats[lifetime]["avg_dynamic_sa_energy_consumption_kWh"] = avg_dynamic_sa_energy_consumption
            carbon_stats[lifetime]["avg_dynamic_vu_energy_consumption_kWh"] = avg_dynamic_vu_energy_consumption
            carbon_stats[lifetime]["avg_dynamic_sram_energy_consumption_kWh"] = avg_dynamic_sram_energy_consumption
            carbon_stats[lifetime]["avg_dynamic_ici_energy_consumption_kWh"] = avg_dynamic_ici_energy_consumption
            carbon_stats[lifetime]["avg_dynamic_hbm_energy_consumption_kWh"] = avg_dynamic_hbm_energy_consumption
            carbon_stats[lifetime]["avg_dynamic_other_energy_consumption_kWh"] = avg_dynamic_other_energy_consumption
            carbon_stats[lifetime][f"avg_power_efficiency_{power_eff_unit}"] = avg_power_efficiency
            carbon_stats[lifetime]["avg_embodied_carbon_percentage"] = avg_embodied_carbon_percentage
            carbon_stats[lifetime]["avg_total_carbon_kgCO2e"]  = avg_total_carbon
            carbon_stats[lifetime]["embodied_carbon_kgCO2e"] = embodied_carbon

        return carbon_stats

    ### END OF get_carbon_and_energy_stats ###

    stats["carbon_and_energy_stats"] = get_carbon_and_energy_stats(embodied_carbon_zero=False)
    stats["carbon_and_energy_stats_zero_embodied"] = get_carbon_and_energy_stats(embodied_carbon_zero=True)

    if dump_to_file_fn:
        # dump stats to the output file
        outfile = dump_to_file_fn(stats)
        print(f"Generating carbon analysis results to {outfile}")

    # return stats


@ray.remote
def analyze_all_raw_results(
    model: str,
    version: str,
    workload: str,
    prefill_or_decode: str,
    carbon_intensity: float,
):
    global RESULTS_PATH
    assert RESULTS_PATH
    results_lib.set_results_path(RESULTS_PATH)
    all_stats = results_lib.get_all_stats(
        model, version, workload, prefill_or_decode
    )
    # logging.info(
    #     "%s, v%s, %s, %s, %s: Number of stats: %s",
    #     model, version, workload, prefill_or_decode, carbon_intensity, len(all_stats),
    # )
    print(f"{model}, v{version}, {workload}, {prefill_or_decode}, {carbon_intensity}: Number of stats: {len(all_stats)}")
    print(RESULTS_PATH)

    def dump_to_file_fn(stats: dict[str, Any]):
        outfile = re.sub(
            r"raw_[^/]*",
            f"carbon_{POWER_GATING_STRATEGY}/dvfs_{DVFS_POLICY}/CI{carbon_intensity}/UTIL{UTILIZATION_FACTOR}/",
            stats["sim_config"]["out_stats_file_path"],
        )
        # outfile = stats["sim_config"]["out_stats_file_path"].replace("/raw_energy/", f"/carbon_{POWER_GATING_STRATEGY}/CI{carbon_intensity}/UTIL{UTILIZATION_FACTOR}/")
        outpath = os.path.dirname(outfile)
        os.makedirs(outpath, exist_ok=True)
        with open(outfile, "w") as f:
            json.dump(stats, f, indent=4)
        return outfile

    if results_lib.is_model_llm(model):
        if workload == "training":
            def unit_per_sec_fn(stats: dict[str, Any]) -> float:
                '''training iteration/second'''
                raw_val = stats["total_execution_time_ns"]
                # if stats["sim_config"]["num_chips"] == 1:
                #     # use single layer stat for single chip
                #     num_layers = stats["sim_config"]["num_layers"]
                #     raw_val = raw_val / num_layers
                return 1 / (raw_val / 1e9)
            carbon_eff_unit="iteration_per_kgCO2e"
            power_eff_unit="iteration_per_joule"
        elif workload == "inference":
            if prefill_or_decode == "prefill":
                def unit_per_sec_fn(stats: dict[str, Any]) -> float:
                    '''inference tokens/second'''
                    raw_val = stats["throughput_tokens_per_sec"]
                    # if stats["sim_config"]["num_chips"] == 1:
                    #     # use single layer stat for single chip
                    #     num_layers = stats["sim_config"]["num_layers"]
                    #     raw_val = raw_val * num_layers
                    return raw_val
                # def dump_to_file_fn(stats: dict[str, Any]):
                #     # outfile = stats["sim_config"]["out_stats_file_path"].replace("/raw_energy/", f"/carbon_{POWER_GATING_STRATEGY}/CI{carbon_intensity}/UTIL{UTILIZATION_FACTOR}/")
                #     outfile = re.sub(
                #         r"raw_energy[^/]*",
                #         f"carbon_{POWER_GATING_STRATEGY}/CI{carbon_intensity}/UTIL{UTILIZATION_FACTOR}/",
                #         stats["sim_config"]["out_stats_file_path"],
                #     )
                #     outpath = os.path.dirname(outfile)
                #     os.makedirs(outpath, exist_ok=True)
                #     with open(outfile, "w") as f:
                #         json.dump(stats, f, indent=4)
                #     return outfile
            elif prefill_or_decode == "decode":
                def unit_per_sec_fn(stats: dict[str, Any]) -> float:
                    '''inference tokens/second'''
                    raw_val = stats["throughput_tokens_per_sec"]
                    # if stats["sim_config"]["num_chips"] == 1:
                    #     # use single layer stat for single chip
                    #     num_layers = stats["sim_config"]["num_layers"]
                    #     raw_val = raw_val * num_layers
                    return raw_val
                # def dump_to_file_fn(stats: dict[str, Any]):
                #     # outfile = stats["sim_config"]["out_stats_file_path"].replace("/raw_energy/", f"/carbon_{POWER_GATING_STRATEGY}/CI{carbon_intensity}/UTIL{UTILIZATION_FACTOR}/")
                #     outfile = re.sub(
                #         r"raw_energy[^/]*",
                #         f"carbon_{POWER_GATING_STRATEGY}/CI{carbon_intensity}/UTIL{UTILIZATION_FACTOR}/",
                #         stats["sim_config"]["out_stats_file_path"],
                #     )
                #     outpath = os.path.dirname(outfile)
                #     os.makedirs(outpath, exist_ok=True)
                #     with open(outfile, "w") as f:
                #         json.dump(stats, f, indent=4)
                #     return outfile
            else:
                raise ValueError(f"Invalid prefill_or_decode: {prefill_or_decode}")
            carbon_eff_unit="tkn_per_kgCO2e"
            power_eff_unit="tkn_per_joule"
        else:
            raise ValueError(f"Invalid workload: {workload}")
    else:
        if workload == "training":
            raise NotImplementedError(f"Training analysis for {model} is not implemented yet.")
        elif workload == "inference":
            def unit_per_sec_fn(stats: dict[str, Any]) -> float:
                '''inference requests/second'''
                raw_val = stats["throughput_requests_per_sec"]
                return raw_val
            carbon_eff_unit="req_per_kgCO2e"
            power_eff_unit="req_per_joule"
        else:
            raise ValueError(f"Invalid workload: {workload}")

    params = [
        (
            stats,
            unit_per_sec_fn,
            carbon_eff_unit,
            power_eff_unit,
            UTILIZATION_FACTOR,
            carbon_intensity,
            dump_to_file_fn,
        ) for stats in all_stats.values()
    ]

    if DEBUG:
        for param in tqdm.tqdm(params):
            analyze_carbon_and_energy_for_stats.remote(*param)  # type: ignore
    else:
        futures = []
        for param in params:
            futures.append(analyze_carbon_and_energy_for_stats.remote(*param))  # type: ignore
        # progress_starmap(
        #     analyze_carbon_and_energy_for_stats,
        #     params,
        # )
        ray.get(futures)


def main(argv):
    del argv  # Unused.

    global RESULTS_PATH
    global WORKLOAD
    global LIFETIME
    global CARBON_INTENSITIES
    global UTILIZATION_FACTOR
    global POWER_GATING_STRATEGY
    global DEBUG

    init_flags()
    assert RESULTS_PATH and WORKLOAD and LIFETIME and CARBON_INTENSITIES and UTILIZATION_FACTOR and POWER_GATING_STRATEGY

    # analyze all raw results
    futures = []
    for model in __MODELS.value:
        for v in __NPU_VERSIONS.value:
            for ci in CARBON_INTENSITIES:
                # logging.info("Analyzing %s v%s prefill", model, v)
                # analyze_all_raw_results(
                #     model, v, WORKLOAD, "prefill", float(ci)
                # )
                # logging.info("Analyzing %s v%s decode", model, v)
                # if "llama" in model and WORKLOAD == "inference":
                #     analy7ze_all_raw_results(
                #         model, v, WORKLOAD, "decode", float(ci)
                #     )
                if results_lib.is_model_llm(model) and WORKLOAD == "inference":
                    logging.info("Analyzing %s v%s prefill", model, v)
                else:
                    logging.info("Analyzing %s v%s", model, v)
                futures.append(
                    analyze_all_raw_results.remote(
                        model, v, WORKLOAD, "prefill", float(ci)
                    )
                )
                if results_lib.is_model_llm(model) and WORKLOAD == "inference":
                    logging.info("Analyzing %s v%s decode", model, v)
                    futures.append(
                        analyze_all_raw_results.remote(
                            model, v, WORKLOAD, "decode", float(ci)
                        )
                    )
    ray.get(futures)


if __name__ == "__main__":
    app.run(main)
