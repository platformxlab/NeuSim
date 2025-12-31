### Read from CSV trace file and add columns for per-op energy/power statistics


from copy import deepcopy
import json
import os
import csv
from typing import Any, Callable, Sequence

import numpy as np
import ray
from absl import app, flags, logging

from neusim.npusim.frontend import query_results_helper_lib as results_lib
from neusim.npusim.frontend import power_analysis_lib as power_lib
from neusim.configs.models.ModelConfig import ModelConfig
import neusim.npusim.frontend.Operator as Operator
from neusim.npusim.frontend.Operator import DVFSPolicy
from neusim.npusim.frontend.util import compute_component_slack_for_op


__RESULTS_PATH = flags.DEFINE_string(
    "results_path",
    "../../results/raw",
    "Path to the results directory",
)
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
    ["2", "3", "4", "5p"],
    "List of NPU versions to analyze",
)
__WORKLOAD = flags.DEFINE_string(
    "workload", "training", "Workload type: training or inference"
)
__POWER_GATING_STRATEGY = flags.DEFINE_list(
    "power_gating_strategy",
    ["NoPG", "Base", "HW", "Full", "Ideal"],
    "Power gating strategy. See power_lib.get_power_gating_config() for details.",
)
__DVFS_POLICY = flags.DEFINE_string(
    "dvfs_policy",
    DVFSPolicy.NONE.value,
    "DVFS policy mode. Options: Ideal, None",
)

__SKIP_EXIST = flags.DEFINE_boolean(
    "skip_exist", False, "Skip existing trace files"
)
__DEBUG = flags.DEFINE_boolean(
    "debug", False, "Debug mode"
)
__DUMP_INTERMEDIATE_ANALYSIS_RESULTS = flags.DEFINE_boolean(
    "dump_intermediate_analysis_results",
    False,
    "Whether to dump intermediate analysis results to files (may take a lot of time and disk space!)",
)
DEBUG_MODE = False
RESULTS_PATH = None
DUMP_INTERMEDIATE_ANALYSIS_RESULTS = False


def analyze_operator_energy(
    ops: list[Operator.Operator],
    config: ModelConfig,
    power_gating_strategy: str,
    energy_stats: dict[str, Any],
    dvfs_mode: str,
):
    # for op in ops:
    #     power_lib.analyze_operator_energy(
    #         op,
    #         config,
    #         pg_config=power_gating_strategy,
    #         dvfs_config=dvfs_mode,
    #     )
    if dvfs_mode != DVFSPolicy.NONE.value:
        config.enable_dvfs = True
    power_lib.analyze_all_operator_energy(
        ops,
        config,
        power_gating_strategy,
        dvfs_mode,
    )

    # get overall energy stats
    energy_stats["total_energy_J"] = sum([op.stats.total_energy_J * op.stats.count for op in ops])
    energy_stats["sa_energy_J"] = sum([(op.stats.static_energy_sa_J + op.stats.dynamic_energy_sa_J) * op.stats.count for op in ops])
    energy_stats["vu_energy_J"] = sum([(op.stats.static_energy_vu_J + op.stats.dynamic_energy_vu_J) * op.stats.count for op in ops])
    energy_stats["sram_energy_J"] = sum([(op.stats.static_energy_sram_J + op.stats.dynamic_energy_sram_J) * op.stats.count for op in ops])
    energy_stats["ici_energy_J"] = sum([(op.stats.static_energy_ici_J + op.stats.dynamic_energy_ici_J) * op.stats.count for op in ops])
    energy_stats["hbm_energy_J"] = sum([(op.stats.static_energy_hbm_J + op.stats.dynamic_energy_hbm_J) * op.stats.count for op in ops])
    energy_stats["other_energy_J"] = sum([(op.stats.static_energy_other_J + op.stats.dynamic_energy_other_J) * op.stats.count for op in ops])

    energy_stats["total_static_energy_J"] = sum([op.stats.static_energy_J * op.stats.count for op in ops])
    energy_stats["static_sa_energy_J"] = sum([op.stats.static_energy_sa_J * op.stats.count for op in ops])
    energy_stats["static_vu_energy_J"] = sum([op.stats.static_energy_vu_J * op.stats.count for op in ops])
    energy_stats["static_sram_energy_J"] = sum([op.stats.static_energy_sram_J * op.stats.count for op in ops])
    energy_stats["static_ici_energy_J"] = sum([op.stats.static_energy_ici_J * op.stats.count for op in ops])
    energy_stats["static_hbm_energy_J"] = sum([op.stats.static_energy_hbm_J * op.stats.count for op in ops])
    energy_stats["static_other_energy_J"] = sum([op.stats.static_energy_other_J * op.stats.count for op in ops])

    energy_stats["total_dynamic_energy_J"] = sum([op.stats.dynamic_energy_J * op.stats.count for op in ops])
    energy_stats["dynamic_sa_energy_J"] = sum([op.stats.dynamic_energy_sa_J * op.stats.count for op in ops])
    energy_stats["dynamic_vu_energy_J"] = sum([op.stats.dynamic_energy_vu_J * op.stats.count for op in ops])
    energy_stats["dynamic_sram_energy_J"] = sum([op.stats.dynamic_energy_sram_J * op.stats.count for op in ops])
    energy_stats["dynamic_ici_energy_J"] = sum([op.stats.dynamic_energy_ici_J * op.stats.count for op in ops])
    energy_stats["dynamic_hbm_energy_J"] = sum([op.stats.dynamic_energy_hbm_J * op.stats.count for op in ops])
    energy_stats["dynamic_other_energy_J"] = sum([op.stats.dynamic_energy_other_J * op.stats.count for op in ops])

    total_exe_time_s = sum([op.stats.execution_time_ns * op.stats.count for op in ops]) / 1e9
    energy_stats["peak_power_W"] = max([op.stats.total_power_W for op in ops])
    energy_stats["avg_power_W"] = energy_stats["total_energy_J"] / total_exe_time_s

    energy_stats["avg_static_power_W"] = energy_stats["total_static_energy_J"] / total_exe_time_s
    energy_stats["avg_static_sa_power_W"] = energy_stats["static_sa_energy_J"] / total_exe_time_s
    energy_stats["avg_static_vu_power_W"] = energy_stats["static_vu_energy_J"] / total_exe_time_s
    energy_stats["avg_static_sram_power_W"] = energy_stats["static_sram_energy_J"] / total_exe_time_s
    energy_stats["avg_static_ici_power_W"] = energy_stats["static_ici_energy_J"] / total_exe_time_s
    energy_stats["avg_static_hbm_power_W"] = energy_stats["static_hbm_energy_J"] / total_exe_time_s
    energy_stats["avg_static_other_power_W"] = energy_stats["static_other_energy_J"] / total_exe_time_s

    energy_stats["avg_dynamic_power_W"] = energy_stats["total_dynamic_energy_J"] / total_exe_time_s
    energy_stats["avg_dynamic_sa_power_W"] = energy_stats["dynamic_sa_energy_J"] / total_exe_time_s
    energy_stats["avg_dynamic_vu_power_W"] = energy_stats["dynamic_vu_energy_J"] / total_exe_time_s
    energy_stats["avg_dynamic_sram_power_W"] = energy_stats["dynamic_sram_energy_J"] / total_exe_time_s
    energy_stats["avg_dynamic_ici_power_W"] = energy_stats["dynamic_ici_energy_J"] / total_exe_time_s
    energy_stats["avg_dynamic_hbm_power_W"] = energy_stats["dynamic_hbm_energy_J"] / total_exe_time_s
    energy_stats["avg_dynamic_other_power_W"] = energy_stats["dynamic_other_energy_J"] / total_exe_time_s


def analyze_operator_component_time(
    stats: dict[str, Any],
    ops: list[Operator.Operator],
    config: ModelConfig,
    component_stats: dict[str, Any],
):
    total_exe_time_ns = sum([op.stats.execution_time_ns * op.stats.count for op in ops])
    total_flops = sum([
        op.stats.flop_count * op.stats.count for op in ops
        if op.stats.sa_time_ns > 0
    ])
    total_sa_time_ns = sum(
        [op.stats.sa_time_ns * op.stats.count for op in ops]
    )

    def sa_freq_GHz(op: Operator.Operator) -> float:
        return op.dvfs_sa.frequency_GHz or config.freq_GHz
    def vu_freq_GHz(op: Operator.Operator) -> float:
        return op.dvfs_vu.frequency_GHz or config.freq_GHz

    peak_flops = sum([
        op.stats.sa_time_ns * sa_freq_GHz(op) * config.num_sa * (config.sa_dim ** 2) * 2 * op.stats.count
        + op.stats.vu_time_ns * vu_freq_GHz(op) * config.num_vu * (8 * 128) * op.stats.count
        for op in ops
    ])

    component_stats["total_exe_time_ns"] = total_exe_time_ns
    component_stats["sa_time_ns"] = total_sa_time_ns
    component_stats["vu_time_ns"] = sum([op.stats.vu_time_ns * op.stats.count for op in ops])
    component_stats["vmem_time_ns"] = sum([op.stats.vmem_time_ns * op.stats.count for op in ops])
    component_stats["hbm_time_ns"] = sum([op.stats.memory_time_ns * op.stats.count for op in ops])
    component_stats["ici_time_ns"] = sum([op.stats.ici_time_ns * op.stats.count for op in ops])

    component_stats["sa_temp_util"] = component_stats["sa_time_ns"] / total_exe_time_ns
    component_stats["vu_temp_util"] = component_stats["vu_time_ns"] / total_exe_time_ns
    component_stats["vmem_temp_util"] = component_stats["vmem_time_ns"] / total_exe_time_ns
    component_stats["hbm_temp_util"] = component_stats["hbm_time_ns"] / total_exe_time_ns
    component_stats["ici_temp_util"] = component_stats["ici_time_ns"] / total_exe_time_ns
    if component_stats["sa_time_ns"] > 0:
        assert abs(total_flops / peak_flops) <= 1 + 1e-3, \
            f"Total SA flops {total_flops} exceeds peak flops {peak_flops}:" \
            f"stats: {stats}"
        component_stats["sa_spatial_util"] = total_flops / peak_flops
    else:
        component_stats["sa_spatial_util"] = 0

    component_stats["flop_count"] = sum([
        op.stats.flop_count * op.stats.count for op in ops
    ])
    component_stats["hbm_bytes_accessed"] = sum([
        op.stats.memory_traffic_bytes * op.stats.count for op in ops
    ])
    component_stats["ici_bytes_accessed"] = sum([
        op.stats.ici_traffic_bytes * op.stats.count for op in ops
    ])
    component_stats["tflops_per_sec"] = (component_stats["flop_count"] / 1e12) / (total_exe_time_ns / 1e9)
    component_stats["flop_per_byte_hbm"] = component_stats["flop_count"] / component_stats["hbm_bytes_accessed"]
    component_stats["flop_per_byte_ici"] = component_stats["flop_count"] / component_stats["ici_bytes_accessed"] if component_stats["ici_bytes_accessed"] > 0 else 0

    component_stats["model_flops_util"] = sum([
        op.stats.flops_util * (op.stats.execution_time_ns * op.stats.count / total_exe_time_ns)
        for op in ops
    ])
    component_stats["hbm_bw_util"] = sum([
        op.stats.hbm_bw_util * (op.stats.execution_time_ns * op.stats.count / total_exe_time_ns)
        for op in ops
    ])

    component_stats["avg_einsum_B"] = sum([
        op.stats.einsum_B_size * op.stats.sa_time_ns * op.stats.count for op in ops
        if isinstance(op.stats, (Operator.EinsumStatistics, Operator.FlashAttentionStatistics))
    ]) / total_sa_time_ns if total_sa_time_ns > 0 else 0
    component_stats["avg_einsum_K"] = float(np.mean([
        op.stats.einsum_K_size * op.stats.sa_time_ns * op.stats.count for op in ops
        if isinstance(op.stats, (Operator.EinsumStatistics, Operator.FlashAttentionStatistics))
    ])) / total_sa_time_ns if total_sa_time_ns > 0 else 0
    component_stats["avg_einsum_N"] = float(np.mean([
        op.stats.einsum_N_size * op.stats.sa_time_ns * op.stats.count for op in ops
        if isinstance(op.stats, (Operator.EinsumStatistics, Operator.FlashAttentionStatistics))
    ])) / total_sa_time_ns if total_sa_time_ns > 0 else 0
    component_stats["avg_einsum_M"] = float(np.mean([
        op.stats.einsum_M_size * op.stats.sa_time_ns * op.stats.count for op in ops
        if isinstance(op.stats, (Operator.EinsumStatistics, Operator.FlashAttentionStatistics))
    ])) / total_sa_time_ns if total_sa_time_ns > 0 else 0

    component_stats["avg_vmem_demand_MB"] = float(np.mean([
        (
            op.stats.max_vmem_demand_bytes / 1024 / 1024
            if isinstance(op.stats, (Operator.EinsumStatistics, Operator.FlashAttentionStatistics))
            else
            4
        ) for op in ops
    ]))

     # === Aggregate DVFS slack across ops (sum of per-op slack) ===
    total_extra = { "sa": 0, "vu": 0, "hbm": 0, "ici": 0 ,"vmem": 0}
    for op in ops:
        extras, _ = compute_component_slack_for_op(op)
        for k in total_extra.keys():
            total_extra[k] += extras[k] * op.stats.count

    component_stats["dvfs_extra_sa_time_ns"] = total_extra["sa"]
    component_stats["dvfs_extra_vu_time_ns"] = total_extra["vu"]
    component_stats["dvfs_extra_hbm_time_ns"] = total_extra["hbm"]
    component_stats["dvfs_extra_ici_time_ns"] = total_extra["ici"]
    component_stats["dvfs_extra_vmem_time_ns"] = total_extra["vmem"]

    def _safe_ratio(extra_ns: int, cur_ns: int) -> float:
        return float(extra_ns / cur_ns) if cur_ns > 0 else 0.0

    component_stats["dvfs_extra_sa_ratio"] = _safe_ratio(total_extra["sa"], component_stats["sa_time_ns"])
    component_stats["dvfs_extra_vu_ratio"] = _safe_ratio(total_extra["vu"], component_stats["vu_time_ns"])
    component_stats["dvfs_extra_hbm_ratio"] = _safe_ratio(total_extra["hbm"], component_stats["hbm_time_ns"])
    component_stats["dvfs_extra_ici_ratio"] = _safe_ratio(total_extra["ici"], component_stats["ici_time_ns"])
    component_stats["dvfs_extra_vmem_ratio"] = _safe_ratio(total_extra["vmem"], component_stats["vmem_time_ns"])


@ray.remote
def analyze_operator_energy_for_stats(
    stats: dict[str, Any],
    opdicts: list[dict[str, Any]],
    power_gating_strategies: Sequence[str],
    dvfs_policy_str: str,
    dump_to_file_fn: Callable[[dict[str, Any], list[dict[str, Any]], str], None] | None = None,
):
    # read power config
    config = stats["sim_config"]
    config = ModelConfig.model_validate(config)

    # convert ops from dict to Operator
    orig_ops = [Operator.from_csv_dict(opdict) for opdict in opdicts]
    # analyze energy for each power-gating strategy
    stats["energy_stats"] = {}
    for pg_strategy in power_gating_strategies:
        ops = deepcopy(orig_ops)

        energy_stats = {}
        analyze_operator_energy(
            ops, config, pg_strategy, energy_stats, dvfs_policy_str
        )
        stats["energy_stats"][pg_strategy] = energy_stats

        # analyze component breakdown
        component_stats = {}
        analyze_operator_component_time(
            stats, ops, config, component_stats
        )
        stats["energy_stats"][pg_strategy]["component_stats"] = component_stats
        stats["energy_stats"][pg_strategy]["component_stats"]["slowdown"] = (
            stats["energy_stats"][pg_strategy]["component_stats"]["total_exe_time_ns"]
            / stats["energy_stats"]["NoPG"]["component_stats"]["total_exe_time_ns"]
        )
        num_setpm_vu = sum([
            op.stats.num_setpm_vu * op.stats.count for op in ops
        ])
        stats["energy_stats"][pg_strategy]["component_stats"]["num_setpm_vu"] = num_setpm_vu
        num_setpm_sram = sum([
            op.stats.num_setpm_sram * op.stats.count for op in ops
        ])
        stats["energy_stats"][pg_strategy]["component_stats"]["num_setpm_sram"] = num_setpm_sram


        # output_opdicts = [op.to_csv_dict() for op in ops]
        # if dump_to_file_fn is not None:
        #     dump_to_file_fn(stats, output_opdicts, pg_strategy)
         # === enrich per-op CSV with DVFS slack columns ===
        output_opdicts = []
        for op in ops:
            row = op.to_csv_dict()
            extras, ratios = compute_component_slack_for_op(op)
            row["dvfs_extra_sa_time_ns"] = int(extras["sa"])
            row["dvfs_extra_vu_time_ns"] = int(extras["vu"])
            row["dvfs_extra_hbm_time_ns"] = int(extras["hbm"])
            row["dvfs_extra_ici_time_ns"] = int(extras["ici"])
            row["dvfs_extra_vmem_time_ns"] = int(extras["vmem"])

            row["dvfs_extra_sa_ratio"] = float(ratios["sa"])
            row["dvfs_extra_vu_ratio"] = float(ratios["vu"])
            row["dvfs_extra_hbm_ratio"] = float(ratios["hbm"])
            row["dvfs_extra_ici_ratio"] = float(ratios["ici"])
            row["dvfs_extra_vmem_ratio"] = float(ratios["vmem"])
            output_opdicts.append(row)

        if dump_to_file_fn is not None:
            dump_to_file_fn(stats, output_opdicts, pg_strategy)

    logging.info("Finished analyzing energy for stats with output file %s", stats["sim_config"]["output_file_path"])


@ray.remote
def analyze_all_raw_results(
    model: str,
    version: str,
    workload: str,
    prefill_or_decode: str,
    power_gating_strategies: Sequence[str],
    dvfs_policy_str: str,
):
    global RESULTS_PATH

    assert RESULTS_PATH
    results_lib.set_results_path(
        os.path.join(RESULTS_PATH)
    )
    results_folder_name = RESULTS_PATH.split("/")[-1]

    all_stats = results_lib.get_all_op_stats(
        model, version, workload, prefill_or_decode, read_json_with_csv=True,
        results_path=RESULTS_PATH
    )

    # define output file function
    def dump_to_file_fn(stats: dict[str, Any], opdicts: list[dict[str, Any]], pg_strategy: str):
        stats = deepcopy(stats)
        outfile = stats["sim_config"]["output_file_path"]
        outfile = outfile.replace(f"{results_folder_name}/", f"{results_folder_name}_{dvfs_policy_str}/")
        json_out_file = outfile.replace(".csv", ".json")
        outfile = outfile.replace(".csv", f"_{pg_strategy}.csv")
        outpath = os.path.dirname(outfile)
        stats["sim_config"]["output_file_path"] = outfile
        stats["sim_config"]["out_stats_file_path"] = json_out_file
        os.makedirs(outpath, exist_ok=True)
        with open(json_out_file, "w") as f:
            json.dump(stats, f, indent=4)
        with open(outfile, "w") as f:
            # write opdicts as csv
            writer = csv.DictWriter(f, fieldnames=opdicts[0].keys())
            writer.writeheader()
            writer.writerows(opdicts)
    if results_lib.is_model_llm(model) and workload == "inference":
        def dump_to_file_fn(stats: dict[str, Any], opdicts: list[dict[str, Any]], pg_strategy: str):
            stats = deepcopy(stats)
            outfile = stats["sim_config"]["output_file_path"].replace(".csv", f"_{prefill_or_decode}.csv")
            outfile = outfile.replace(f"{results_folder_name}/", f"{results_folder_name}_{dvfs_policy_str}/")
            json_out_file = outfile.replace(".csv", ".json")
            outfile = outfile.replace(".csv", f"_{pg_strategy}.csv")
            outpath = os.path.dirname(outfile)
            stats["sim_config"]["output_file_path"] = outfile
            stats["sim_config"]["out_stats_file_path"] = json_out_file
            os.makedirs(outpath, exist_ok=True)
            with open(json_out_file, "w") as f:
                json.dump(stats, f, indent=4)
            with open(outfile, "w") as f:
                # write opdicts as csv
                writer = csv.DictWriter(f, fieldnames=opdicts[0].keys())
                writer.writeheader()
                writer.writerows(opdicts)

    params = [
        (stat[0], stat[1], power_gating_strategies, dvfs_policy_str, dump_to_file_fn)
        for stat in all_stats.values()
    ]
    global DEBUG_MODE
    if DEBUG_MODE:
        for p in params:
            analyze_operator_energy_for_stats.remote(*p)  # type: ignore
    else:
        # progress_starmap(
        #     analyze_operator_energy_for_stats,
        #     params,
        # )
        futures = [
            analyze_operator_energy_for_stats.remote(*p) for p in params  # type: ignore
        ]
        ray.get(futures)


def main(argv: list[str]):
    del argv  # Unused

    global RESULTS_PATH
    RESULTS_PATH = __RESULTS_PATH.value

    global DEBUG_MODE
    DEBUG_MODE = __DEBUG.value

    global DUMP_INTERMEDIATE_ANALYSIS_RESULTS
    DUMP_INTERMEDIATE_ANALYSIS_RESULTS = __DUMP_INTERMEDIATE_ANALYSIS_RESULTS.value

    workload = __WORKLOAD.value
    dvfs_policy_str = __DVFS_POLICY.value
    # analyze all raw results
    futures = []
    for model in __MODELS.value:
        for v in __NPU_VERSIONS.value:
            # logging.info("Analyzing %s v%s prefill", model, v)
            # analyze_all_raw_results(
            #     model, v, workload, "prefill", __POWER_GATING_STRATEGY.value
            # )
            # logging.info("Analyzing %s v%s decode", model, v)
            # if "llama" in model and workload == "inference":
            #     analyze_all_raw_results(
            #         model, v, workload, "decode", __POWER_GATING_STRATEGY.value
            #     )
            if results_lib.is_model_llm(model) and __WORKLOAD.value == "inference":
                logging.info("Analyzing %s v%s prefill", model, v)
            else:
                logging.info("Analyzing %s v%s", model, v)
            futures.append(
                analyze_all_raw_results.remote(
                    model, v, workload, "prefill", __POWER_GATING_STRATEGY.value, dvfs_policy_str
                )
            )
            if results_lib.is_model_llm(model) and workload == "inference":
                logging.info("Analyzing %s v%s decode", model, v)
                futures.append(
                    analyze_all_raw_results.remote(
                        model, v, workload, "decode", __POWER_GATING_STRATEGY.value, dvfs_policy_str
                    )
                )

    ray.get(futures)


if __name__ == "__main__":
    app.run(main)
