"""CLI for heterogeneous NPU fleet simulation with NeuSim as the backend."""

import json
import math
import os
from pathlib import Path

from absl import app, flags, logging

from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig
from neusim.configs.systems.NPUFleetConfig import NPUFleetConfig
from neusim.fleetsim import ideal_baseline as ideal_sim
from neusim.fleetsim import npusim_backend_interface as npusim_backend
from neusim.fleetsim.NPUFleetSimulator import NPUFleetSimulator
from neusim.npusim.frontend.query_results_helper_lib import is_model_llm_moe

NEUSIM_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = NEUSIM_PACKAGE_ROOT.parent
DEFAULT_CONFIGS_DIR = Path(
    os.environ.get("NEUSIM_CONFIGS_DIR") or REPO_ROOT / "configs"
).expanduser()
DEFAULT_RESULTS_DIR = Path(
    os.environ.get("NEUSIM_RESULTS_DIR", Path.cwd() / "results" / "fleetsim")
)
DEFAULT_TRACES_DIR = Path(
    os.environ.get("NEUSIM_TRACES_DIR", Path.cwd() / "traces" / "inference")
)
TEST_TRACES_DIR = REPO_ROOT / "neusim" / "fleetsim" / "tests" / "data" / "traces"


__MODEL = flags.DEFINE_string("model", "llama3-8b", "Model name")
__SYSTEM = flags.DEFINE_string(
    "system", "Base", "System under test. Choose from [Base, Ideal, NeuScale]"
)
__TRACE = flags.DEFINE_string(
    "trace",
    "Azure-Conv",
    "Trace name. Choose from [Azure-Conv, Azure-Code, Azure2Hr, BurstGPT-Conv, BurstGPT-API].",
)
__TRACE_FILE = flags.DEFINE_string(
    "trace_file",
    None,
    "Explicit request-trace CSV. Overrides --trace when request_pattern=trace.",
)
__TRACES_DIR = flags.DEFINE_string(
    "traces_dir",
    str(DEFAULT_TRACES_DIR),
    "Root of the external AzurePublicDataset/ and BurstGPT/ trace trees.",
)
__OPT_GOAL = flags.DEFINE_string(
    "opt_goal", "energy", "Optimization goal. Choose from [energy, monetary]."
)
__CHIP_VERSIONS = flags.DEFINE_list(
    "chip_versions",
    ["5p", "6e"],
    "Chip versions to use.",
)
__ALLOCATION_SUCCESS_RATE = flags.DEFINE_float(
    "allocation_success_rate", 1.0, "Allocation success rate."
)
__HS_INTERVAL_MINUTES = flags.DEFINE_float(
    "hs_interval_minutes", 30.0, "Horizontal scaling interval in minutes."
)
__VS_INTERVAL_MINUTES = flags.DEFINE_float(
    "vs_interval_minutes", 10.0, "Vertical scaling interval in minutes."
)
__HS_WINDOW_MINUTES = flags.DEFINE_float(
    "hs_window_minutes", 30.0, "Horizontal scaling window in minutes."
)
__VS_WINDOW_MINUTES = flags.DEFINE_float(
    "vs_window_minutes", 30.0, "Vertical scaling window in minutes."
)
__CONFIGS_PATH = flags.DEFINE_string(
    "configs_path", str(DEFAULT_CONFIGS_DIR), "Path to the NeuSim configs directory."
)
__OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", str(DEFAULT_RESULTS_DIR), "Path to the output directory."
)
__MAX_TIMESTAMP_HOURS = flags.DEFINE_float(
    "max_timestamp_hours", -1.0, "Maximum simulation timestamp in hours."
)
__MAX_NUM_REQUESTS = flags.DEFINE_integer(
    "max_num_requests", -1, "Maximum number of requests to simulate."
)
__NPUSIM_BACKEND_CACHE_DIR = flags.DEFINE_string(
    "npusim_backend_cache_dir",
    str(DEFAULT_RESULTS_DIR / ".cache" / "npusim_backend"),
    "Path to the NeuSim backend cache directory.",
)
__NPUSIM_BACKEND_CACHE_USE_MMAP = flags.DEFINE_bool(
    "npusim_backend_cache_use_mmap",
    False,
    "Whether to use memory-mapped files for the npusim backend cache. This cannot be used when sharing the cache directory across multiple machines.",
)
CONFIGS_PATH: str | None = None
__EXPERT_LOAD_IMBALANCE_FACTOR = flags.DEFINE_float(
    "expert_load_imbalance_factor",
    -1.0,
    "Expert load imbalance factor for MoE models. "
    "1.0 = balanced, E/K = worst case, -1.0 = auto (worst case). Ignored for non-MoE models.",
)
__REQUEST_RESULTS_CACHE_DIR = flags.DEFINE_string(
    "request_results_cache_dir",
    None,
    "Directory containing pre-computed optimal configs per model/seqlen/version. "
    "Overrides the default in LLMInferenceWorkloadConfig if set.",
)
__TQDM = flags.DEFINE_bool("tqdm", False, "Show tqdm progress bar.")
__VALIDATE_ONLY = flags.DEFINE_bool(
    "validate_only",
    False,
    "Build and validate the FleetSim configuration, then exit without simulating.",
)
__N_CPU = flags.DEFINE_integer(
    "n_cpu",
    None,
    "Number of CPU workers for parallel simulation. Defaults to os.cpu_count().",
)
__REQUEST_RATE = flags.DEFINE_float(
    "request_rate",
    1.0,
    "Multiply the trace's request rate by this integer factor. Each request is replicated "
    "and the copies are spread evenly across its inter-arrival gap, so the arrival rate is "
    "scaled at every instant while the trace's temporal pattern (ramp/shape) is preserved.",
)
__NUM_POOLS = flags.DEFINE_integer(
    "num_pools", 3, "Number of pools for MultiPoolAutoScaler."
)
__MAX_CHIPS_PER_VERSION = flags.DEFINE_string(
    "max_chips_per_version",
    None,
    "Max chips per NPU version, format: 'version1=count1,version2=count2' "
    "(e.g., '5p=1024,6e=2048'). If not set, no limit is enforced.",
)
__STATIC_VPOD_ALLOCATION = flags.DEFINE_string(
    "static_vpod_allocation",
    None,
    "Path to JSON file specifying static vPod allocation. "
    "Required when --system=Static. Format: {prefill: {...}, decode: {...}}",
)
__PREFILL_CHIP_VERSIONS = flags.DEFINE_list(
    "prefill_chip_versions",
    None,
    "Override chip versions for prefill. If unset, uses --chip_versions.",
)
__DECODE_CHIP_VERSIONS = flags.DEFINE_list(
    "decode_chip_versions",
    None,
    "Override chip versions for decode. If unset, uses --chip_versions.",
)
__REQUEST_PATTERN = flags.DEFINE_string(
    "request_pattern",
    "trace",
    "Request pattern type. Choose from [trace, synthetic].",
)
__SYNTHETIC_NUM_REQUESTS = flags.DEFINE_integer(
    "synthetic_num_requests", 2000, "Number of synthetic requests to generate."
)
__SYNTHETIC_REQUEST_RATE = flags.DEFINE_float(
    "synthetic_request_rate",
    10.0,
    "Synthetic request rate in RPS. 0 = infinity (all at t=0).",
)
__SYNTHETIC_INPUT_LEN = flags.DEFINE_integer(
    "synthetic_input_len", 512, "Mean input sequence length for synthetic requests."
)
__SYNTHETIC_INPUT_LEN_STD = flags.DEFINE_integer(
    "synthetic_input_len_std", 0, "Std dev of input sequence length. 0 = fixed."
)
__SYNTHETIC_OUTPUT_LEN = flags.DEFINE_integer(
    "synthetic_output_len", 128, "Mean output sequence length for synthetic requests."
)
__SYNTHETIC_OUTPUT_LEN_STD = flags.DEFINE_integer(
    "synthetic_output_len_std", 0, "Std dev of output sequence length. 0 = fixed."
)
__SYNTHETIC_SEED = flags.DEFINE_integer(
    "synthetic_seed", 42, "Random seed for synthetic request generation."
)
__EWMA_ALPHA = flags.DEFINE_float(
    "ewma_alpha", 0.6, "EWMA smoothing factor for peak request rate."
)
__EWMA_INTERVAL_SECONDS = flags.DEFINE_float(
    "ewma_interval_seconds",
    10.0,
    "Duration (seconds) of each time bin for computing per-interval request counts.",
)
__SCALING_HEADROOM_FACTOR = flags.DEFINE_float(
    "scaling_headroom_factor",
    1.1,
    "Multiplicative headroom on EWMA peak rate. 1.0 = exact fit.",
)
__QUEUE_DRAIN_TARGET_SECONDS = flags.DEFINE_float(
    "queue_drain_target_seconds",
    60.0,
    "Target time (s) to drain the request queue; the autoscaler adds vPods so a backlog "
    "clears within this window. Set small (~1s, SLO-aware) so queueing that threatens the "
    "latency SLO triggers immediate scale-up. 0 disables the queue-drain scaling signal.",
)
__COALESCE_NL_THRESHOLD = flags.DEFINE_float(
    "coalesce_nl_threshold",
    0.5,
    "NeuScale vPod-group coalescing threshold on normalized peak throughput "
    "N_L = R_peak / T_pod. Groups below this are merged into the next-larger-seqlen group "
    "when it has spare capacity. 0 disables coalescing.",
)
__DECODE_BATCH_SEQLEN_RATIO_THRESHOLD = flags.DEFINE_float(
    "decode_batch_seqlen_ratio_threshold",
    2.0,
    "Decode batching coherence: two requests batch together only if their total_seqlen "
    "ratio is within this (applies when the longer seq >= decode_batch_seqlen_min_threshold).",
)
__DECODE_BATCH_SEQLEN_MIN_THRESHOLD = flags.DEFINE_integer(
    "decode_batch_seqlen_min_threshold",
    256,
    "Decode batching length floor: below this seqlen, requests batch regardless of ratio.",
)
__DECODE_POOL_SINGLE_CONFIG = flags.DEFINE_bool(
    "decode_pool_single_config",
    False,
    "NeuScale decode-pooling: collapse all decode groups into one largest-seqlen umbrella "
    "config (re-sized from total decode load) so the single deep queue amortizes batch like "
    "the Base-Max baseline. Decode only; prefill stays fragmented. No effect on Base-Max.",
)
__OUTPUT_PREDICTION_ACCURACY = flags.DEFINE_float(
    "output_prediction_accuracy",
    1.0,
    "Output-length prediction accuracy for NeuScale decode routing. A correct "
    "prediction routes to the best-fit vPod group; a misprediction routes "
    "deterministically to another deployed group that can hold the request's "
    "current sequence length.",
)
__OUTPUT_PREDICTION_SEED = flags.DEFINE_integer(
    "output_prediction_seed",
    42,
    "Seed for deterministic output-length prediction outcomes and wrong-group choices.",
)
__ENABLE_PROFILE = flags.DEFINE_bool(
    "enable_profile", False, "Enable per-event-type profiling in the event simulator."
)


def parse_max_chips(s: str) -> dict[str, int]:
    """Parse 'version1=count1,version2=count2' into {version: count} dict."""
    result: dict[str, int] = {}
    for pair in s.split(","):
        try:
            key, raw_value = pair.strip().split("=", maxsplit=1)
            value = int(raw_value.strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid max-chip entry {pair!r}; expected VERSION=NONNEGATIVE_INT."
            ) from exc
        key = key.strip()
        if not key or value < 0:
            raise ValueError(
                f"Invalid max-chip entry {pair!r}; expected VERSION=NONNEGATIVE_INT."
            )
        if key in result:
            raise ValueError(f"Duplicate max-chip entry for NPU version {key!r}.")
        result[key] = value
    return result


def get_autoscaler_type(sysname: str) -> str:
    if sysname in ["Base", "Base-Avg", "Base-Max"]:
        return "HorizontalAutoScaler"
    elif sysname == "Ideal":
        return "IdealAutoScaler"
    elif sysname == "NeuScale":
        return "NeuScaleAutoScaler"
    elif sysname == "MultiPool":
        return "MultiPoolAutoScaler"
    elif sysname == "Static":
        return "StaticAutoScaler"
    else:
        raise ValueError(f"Unknown system name: {sysname}")


def get_trace_filename(
    trace_name: str, traces_dir: str | os.PathLike[str] | None = None
) -> str:
    """Resolve a trace alias, while also accepting an explicit CSV path."""
    explicit_path = Path(trace_name).expanduser()
    if explicit_path.is_file():
        return str(explicit_path.resolve())

    bundled_traces = {
        "Azure-test": TEST_TRACES_DIR / "AzureLLMInferenceTrace_code_test.csv",
        "Batch-test": TEST_TRACES_DIR / "batch_4096_512.csv",
        "BurstGPT-test": TEST_TRACES_DIR / "BurstGPT_test.csv",
    }
    if trace_name in bundled_traces:
        return str(bundled_traces[trace_name])

    relative_traces = {
        "Azure-Conv": "AzurePublicDataset/data/AzureLLMInferenceTrace_conv_1week_sampled.csv",
        "Azure-Code": "AzurePublicDataset/data/AzureLLMInferenceTrace_code_1week_sampled.csv",
        "Azure-Code-1day": "AzurePublicDataset/data/AzureLLMInferenceTrace_code_1day.csv",
        "Azure2Hr": "AzurePublicDataset/data/AzureLLMInferenceTrace_code_14to16hr.csv",
        "Azure4Hr": "AzurePublicDataset/data/AzureLLMInferenceTrace_code_16to20hr.csv",
        "BurstGPT-Conv": "BurstGPT/data/BurstGPT_without_fails_1_conv_sampled.csv",
        "BurstGPT-API": "BurstGPT/data/BurstGPT_without_fails_1_api_sampled.csv",
        "BurstGPT": "BurstGPT/data/BurstGPT_without_fails_1_sampled.csv",
        "BurstGPT-LVEval": "BurstGPT/data/BurstGPT_LVEval_traces.csv",
        "BurstGPT-OpenThoughts": "BurstGPT/data/BurstGPT_OpenThoughts_traces.csv",
        "Azure-LVEval": "BurstGPT/data/Azure_LVEval_traces.csv",
        "Azure-LVEval-10K": "BurstGPT/data/Azure_LVEval_10K_traces.csv",
        "Azure-LVEval-test": "BurstGPT/data/Azure_LVEval_traces_test.csv",
        "Azure-OpenThoughts": "BurstGPT/data/Azure_OpenThoughts_traces.csv",
        "Azure-OpenThoughts-test": "BurstGPT/data/Azure_OpenThoughts_traces_test.csv",
        "SynBurst": "BurstGPT/data/synthetic_BurstGPT_trace.csv",
    }
    try:
        relative_path = relative_traces[trace_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown trace name {trace_name!r}; pass a CSV path via --trace_file "
            "or configure the external trace root with --traces_dir."
        ) from exc
    return str(Path(traces_dir or DEFAULT_TRACES_DIR) / relative_path)


def get_max_decode_batch_size(model: str, trace: str) -> int:
    # model_name: str -> trace_name -> max_decode_batch_size: int
    MAX_DECODE_BATCH_SIZE_MAP = {
        "llama3-70b": {
            "Azure-Code": 2,
            "Azure-LVEval-10K": 4,
            "Azure-LVEval": 4,
            "Azure-OpenThoughts": 2,
        },
        "llama3_1-405b": {
            "Azure-Code": 2,
            "Azure-LVEval-10K": 4,
            "Azure-LVEval": 4,
            "Azure-OpenThoughts": 2,
        },
        "deepseekv2-236b": {
            "Azure-Code": 2,
            "Azure-LVEval-10K": -1,
            "Azure-LVEval": -1,
            "Azure-OpenThoughts": 2,
        },
        "deepseekv3-671b": {
            "Azure-Code": 2,
            "Azure-LVEval-10K": -1,
            "Azure-LVEval": -1,
            "Azure-OpenThoughts": 2,
        },
    }
    if model in MAX_DECODE_BATCH_SIZE_MAP and trace in MAX_DECODE_BATCH_SIZE_MAP[model]:
        return MAX_DECODE_BATCH_SIZE_MAP[model][trace]
    return -1


def get_base_config(model: str):
    global CONFIGS_PATH
    assert CONFIGS_PATH

    p_cfg = {
        "num_chips": 64,
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": 8,
        "pipeline_parallelism_degree": 8,
        "data_parallel_degree_dcn": 1,
        "pipeline_parallel_degree_dcn": 1,
    }
    pp = p_cfg["pipeline_parallelism_degree"]

    v = "5p"
    global_batch_size = 64

    # read base model and NPU configs
    config_root = Path(CONFIGS_PATH)
    with (config_root / "models" / f"{model}.json").open() as config_file:
        base_model_config = json.load(config_file)
    with (config_root / "chips" / f"tpuv{v}.json").open() as config_file:
        base_npu_config = json.load(config_file)
    with (config_root / "systems" / "system_config.json").open() as config_file:
        base_sys_config = json.load(config_file)

    # create config for the ops generator
    base_config = {**base_model_config, **base_npu_config, **base_sys_config}
    base_config.update(p_cfg)

    ## Determine microbatch sizes for ICI and DCN. Determine # of NPU pods and batch size per pod.
    microbatch_size_ici = math.ceil(global_batch_size / pp)
    microbatch_size_dcn = global_batch_size

    base_config["global_batch_size"] = global_batch_size
    base_config["microbatch_size_ici"] = microbatch_size_ici
    base_config["microbatch_size_dcn"] = microbatch_size_dcn

    base_config["num_data_parallel_axes"] = 0
    base_config["num_tensor_parallel_axes"] = 2
    base_config["num_pipeline_parallel_axes"] = 1
    if "deepseek" in model.lower():
        base_config["num_expert_parallel_axes"] = 1

    if is_model_llm_moe(model):
        base_config[
            "expert_load_imbalance_factor"
        ] = __EXPERT_LOAD_IMBALANCE_FACTOR.value

    cluster_config = {
        "npu_types": __CHIP_VERSIONS.value,
        "satisfaction_probability": [__ALLOCATION_SUCCESS_RATE.value],
        "chip_config_path": str(config_root / "chips"),
    }
    if __PREFILL_CHIP_VERSIONS.value:
        cluster_config["prefill_npu_types"] = __PREFILL_CHIP_VERSIONS.value
    if __DECODE_CHIP_VERSIONS.value:
        cluster_config["decode_npu_types"] = __DECODE_CHIP_VERSIONS.value
    # Ensure npu_types is the union so the cluster manager loads all needed chip configs
    all_types = set(__CHIP_VERSIONS.value)
    if __PREFILL_CHIP_VERSIONS.value:
        all_types.update(__PREFILL_CHIP_VERSIONS.value)
    if __DECODE_CHIP_VERSIONS.value:
        all_types.update(__DECODE_CHIP_VERSIONS.value)
    cluster_config["npu_types"] = sorted(all_types)
    if __MAX_CHIPS_PER_VERSION.value:
        cluster_config["max_chips_per_version"] = parse_max_chips(
            __MAX_CHIPS_PER_VERSION.value
        )

    fleet_config = {
        "cluster_scheduler_config": cluster_config,
        "workload_config": {
            "request_pattern": __REQUEST_PATTERN.value,
            "trace_file_path": (
                __TRACE_FILE.value
                or get_trace_filename(__TRACE.value, __TRACES_DIR.value)
                if __REQUEST_PATTERN.value == "trace"
                else ""
            ),
            "request_rate": __REQUEST_RATE.value,
            "hs_interval_minutes": __HS_INTERVAL_MINUTES.value,
            "vs_interval_minutes": __VS_INTERVAL_MINUTES.value,
            "hs_window_minutes": __HS_WINDOW_MINUTES.value,
            "vs_window_minutes": __VS_WINDOW_MINUTES.value,
            "model_name": model,
            "autoscaler_type": get_autoscaler_type(__SYSTEM.value),
            "hs_initial_alloc_sample_criteria": "max"
            if __SYSTEM.value == "Base-Max"
            else ("average" if __SYSTEM.value == "Base-Avg" else "max"),
            "optimization_goal": __OPT_GOAL.value,
            "max_timestamp": int(__MAX_TIMESTAMP_HOURS.value * 60 * 60 * 1e9)
            if __MAX_TIMESTAMP_HOURS.value > 0
            else -1,
            "max_num_requests": __MAX_NUM_REQUESTS.value
            if __MAX_NUM_REQUESTS.value > 0
            else -1,
            "min_decode_schedule_num_iterations": 4,
            "use_ideal_batch_size": False,
            "max_decode_batch_size": get_max_decode_batch_size(model, __TRACE.value),
            "num_pools": __NUM_POOLS.value,
            "synthetic_num_requests": __SYNTHETIC_NUM_REQUESTS.value,
            "synthetic_request_rate": float("inf")
            if __SYNTHETIC_REQUEST_RATE.value == 0
            else __SYNTHETIC_REQUEST_RATE.value,
            "synthetic_input_len": __SYNTHETIC_INPUT_LEN.value,
            "synthetic_input_len_std": __SYNTHETIC_INPUT_LEN_STD.value,
            "synthetic_output_len": __SYNTHETIC_OUTPUT_LEN.value,
            "synthetic_output_len_std": __SYNTHETIC_OUTPUT_LEN_STD.value,
            "synthetic_seed": __SYNTHETIC_SEED.value,
            "ewma_alpha": __EWMA_ALPHA.value,
            "ewma_interval_seconds": __EWMA_INTERVAL_SECONDS.value,
            "scaling_headroom_factor": __SCALING_HEADROOM_FACTOR.value,
            "queue_drain_target_seconds": __QUEUE_DRAIN_TARGET_SECONDS.value,
            "coalesce_nl_threshold": __COALESCE_NL_THRESHOLD.value,
            "decode_batch_seqlen_ratio_threshold": __DECODE_BATCH_SEQLEN_RATIO_THRESHOLD.value,
            "decode_batch_seqlen_min_threshold": __DECODE_BATCH_SEQLEN_MIN_THRESHOLD.value,
            "decode_pool_single_config": __DECODE_POOL_SINGLE_CONFIG.value,
            "output_prediction_accuracy": __OUTPUT_PREDICTION_ACCURACY.value,
            "output_prediction_seed": __OUTPUT_PREDICTION_SEED.value,
        },
        "tqdm": __TQDM.value,
        "enable_profile": __ENABLE_PROFILE.value,
        "system_config": base_sys_config,
        "npusim_backend_cache_dir": __NPUSIM_BACKEND_CACHE_DIR.value,
        "npusim_backend_cache_use_mmap": __NPUSIM_BACKEND_CACHE_USE_MMAP.value,
        "output_dir": __OUTPUT_DIR.value,
    }

    if "deepseek" in model.lower():
        model_config = DeepSeekConfig.model_validate(base_config)
    else:
        model_config = LLMConfig.model_validate(base_config)

    if __STATIC_VPOD_ALLOCATION.value:
        with open(__STATIC_VPOD_ALLOCATION.value) as f:
            fleet_config["workload_config"]["static_vpod_allocation"] = json.load(f)

    sim_config = NPUFleetConfig.model_validate(fleet_config)
    sim_config.workload_config.llm_config = model_config
    if __REQUEST_RESULTS_CACHE_DIR.value is not None:
        sim_config.workload_config.request_results_cache_dir = (
            __REQUEST_RESULTS_CACHE_DIR.value
        )

    # # only 6 hr for Azure-Conv to reduce sim time
    # if sim_config.workload_config.max_timestamp == -1:
    #     if "Azure-Conv" in __TRACE.value:
    #         sim_config.workload_config.max_timestamp = int(6 * 60 * 60 * 1e9)

    return sim_config


def _initialize_ideal_backend(sim_config: NPUFleetConfig) -> None:
    """Configure the backend cache that Ideal bypasses NPUFleetSimulator to use."""
    cache_dir = Path(sim_config.npusim_backend_cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if sim_config.npusim_backend_cache_use_mmap:
        npusim_backend.set_npusim_backend_cache_dir(cache_dir, mmap_mode="r")
    else:
        npusim_backend.set_npusim_backend_cache_dir(cache_dir)
    npusim_backend.set_enable_profile(sim_config.enable_profile)


def main(argv):
    del argv  # Unused.
    global CONFIGS_PATH

    init_cmd_args()
    if __REQUEST_PATTERN.value == "synthetic":
        rate_str = (
            "inf"
            if __SYNTHETIC_REQUEST_RATE.value == 0
            else f"{__SYNTHETIC_REQUEST_RATE.value}rps"
        )
        sim_exp_name = f"{__SYSTEM.value}/{__MODEL.value}_synthetic_{rate_str}_{__OPT_GOAL.value}_{__ALLOCATION_SUCCESS_RATE.value}"
    else:
        sim_exp_name = f"{__SYSTEM.value}/{__MODEL.value}_{__TRACE.value}_{__OPT_GOAL.value}_{__ALLOCATION_SUCCESS_RATE.value}"
    if __SYSTEM.value == "MultiPool":
        sim_exp_name += f"_{__NUM_POOLS.value}pools"

    # create sim config
    sim_config = get_base_config(__MODEL.value)

    if __VALIDATE_ONLY.value:
        logging.info("FleetSim configuration validated: %s", sim_config)
        return

    # directly compute Ideal stats as this does not need event-driven simulation.
    if __SYSTEM.value == "Ideal":
        _initialize_ideal_backend(sim_config)
        ideal_sim.run(sim_config, name=sim_exp_name, n_cpu=__N_CPU.value)
        return

    # Initialize the simulator with the configuration.
    simulator = NPUFleetSimulator(sim_config, name=sim_exp_name)

    # Run the simulation.
    # ${sysname}/${modelname}_${tracename}_${optgoal}_${alloc_rate}"
    logging.info("%s Starting simulation...", sim_exp_name)
    simulator.run()
    # logging.info("%s Simulation completed.", sim_exp_name)

    # Dump simulation statistics.
    simulator.dump_simulation_stats()

    # checkpoint the simulator object into pickle file
    simulator.save_to_checkpoint()


def init_cmd_args():
    global CONFIGS_PATH
    CONFIGS_PATH = __CONFIGS_PATH.value

    logging.info(f"Using configs path: {CONFIGS_PATH}")


if __name__ == "__main__":
    app.run(main)
