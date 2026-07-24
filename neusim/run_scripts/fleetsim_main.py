"""Run a fixed-vPod FleetSim experiment with NeuSim as the backend."""

import json
import os
from pathlib import Path

from absl import app, flags, logging

from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig
from neusim.configs.systems.NPUFleetConfig import NPUFleetConfig
from neusim.configs.workloads.LLMInferenceWorkloadConfig import StaticVPodAllocation
from neusim.fleetsim.NPUFleetSimulator import NPUFleetSimulator
from neusim.npusim.frontend.query_results_helper_lib import is_model_llm_moe

NEUSIM_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = NEUSIM_PACKAGE_ROOT.parent
DEFAULT_CONFIGS_DIR = Path(
    os.environ.get("NEUSIM_CONFIGS_DIR") or REPO_ROOT / "configs"
).expanduser()
DEFAULT_RESULTS_DIR = Path(
    os.environ.get("NEUSIM_RESULTS_DIR", Path.cwd() / "results" / "fleetsim")
).expanduser()
DEFAULT_TRACES_DIR = Path(
    os.environ.get("NEUSIM_TRACES_DIR", Path.cwd() / "traces" / "inference")
).expanduser()


__MODEL = flags.DEFINE_string("model", "llama3-70b", "Model config name.")
__TRACE = flags.DEFINE_string("trace", "Azure-Code", "Trace alias or CSV path.")
__TRACE_FILE = flags.DEFINE_string(
    "trace_file", None, "Explicit trace CSV; overrides --trace."
)
__TRACES_DIR = flags.DEFINE_string(
    "traces_dir", str(DEFAULT_TRACES_DIR), "Root of external trace datasets."
)
__STATIC_VPOD_ALLOCATION = flags.DEFINE_string(
    "static_vpod_allocation",
    None,
    "Required JSON file with fixed prefill and decode vPod groups.",
)
__CONFIGS_PATH = flags.DEFINE_string(
    "configs_path", str(DEFAULT_CONFIGS_DIR), "NeuSim configs directory."
)
__OUTPUT_DIR = flags.DEFINE_string(
    "output_dir", str(DEFAULT_RESULTS_DIR), "Simulation output directory."
)
__MAX_TIMESTAMP_HOURS = flags.DEFINE_float(
    "max_timestamp_hours", -1.0, "Stop after this many trace hours; -1 is unlimited."
)
__MAX_NUM_REQUESTS = flags.DEFINE_integer(
    "max_num_requests", -1, "Stop after this many requests; -1 is unlimited."
)
__NPUSIM_BACKEND_CACHE_DIR = flags.DEFINE_string(
    "npusim_backend_cache_dir",
    str(DEFAULT_RESULTS_DIR / ".cache" / "npusim_backend"),
    "Persistent NeuSim backend cache directory.",
)
__NPUSIM_BACKEND_CACHE_USE_MMAP = flags.DEFINE_bool(
    "npusim_backend_cache_use_mmap",
    False,
    "Read the NeuSim backend cache through memory mapping.",
)
__EXPERT_LOAD_IMBALANCE_FACTOR = flags.DEFINE_float(
    "expert_load_imbalance_factor",
    -1.0,
    "MoE expert imbalance: -1 selects the model's worst-case default.",
)
__TQDM = flags.DEFINE_bool("tqdm", False, "Show simulation progress.")
__VALIDATE_ONLY = flags.DEFINE_bool(
    "validate_only", False, "Validate all inputs, then exit without simulating."
)
__REQUEST_RATE = flags.DEFINE_float(
    "request_rate", 1.0, "Multiplier applied to a trace's arrival rate."
)
__REQUEST_PATTERN = flags.DEFINE_enum(
    "request_pattern", "trace", ["trace", "synthetic"], "Request source."
)
__SYNTHETIC_NUM_REQUESTS = flags.DEFINE_integer(
    "synthetic_num_requests", 2000, "Number of synthetic requests."
)
__SYNTHETIC_REQUEST_RATE = flags.DEFINE_float(
    "synthetic_request_rate",
    10.0,
    "Synthetic arrivals per second; 0 enqueues all requests at time zero.",
)
__SYNTHETIC_INPUT_LEN = flags.DEFINE_integer(
    "synthetic_input_len", 512, "Synthetic mean input length."
)
__SYNTHETIC_INPUT_LEN_STD = flags.DEFINE_integer(
    "synthetic_input_len_std", 0, "Synthetic input-length standard deviation."
)
__SYNTHETIC_OUTPUT_LEN = flags.DEFINE_integer(
    "synthetic_output_len", 128, "Synthetic mean output length."
)
__SYNTHETIC_OUTPUT_LEN_STD = flags.DEFINE_integer(
    "synthetic_output_len_std", 0, "Synthetic output-length standard deviation."
)
__SYNTHETIC_SEED = flags.DEFINE_integer(
    "synthetic_seed", 42, "Synthetic workload random seed."
)
__PAD_SEQLEN_LOADGEN = flags.DEFINE_bool(
    "pad_seqlen_loadgen", True, "Pad trace lengths to the simulator buckets."
)
__MAX_DECODE_BATCH_SIZE = flags.DEFINE_integer(
    "max_decode_batch_size",
    -1,
    "Decode request cap; -1 selects the paper default for known model/trace pairs.",
)
__MIN_DECODE_SCHEDULE_NUM_ITERATIONS = flags.DEFINE_integer(
    "min_decode_schedule_num_iterations",
    4,
    "Minimum tokens per decode scheduling turn.",
)
__MAX_DECODE_SCHEDULE_NUM_ITERATIONS = flags.DEFINE_integer(
    "max_decode_schedule_num_iterations",
    256,
    "Maximum tokens per decode scheduling turn.",
)
__DECODE_BATCH_SEQLEN_RATIO_THRESHOLD = flags.DEFINE_float(
    "decode_batch_seqlen_ratio_threshold",
    2.0,
    "Maximum total-length ratio for batching long decode requests.",
)
__DECODE_BATCH_SEQLEN_MIN_THRESHOLD = flags.DEFINE_integer(
    "decode_batch_seqlen_min_threshold",
    256,
    "Length below which decode requests may batch regardless of ratio.",
)
__ENABLE_PROFILE = flags.DEFINE_bool(
    "enable_profile", False, "Enable event and backend profiling."
)
__ENABLE_DVFS_POWER_MODEL = flags.DEFINE_bool(
    "enable_dvfs_power_model",
    False,
    (
        "Use component-level voltage/frequency power accounting, including "
        "for a peak-frequency NoDVFS baseline."
    ),
)
__ENABLE_DVFS = flags.DEFINE_bool(
    "enable_dvfs",
    False,
    "Enable service-level DVFS while preserving the static vPod allocation.",
)
__DVFS_POLICY = flags.DEFINE_enum(
    "dvfs_policy",
    "Custom",
    ["None", "Ideal", "DVFSC", "DVFSCms", "Custom", "CustomAll", "CustomAllms"],
    "NPU DVFS policy used for each formed request batch.",
)
__DVFS_MAX_PERF_DEGRAD = flags.DEFINE_float(
    "dvfs_max_perf_degrad", 1.0, "Maximum request-batch slowdown ratio."
)
__DVFS_SAFEGUARD_WINDOW_MINUTES = flags.DEFINE_float(
    "dvfs_safeguard_window_minutes",
    5.0,
    "Safeguard-2 sliding SLO window in minutes.",
)
__DVFS_SAFEGUARD_VIOLATION_THRESHOLD = flags.DEFINE_float(
    "dvfs_safeguard_violation_threshold",
    0.01,
    "Violation rate above which safeguard 2 locks execution to peak.",
)
__DVFS_LOOKUP_CACHE_DIR = flags.DEFINE_string(
    "dvfs_lookup_cache_dir", "", "Policy-specific service DVFS lookup cache."
)
__DVFS_REQUIRE_CACHE_HIT = flags.DEFINE_bool(
    "dvfs_require_cache_hit",
    False,
    "Fail on absent/malformed cache data or any runtime cache miss.",
)
__SLO_JSON_PATH = flags.DEFINE_string(
    "slo_json_path",
    "",
    "Percentile-bucket SLO JSON required for service DVFS.",
)
__SLO_MULTIPLIER = flags.DEFINE_string(
    "slo_multiplier", "5x", "SLO multiplier key selected from the SLO JSON."
)

CONFIGS_PATH: str | None = None


def get_trace_filename(
    trace_name: str, traces_dir: str | os.PathLike[str] | None = None
) -> str:
    """Resolve an external-dataset alias or direct CSV path."""
    explicit_path = Path(trace_name).expanduser()
    if explicit_path.is_file():
        return str(explicit_path.resolve())

    relative_traces = {
        "Azure-Conv": "AzurePublicDataset/data/AzureLLMInferenceTrace_conv_1week_sampled.csv",
        "Azure-Code": "AzurePublicDataset/data/AzureLLMInferenceTrace_code_1week_sampled.csv",
        "Azure-Code-1day": "AzurePublicDataset/data/AzureLLMInferenceTrace_code_1day.csv",
        "Azure2Hr": "AzurePublicDataset/data/AzureLLMInferenceTrace_code_14to16hr.csv",
        "Azure4Hr": "AzurePublicDataset/data/AzureLLMInferenceTrace_code_16to20hr.csv",
        "BurstGPT-Conv": "BurstGPT/data/BurstGPT_without_fails_1_conv_sampled.csv",
        "BurstGPT-API": "BurstGPT/data/BurstGPT_without_fails_1_api_sampled.csv",
    }
    try:
        relative_path = relative_traces[trace_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown trace {trace_name!r}; use --trace_file for an explicit CSV."
        ) from exc
    return str(Path(traces_dir or DEFAULT_TRACES_DIR) / relative_path)


def get_max_decode_batch_size(model: str, trace: str) -> int:
    """Return the fixed cap used by the paper, or no cap for other workloads."""
    if __MAX_DECODE_BATCH_SIZE.value != -1:
        return __MAX_DECODE_BATCH_SIZE.value
    paper_caps = {
        "llama3-70b": {"Azure-Code": 2},
        "llama3_1-405b": {"Azure-Code": 2},
        "deepseekv2-236b": {"Azure-Code": 2},
        "deepseekv3-671b": {"Azure-Code": 2},
    }
    return paper_caps.get(model, {}).get(trace, -1)


def _load_static_allocation(path: str | None) -> StaticVPodAllocation:
    if not path:
        raise ValueError("--static_vpod_allocation is required")
    with Path(path).expanduser().open() as allocation_file:
        return StaticVPodAllocation.model_validate(json.load(allocation_file))


def _build_model_config(
    model: str,
    allocation: StaticVPodAllocation,
    config_root: Path,
) -> tuple[LLMConfig | DeepSeekConfig, dict]:
    """Build the base model with the same topology axes as the paper simulator."""
    entry = allocation.prefill
    with (config_root / "models" / f"{model}.json").open() as config_file:
        model_config = json.load(config_file)
    with (config_root / "chips" / f"tpuv{entry.npu_type}.json").open() as config_file:
        chip_config = json.load(config_file)
    with (config_root / "systems" / "system_config.json").open() as config_file:
        system_config = json.load(config_file)

    config = {**model_config, **chip_config, **system_config}
    config.update(
        {
            "name": entry.npu_type,
            "num_chips": entry.num_chips,
            "data_parallelism_degree": entry.dp,
            "tensor_parallelism_degree": entry.tp,
            "pipeline_parallelism_degree": entry.pp,
            "microbatch_size_ici": entry.batch_size,
            "global_batch_size": entry.batch_size * entry.pp,
            "microbatch_size_dcn": entry.batch_size * entry.pp,
            "num_data_parallel_axes": 0,
            "num_tensor_parallel_axes": 2,
            "num_pipeline_parallel_axes": 1,
        }
    )
    if is_model_llm_moe(model):
        config["num_expert_parallel_axes"] = 1
        config["expert_parallelism_degree"] = entry.ep
        config["expert_load_imbalance_factor"] = __EXPERT_LOAD_IMBALANCE_FACTOR.value
    cls = DeepSeekConfig if "deepseek" in model.lower() else LLMConfig
    return cls.model_validate(config), system_config


def get_base_config(model: str) -> NPUFleetConfig:
    """Build the complete fixed-deployment simulation configuration."""
    global CONFIGS_PATH
    if not CONFIGS_PATH:
        raise RuntimeError("init_cmd_args() must be called before get_base_config()")
    config_root = Path(CONFIGS_PATH).expanduser()
    allocation = _load_static_allocation(__STATIC_VPOD_ALLOCATION.value)
    model_config, system_config = _build_model_config(model, allocation, config_root)
    request_pattern = __REQUEST_PATTERN.value
    trace_path = (
        __TRACE_FILE.value or get_trace_filename(__TRACE.value, __TRACES_DIR.value)
        if request_pattern == "trace"
        else ""
    )
    workload = {
        "static_vpod_allocation": allocation,
        "request_pattern": request_pattern,
        "trace_file_path": trace_path,
        "request_rate": __REQUEST_RATE.value,
        "model_name": model,
        "llm_config": model_config,
        "max_timestamp": int(__MAX_TIMESTAMP_HOURS.value * 60 * 60 * 1e9)
        if __MAX_TIMESTAMP_HOURS.value > 0
        else -1,
        "max_num_requests": __MAX_NUM_REQUESTS.value,
        "pad_seqlen_loadgen": __PAD_SEQLEN_LOADGEN.value,
        "max_decode_batch_size": get_max_decode_batch_size(model, __TRACE.value),
        "min_decode_schedule_num_iterations": __MIN_DECODE_SCHEDULE_NUM_ITERATIONS.value,
        "max_decode_schedule_num_iterations": __MAX_DECODE_SCHEDULE_NUM_ITERATIONS.value,
        "decode_batch_seqlen_ratio_threshold": __DECODE_BATCH_SEQLEN_RATIO_THRESHOLD.value,
        "decode_batch_seqlen_min_threshold": __DECODE_BATCH_SEQLEN_MIN_THRESHOLD.value,
        "synthetic_num_requests": __SYNTHETIC_NUM_REQUESTS.value,
        "synthetic_request_rate": float("inf")
        if __SYNTHETIC_REQUEST_RATE.value == 0
        else __SYNTHETIC_REQUEST_RATE.value,
        "synthetic_input_len": __SYNTHETIC_INPUT_LEN.value,
        "synthetic_input_len_std": __SYNTHETIC_INPUT_LEN_STD.value,
        "synthetic_output_len": __SYNTHETIC_OUTPUT_LEN.value,
        "synthetic_output_len_std": __SYNTHETIC_OUTPUT_LEN_STD.value,
        "synthetic_seed": __SYNTHETIC_SEED.value,
        "enable_dvfs_power_model": __ENABLE_DVFS_POWER_MODEL.value,
        "enable_dvfs": __ENABLE_DVFS.value,
        "dvfs_policy": __DVFS_POLICY.value,
        "dvfs_max_perf_degrad": __DVFS_MAX_PERF_DEGRAD.value,
        "dvfs_safeguard_window_minutes": __DVFS_SAFEGUARD_WINDOW_MINUTES.value,
        "dvfs_safeguard_violation_threshold": __DVFS_SAFEGUARD_VIOLATION_THRESHOLD.value,
        "dvfs_lookup_cache_dir": __DVFS_LOOKUP_CACHE_DIR.value,
        "dvfs_require_cache_hit": __DVFS_REQUIRE_CACHE_HIT.value,
        "slo_json_path": __SLO_JSON_PATH.value,
        "slo_multiplier": __SLO_MULTIPLIER.value,
    }
    return NPUFleetConfig.model_validate(
        {
            "cluster_scheduler_config": {
                "chip_config_path": str(config_root / "chips")
            },
            "workload_config": workload,
            "system_config": system_config,
            "tqdm": __TQDM.value,
            "enable_profile": __ENABLE_PROFILE.value,
            "npusim_backend_cache_dir": __NPUSIM_BACKEND_CACHE_DIR.value,
            "npusim_backend_cache_use_mmap": __NPUSIM_BACKEND_CACHE_USE_MMAP.value,
            "output_dir": __OUTPUT_DIR.value,
        }
    )


def init_cmd_args() -> None:
    global CONFIGS_PATH
    CONFIGS_PATH = __CONFIGS_PATH.value
    logging.info("Using configs path: %s", CONFIGS_PATH)


def main(argv) -> None:
    del argv
    init_cmd_args()
    sim_config = get_base_config(__MODEL.value)
    workload_config = sim_config.workload_config
    if workload_config.enable_dvfs:
        from neusim.fleetsim.dvfs_scheduler import load_dvfs_lookup_cache

        if not Path(workload_config.slo_json_path).expanduser().is_file():
            raise FileNotFoundError(
                f"SLO JSON not found: {workload_config.slo_json_path}"
            )
        load_dvfs_lookup_cache(
            workload_config.dvfs_lookup_cache_dir,
            strict=workload_config.dvfs_require_cache_hit,
        )
    if __VALIDATE_ONLY.value:
        logging.info("FleetSim configuration validated: %s", sim_config)
        return

    workload_name = (
        f"synthetic_{__SYNTHETIC_REQUEST_RATE.value}rps"
        if __REQUEST_PATTERN.value == "synthetic"
        else __TRACE.value
    )
    simulator = NPUFleetSimulator(
        sim_config, name=f"Static/{__MODEL.value}_{workload_name}"
    )
    logging.info("Starting static FleetSim simulation")
    simulator.run()
    simulator.dump_simulation_stats()


if __name__ == "__main__":
    app.run(main)
