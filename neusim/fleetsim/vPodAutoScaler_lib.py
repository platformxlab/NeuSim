### Library that contains helper functions for vPodAutoScaler.py.

import csv
import glob
import json
import math
import os
from collections.abc import Sequence
from copy import deepcopy
from functools import cache
from typing import Any

import numpy as np
from absl import logging

import neusim.fleetsim.npusim_backend_interface as npusim_backend
import neusim.fleetsim.util as sim_util
from neusim.configs.models.LLMConfig import (
    DeepSeekConfig,
    LLMConfig,
    MoELLMConfig,
)
from neusim.configs.systems.NPUFleetConfig import NPUFleetConfig
from neusim.fleetsim import cost_model
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.npusim.frontend.Operator import Operator

# Module-level caches for expensive lookups that are deterministic within a simulation run.
_seqlen_to_cfgs_cache: dict[tuple[int, int, str], list[LLMConfig]] = {}
_ideal_latency_cache: dict[tuple, float] = {}


def num_chips_to_shape_2D(num_chips: int) -> list[int]:
    """
    Convert number of chips to a 2D shape (dimX, dimY).
    The shape is chosen to be as square as possible.
    """
    dimX = math.isqrt(num_chips)
    while num_chips % dimX != 0:
        dimX -= 1
    dimY = num_chips // dimX
    return [dimX, dimY]


def num_chips_to_shape_3D(num_chips: int) -> list[int]:
    """
    Convert number of chips to a 3D shape (dimX, dimY, dimZ).
    The shape is chosen to be as cube-like as possible.
    Examples:
        1 -> [1, 1, 1];
        4 -> [1, 2, 2];
        4096 -> [16, 16, 16];
    """
    dimZ = int(math.cbrt(num_chips))
    while num_chips % dimZ != 0:
        dimZ -= 1
    num_chips //= dimZ
    dimY = int(math.sqrt(num_chips))
    while num_chips % dimY != 0:
        dimY -= 1
    dimX = num_chips // dimY
    return [dimX, dimY, dimZ]


def shape_to_num_chips(shape: Sequence[int]) -> int:
    """
    Convert a shape to the number of chips.
    """
    return math.prod(shape)


def get_vPod_create_delay_ns(config: LLMConfig) -> int:
    """
    Get the vPod creation delay in nanoseconds based on how many chips and VMs
    need to be initialized and how large the model needs to be loaded from the (remote) storage.
    """
    # raise NotImplementedError("get_vPod_create_delay_ns is not implemented yet.")
    # TODO: implement actual logic here
    return int(1 * 60 * 1e9)  # 1 min; for debugging purposes only


def pad_to(x: int, pad_to_multiple_of: int = 128) -> int:
    """
    Pad the input integer to the next multiple of the specified value.
    """
    return (x + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of


def pad_seqlen(
    x: int, padding_factors: Sequence[int], padding_steps: Sequence[int]
) -> int:
    """
    Pad @x to the appropriate length based on the padding factors and steps.
    The returned value must be >= @x.
    For example, if padding factors = [128, 512, 1024] and padding steps = [1024, 4096], then we will have:

        seqlen <= 1024 -> pad to multiples of 128
        seqlen <= 4096 -> pad to multiples of 512
        seqlen > 4096 -> pad to multiples of 1024
    """
    assert (
        len(padding_factors) == len(padding_steps) + 1
    ), "padding_factors must have one more element than padding_steps"

    for i, step in enumerate(padding_steps):
        if x <= step:
            return pad_to(x, padding_factors[i])

    return pad_to(x, padding_factors[-1])


@cache
def load_lookup_cache(
    model: str,
    input_seqlen: int,
    output_seqlen: int,
    versions: tuple[str],
    req_lookup_cache_dir: str,
    opt_goal: str,
    prefill_or_decode: str,
) -> dict[str, list[tuple[list[dict[str, Any]], dict[str, Any]]]]:
    """
    Returns a mapping: (npu version) -> (list of top-k optimal (op_dicts, stats)).
    """
    cache_dict = {v: [] for v in versions}
    for v in versions:
        cache_dir = os.path.join(
            req_lookup_cache_dir,
            opt_goal,
            model,
            f"{input_seqlen}_{output_seqlen}",
            v,
            prefill_or_decode,
        )
        all_stats_filenames = glob.glob(os.path.join(cache_dir, "*.json"))
        all_stats_filenames.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
        # assert len(all_stats_filenames) > 0, f"No stats found for {v} in {cache_dir}"
        if len(all_stats_filenames) == 0:
            logging.debug(f"No valid config found for {v} in {cache_dir}.")
        for stats_filename in all_stats_filenames:
            # load stats from json files
            with open(stats_filename) as f:
                stats = json.load(f)
            # Operator rows are retained for provenance by full search caches, but
            # FleetSim's runtime selection consumes only ``stats``. Compact
            # artifact caches may therefore omit the matching CSV without changing
            # any configuration, cost, latency, or SLO decision.
            csv_filename = stats_filename.replace(".json", ".csv")
            if os.path.isfile(csv_filename):
                with open(csv_filename) as f:
                    reader = csv.DictReader(f)
                    op_dicts = list(reader)
            else:
                op_dicts = []
            if (
                len(cache_dict[v]) == 0
                or cache_dict[v][-1][1]["sim_config"]["num_chips"]
                > stats["sim_config"]["num_chips"]
            ):
                # only add the sub-optimal config if it requires fewer number of chips
                cache_dict[v].append((op_dicts, stats))
    # check that there is at least one valid config in cache_dict
    if not any(len(cfgs) > 0 for cfgs in cache_dict.values()):
        logging.warning(
            "No valid config found in %s.",
            os.path.join(
                req_lookup_cache_dir,
                opt_goal,
                model,
                f"{input_seqlen}_{output_seqlen}",
                f"{versions}",
                prefill_or_decode,
            ),
        )

    return cache_dict


def get_cost_metric_name(opt_goal: str) -> str:
    if opt_goal == "energy":
        return "avg_power_efficiency_tkn_per_joule"
    elif opt_goal == "monetary":
        return "monetary_cost_tkn_per_dollar"
    else:
        raise ValueError(f"Unknown optimization goal: {opt_goal}")


def get_slo_latency(
    model_name: str,
    input_seqlen: int,
    output_seqlen: int,
    mode: str,
    opt_goal: str,
    req_lookup_cache_dir: str,
    baseline_versions: Sequence[str] = ("5p", "6e"),
) -> float:
    """
    Get the SLO latency based on the input and output sequence lengths and mode (prefill or decode).

    The SLO threshold is stamped identically into every NPU version's cached config
    (it is fixed by --slo_baseline_version at cache-generation time), so we read it from
    the first @baseline_versions entry that actually has a populated cache. Pass the
    deployed/baseline versions here (e.g. ("6e", "5e", "4")) so that a missing "5p" cache
    does not spuriously yield an infinite (always-satisfied) SLO.
    """
    lookup_cache = load_lookup_cache(
        model_name,
        input_seqlen,
        output_seqlen,
        tuple(baseline_versions),
        req_lookup_cache_dir,
        opt_goal,
        mode,
    )
    v_key = next(
        (v for v in baseline_versions if len(lookup_cache.get(v, [])) > 0), None
    )
    if v_key is None:
        logging.warning(
            "No valid config for %s seqlen=%d_%d mode=%s; returning inf SLO latency.",
            model_name,
            input_seqlen,
            output_seqlen,
            mode,
        )
        return float("inf")
    stats = lookup_cache[v_key][0][1]

    slo_key = "slo_TTFT_sec" if mode == "prefill" else "slo_TPOT_ms_request"
    return stats[slo_key]


def get_ideal_latency_for_config(
    config: NPUFleetConfig,
    llm_config: LLMConfig,
    prefill_or_decode: str,
) -> float:
    """
    Get ideal latency for a specific vPod config from the lookup cache.
    Returns TTFT_sec (seconds) for prefill, TPOT_ms_request (milliseconds) for decode.
    """
    cache_key = (
        llm_config.name,
        llm_config.num_chips,
        llm_config.input_seqlen,
        llm_config.output_seqlen,
        prefill_or_decode,
    )
    if cache_key in _ideal_latency_cache:
        return _ideal_latency_cache[cache_key]

    model_name = config.workload_config.model_name
    input_seqlen = llm_config.input_seqlen
    output_seqlen = llm_config.output_seqlen
    req_lookup_cache_dir = config.workload_config.request_results_cache_dir
    opt_goal = config.workload_config.optimization_goal

    sched = config.cluster_scheduler_config
    if prefill_or_decode == "prefill" and sched.prefill_npu_types is not None:
        npu_types = sched.prefill_npu_types
    elif prefill_or_decode == "decode" and sched.decode_npu_types is not None:
        npu_types = sched.decode_npu_types
    else:
        npu_types = sched.npu_types

    lookup_cache = load_lookup_cache(
        model_name,
        input_seqlen,
        output_seqlen,
        tuple(npu_types),
        req_lookup_cache_dir,
        opt_goal,
        prefill_or_decode,
    )

    latency_key = "TTFT_sec" if prefill_or_decode == "prefill" else "TPOT_ms_request"

    # Search for entry matching the given config
    target_name = llm_config.name
    target_num_chips = llm_config.num_chips
    target_batch_size = llm_config.microbatch_size_ici

    best_entry = None
    for _version, cfgs in lookup_cache.items():
        for _, stats in cfgs:
            sc = stats["sim_config"]
            if (
                sc["name"] == target_name
                and sc["num_chips"] == target_num_chips
                and sc["microbatch_size_ici"] == target_batch_size
            ):
                best_entry = stats
                break
        if best_entry is not None:
            break

    # Fall back to first entry if no exact match
    if best_entry is None:
        for _version, cfgs in lookup_cache.items():
            if len(cfgs) > 0:
                best_entry = cfgs[0][1]
                break

    if best_entry is None:
        logging.warning(
            "No lookup cache entry for %s %s (name=%s, chips=%d, batch=%d); returning inf.",
            model_name,
            prefill_or_decode,
            target_name,
            target_num_chips,
            target_batch_size,
        )
        return float("inf")

    result = best_entry[latency_key]
    _ideal_latency_cache[cache_key] = result
    return result


def get_optimal_vPod_config(
    config: NPUFleetConfig,
    prefill_or_decode: str,
    npu_types: Sequence[str] | None = None,
) -> list[LLMConfig]:
    """
    @return: the list of top-k optimal vPod configurations [(NPU type, config), ...] based on the model and in/out seq len.
    Do not change the original config.
    """
    model_name = config.workload_config.model_name
    input_seqlen = config.workload_config.llm_config.input_seqlen
    output_seqlen = config.workload_config.llm_config.output_seqlen

    req_lookup_cache_dir = config.workload_config.request_results_cache_dir
    opt_goal = config.workload_config.optimization_goal
    if npu_types is None:
        sched = config.cluster_scheduler_config
        if prefill_or_decode == "prefill" and sched.prefill_npu_types is not None:
            npu_types = sched.prefill_npu_types
        elif prefill_or_decode == "decode" and sched.decode_npu_types is not None:
            npu_types = sched.decode_npu_types
        else:
            npu_types = sched.npu_types
    cost_metric_name = get_cost_metric_name(opt_goal)

    lookup_cache = load_lookup_cache(
        model_name,
        input_seqlen,
        output_seqlen,
        tuple(npu_types),
        req_lookup_cache_dir,
        opt_goal,
        prefill_or_decode,
    )

    # find best config for each NPU type
    # optimal_lookup_cache: list[tuple[str, tuple[list[dict[str, Any]], dict[str, Any]]]] = []
    # for npu in npu_types:
    #     if len(lookup_cache[npu]) > 0:
    #         optimal_lookup_cache.append((npu, max(lookup_cache[npu], key=lambda x: x[1][cost_metric_name])))
    # assert len(optimal_lookup_cache) > 0, f"No lookup cache found for model {model_name} with input_seqlen {input_seqlen} and output_seqlen {output_seqlen} in {req_lookup_cache_dir}"

    # find best NPU type
    # best_npu_type, (best_op_dicts, best_stats) = max(optimal_lookup_cache, key=lambda x: x[1][1][cost_metric_name])

    # load best_stats["sim_config"] into LLMConfig
    if "deepseek" in model_name.lower():
        ConfigBaseModel = DeepSeekConfig
    else:
        ConfigBaseModel = LLMConfig

    # optimal_lookup_cache: dict[str, list[LLMConfig]] = {
    #     v: [ConfigBaseModel.model_validate(cfg) for _, cfg in cfgs]
    #     for v, cfgs in lookup_cache.items()
    # }
    optimal_cfgs_with_cost: list[tuple[LLMConfig, float]] = [
        (ConfigBaseModel.model_validate(cfg["sim_config"]), cfg[cost_metric_name])
        for v, cfgs in lookup_cache.items()
        for _, cfg in cfgs
    ]
    if isinstance(config.workload_config.llm_config, MoELLMConfig):
        override_factor = config.workload_config.llm_config.expert_load_imbalance_factor
        for cfg, _ in optimal_cfgs_with_cost:
            assert isinstance(cfg, MoELLMConfig)
            cfg.expert_load_imbalance_factor = override_factor
    optimal_cfgs_with_cost.sort(key=lambda x: x[1], reverse=True)
    optimal_cfgs: list[LLMConfig] = [cfg for cfg, _ in optimal_cfgs_with_cost]
    if len(optimal_cfgs) == 0:
        logging.warning(
            "No valid config found for model %s with input_seqlen %d and output_seqlen %d in %s.",
            model_name,
            input_seqlen,
            output_seqlen,
            req_lookup_cache_dir,
        )
    return optimal_cfgs  # the caller must handle empty return list case

    # optimal_config = ConfigBaseModel.model_validate(best_stats["sim_config"])
    # return optimal_config


@cache
def get_available_cache_seqlens(
    req_lookup_cache_dir: str,
    opt_goal: str,
    model: str,
) -> list[tuple[int, int]]:
    """
    Scan the cache directory for available (input_seqlen, output_seqlen) subdirectories.
    Returns a sorted list of (input_seqlen, output_seqlen) tuples.
    """
    cache_dir = os.path.join(req_lookup_cache_dir, opt_goal, model)
    if not os.path.isdir(cache_dir):
        return []
    seqlen_pairs: list[tuple[int, int]] = []
    for entry in os.listdir(cache_dir):
        parts = entry.split("_")
        if len(parts) == 2:
            try:
                input_seqlen, output_seqlen = int(parts[0]), int(parts[1])
                seqlen_pairs.append((input_seqlen, output_seqlen))
            except ValueError:
                continue
    seqlen_pairs.sort()
    return seqlen_pairs


def get_ordered_seqlen_fallbacks(
    available_seqlens: Sequence[tuple[int, int]],
    input_seqlen: int,
    output_seqlen: int,
    prefill_or_decode: str,
) -> list[tuple[int, int]]:
    """Return every eligible cache fallback in deterministic preference order.

    Prefill is governed primarily by input length, so equal-input cache entries
    with a different output length remain eligible. Decode is governed by the
    total context length and therefore requires a strictly smaller total.
    """
    requested = (int(input_seqlen), int(output_seqlen))
    available = sorted({(int(pair[0]), int(pair[1])) for pair in available_seqlens})
    if prefill_or_decode == "prefill":
        candidates = [
            pair for pair in available if pair != requested and pair[0] <= requested[0]
        ]
        return sorted(
            candidates,
            key=lambda pair: (
                -pair[0],
                abs(pair[1] - requested[1]),
                -pair[1],
            ),
        )
    if prefill_or_decode == "decode":
        candidates = [pair for pair in available if sum(pair) < sum(requested)]
        return sorted(candidates, key=lambda pair: (-sum(pair), pair[0], pair[1]))
    raise ValueError(
        f"Unknown phase {prefill_or_decode!r}; expected 'prefill' or 'decode'."
    )


def get_optimal_vPod_config_with_seqlen_fallback(
    config: NPUFleetConfig,
    prefill_or_decode: str,
    npu_types: Sequence[str] | None = None,
) -> list[LLMConfig]:
    """
    Wrapper around get_optimal_vPod_config that falls back to progressively
    smaller sequence lengths from the cache when no valid config is found
    for the requested sequence length.
    """
    result = get_optimal_vPod_config(config, prefill_or_decode, npu_types)
    if len(result) > 0:
        return result

    # No valid config for the requested seqlen; try smaller seqlens from cache
    model_name = config.workload_config.model_name
    input_seqlen = config.workload_config.llm_config.input_seqlen
    output_seqlen = config.workload_config.llm_config.output_seqlen
    req_lookup_cache_dir = config.workload_config.request_results_cache_dir
    opt_goal = config.workload_config.optimization_goal

    available_seqlens = get_available_cache_seqlens(
        req_lookup_cache_dir, opt_goal, model_name
    )
    fallbacks = get_ordered_seqlen_fallbacks(
        available_seqlens,
        input_seqlen,
        output_seqlen,
        prefill_or_decode,
    )
    if not fallbacks:
        raise RuntimeError(
            f"No valid config found for model {model_name} ({prefill_or_decode}) "
            f"with input_seqlen {input_seqlen} and output_seqlen {output_seqlen}, "
            f"and no smaller seqlen fallback available in {req_lookup_cache_dir}."
        )

    for cand_input, cand_output in fallbacks:
        fallback_cfg = deepcopy(config)
        fallback_cfg.workload_config.llm_config.input_seqlen = cand_input
        fallback_cfg.workload_config.llm_config.output_seqlen = cand_output
        result = get_optimal_vPod_config(fallback_cfg, prefill_or_decode, npu_types)
        if len(result) > 0:
            logging.warning(
                "Falling back from seqlen (%d, %d) to (%d, %d) for model %s (%s).",
                input_seqlen,
                output_seqlen,
                cand_input,
                cand_output,
                model_name,
                prefill_or_decode,
            )
            return result

    tried = ", ".join(
        f"({input_len}, {output_len})" for input_len, output_len in fallbacks
    )
    raise RuntimeError(
        f"No valid config found for model {model_name} ({prefill_or_decode}) "
        f"with input_seqlen {input_seqlen} and output_seqlen {output_seqlen}, "
        f"even after trying eligible cache fallbacks: {tried}."
    )


def find_percentile_pair(
    seq_len_pairs: Sequence[Sequence[int]], percentile: int | float, mode: str
) -> tuple[int, int]:
    """
    Finds the sequence length pair closest to a specified percentile.

    This function calculates the percentile for either input sequence lengths ('prefill')
    or output sequence lengths ('decode') and returns the original pair from the
    dataset that is numerically closest to that calculated percentile value.

    Args:
        seq_len_pairs: A list of pairs, where each pair is (input_seq_len, output_seq_len).
        percentile: The target percentile to calculate (e.g., 95 for 95th percentile).
        mode: A string, either 'prefill' or 'decode'.
              - 'prefill': Calculates percentile based on input sequence lengths (1st element).
              - 'decode': Calculates percentile based on output sequence lengths (2nd element).

    Returns:
        The pair [input_len, output_len] from the original list that is closest
        to the specified percentile value. Returns None if the input list is empty.
    """

    data = np.array(seq_len_pairs)

    # Determine which column to use based on the mode
    if mode == "prefill":
        target_lengths = data[:, 0]
    elif mode == "decode":
        # use input+output seq len for decode since performance and efficiency is related to both (which determine the KV cache size)
        target_lengths = data[:, 0] + data[:, 1]
    else:
        # Raise an error for invalid mode selection
        raise ValueError("Mode must be either 'prefill' or 'decode'.")

    # Calculate the percentile value for the selected column
    percentile_value = np.percentile(target_lengths, percentile)
    # print(f"Calculated {percentile}th percentile for {column_name} lengths: {percentile_value:.2f}")

    # --- Find the Closest Match ---
    # Find the absolute difference between each length and the percentile value
    differences = np.abs(target_lengths - percentile_value)

    # Find the index of the smallest difference
    closest_index = np.argmin(differences)

    # Retrieve the pair corresponding to that index
    closest_pair = data[closest_index]

    return closest_pair[0], closest_pair[1]


def recommend_seqlen_pair_by_percentile(
    seqlen_pairs: Sequence[Sequence[int]], percentile: int | float, mode: str
) -> tuple[int, int]:
    """
    Recommends a sequence length pair based on a specified percentile.

    This function uses the `find_percentile_pair` function to locate the pair
    of sequence lengths that corresponds to the given percentile.

    Args:
        seqlen_pairs: A list of pairs, where each pair is (input_seq_len, output_seq_len).
        percentile: The target percentile to calculate (e.g., 95 for 95th percentile).
        mode: A string, either 'prefill' or 'decode'.
              - 'prefill': Recommends based on input sequence lengths (1st element).
              - 'decode': Recommends based on output sequence lengths (2nd element).

    Returns:
        The recommended pair [input_len, output_len] from the original list.
    """

    return find_percentile_pair(seqlen_pairs, percentile, mode)


def recommend_seqlen_by_regression_prediction(
    seqlen_pairs: Sequence[tuple[int, int]],
    config: NPUFleetConfig,
    mode: str,
    num_samples: int = 5,
) -> tuple[int, int]:
    """
    Recommends a sequence length pair based on quadratic regression prediction.
    This function will sample a few seqlen pairs and call the backend npusim
    to get the estimated cost. Then, it will use quadratic regression to interpolate
    the cost function (mapping from seqlen pairs to estimated cost).
    Then, it will use the cost function to predict the total cost of all seqlen pairs.
    Based on the predicted total cost, it will recommend a sequence length pair.

    Args:
        seqlen_pairs: A list of pairs, where each pair is (input_seq_len, output_seq_len).
        config: The configuration object containing workload parameters.
        mode: A string, either 'prefill' or 'decode'.
        num_samples: The number of samples to use for regression.

    Returns:
        A tuple containing the recommended (input_seq_len, output_seq_len) pair.
    """

    import ray

    # Sort seqlen_pairs to make sure we can sample as uniformly as possible
    unique_seqlen_pairs = list(set(seqlen_pairs))
    if mode == "prefill":
        unique_seqlen_pairs = sorted(unique_seqlen_pairs, key=lambda x: x[0])
    elif mode == "decode":
        unique_seqlen_pairs = sorted(unique_seqlen_pairs, key=lambda x: x[0] + x[1])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # handle corner cases where we do not have enough samples
    if len(unique_seqlen_pairs) == 1:
        # if we only have a single seqlen pair, just use it
        return (unique_seqlen_pairs[0][0], unique_seqlen_pairs[0][1])
    if len(unique_seqlen_pairs) <= num_samples:
        # if we do not have enough samples, just return the max seqlen
        return (unique_seqlen_pairs[-1][0], unique_seqlen_pairs[-1][1])

    ##### some helper functions #####
    #     TODO: organize them in a better way

    cost_metric = config.workload_config.optimization_goal

    def get_optimal_config(input_seqlen: int, output_seqlen: int, mode: str):
        if mode == "prefill":
            output_seqlen = 4  # ignore output seqlen for prefill
        orig_cfg = deepcopy(config)
        orig_cfg.workload_config.llm_config.input_seqlen = input_seqlen
        orig_cfg.workload_config.llm_config.output_seqlen = output_seqlen
        cfgs = get_optimal_vPod_config_with_seqlen_fallback(orig_cfg, mode)
        for cfg in cfgs:
            cfg.enable_swap_kv_cache = True
        return cfgs[0]

    def get_monetary_cost(
        ops: list[Operator],
        mode: str,
        input_seqlen: int,
        output_seqlen: int,
        cfg: LLMConfig,
    ):
        if mode == "prefill":
            prefill_cost_dollars = cost_model.pipeline_batch_monetary_cost_dollars(
                ops, cfg
            )
            return prefill_cost_dollars / cfg.microbatch_size_ici
        elif mode == "decode":
            decode_cost_dollars_per_token = (
                cost_model.pipeline_batch_monetary_cost_dollars(ops, cfg)
            )
            return decode_cost_dollars_per_token / cfg.microbatch_size_ici
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_energy_cost(
        ops: list[Operator],
        mode: str,
        input_seqlen: int,
        output_seqlen: int,
        cfg: LLMConfig,
    ):
        if mode == "prefill":
            return (
                sum([op.stats.total_energy_J * op.stats.count for op in ops])
                * cfg.num_chips
                / cfg.microbatch_size_ici
            )
        elif mode == "decode":
            return (
                sum([op.stats.total_energy_J * op.stats.count for op in ops])
                * cfg.num_chips
                / cfg.microbatch_size_ici
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def get_cost_for_ops(
        ops: list[Operator],
        mode: str,
        input_seqlen: int,
        output_seqlen: int,
        cfg: LLMConfig,
    ):
        if cost_metric == "monetary":
            return get_monetary_cost(ops, mode, input_seqlen, output_seqlen, cfg)
        elif cost_metric == "energy":
            return get_energy_cost(ops, mode, input_seqlen, output_seqlen, cfg)
        else:
            raise ValueError(f"Unknown cost metric: {cost_metric}")

    def get_cost(input_seqlen: int, output_seqlen: int, cfg: LLMConfig, mode: str):
        prefill_ops, decode_ops = npusim_backend.run_inference_request_batch(
            cfg,
            [
                LLMRequest(input_seqlen, output_seqlen)
                for _ in range(cfg.microbatch_size_ici)
            ],
        )
        seqlen = input_seqlen if mode == "prefill" else (input_seqlen + output_seqlen)
        ops = prefill_ops if mode == "prefill" else decode_ops
        return seqlen, get_cost_for_ops(ops, mode, input_seqlen, output_seqlen, cfg)

    ##### END helper functions #####

    # Sample @num_samples seqlen pairs uniformly
    sample_pairs = sim_util.sample_from_list(unique_seqlen_pairs, num_samples)

    # Get all recommended configs
    recommended_cfgs: list[LLMConfig] = []
    seqlen_to_cfg_mapping: dict[tuple[int, int], LLMConfig] = {}
    for input_seqlen, output_seqlen in unique_seqlen_pairs:
        if mode == "prefill":
            opt_cfg = get_optimal_config(input_seqlen, 4, mode)
        else:
            opt_cfg = get_optimal_config(input_seqlen, output_seqlen, mode)
        if opt_cfg not in recommended_cfgs:
            recommended_cfgs.append(opt_cfg)
        seqlen_to_cfg_mapping[(input_seqlen, output_seqlen)] = opt_cfg

    # Get reverse mapping from cfgs to seqlen
    cfgs_to_seqlen: list[tuple[LLMConfig, list[tuple[int, int]]]] = [
        (cfg, []) for cfg in recommended_cfgs
    ]
    for seqlen, cfg in seqlen_to_cfg_mapping.items():
        for cfg_, seqlens in cfgs_to_seqlen:
            if cfg_ == cfg:
                seqlens.append(seqlen)

    # Generate the estimation sample points for each sampled seqlen pairs
    sampled_costs: dict[
        tuple[int, int], list[tuple[int, float]]
    ] = {}  # (input, output) -> (seqlen, cost)
    for cfg, seqlens in cfgs_to_seqlen:
        # samples: list[tuple[int, float]] = []
        # for iseq, oseq in sample_pairs:
        #     samples.append(
        #         get_cost(
        #             iseq, oseq, opt_cfg, mode
        #         )
        #     )
        # sampled_costs[(input_seqlen, output_seqlen)] = samples
        # with Pool() as p:
        #     costs = p.starmap(
        #         get_cost, [(iseq, oseq, opt_cfg, mode) for iseq, oseq in sample_pairs]
        #     )
        get_cost_remote = ray.remote(get_cost)
        futures = [
            get_cost_remote.remote(iseq, oseq, cfg, mode) for iseq, oseq in sample_pairs
        ]
        costs = ray.get(futures)
        for s_t in seqlens:
            sampled_costs[s_t] = costs

    # fit a polynomial regression model to the sampled points with degree 2
    # and predict the costs for all seqlens
    predicted_costs: dict[tuple[int, int], float] = {}  # (input, output) -> total cost
    for s_t, _costs in sampled_costs.items():
        costs = np.array(_costs).T
        coeffs = np.polyfit(costs[0], costs[1], 2)
        x_fit = [x[0] if mode == "prefill" else (x[0] + x[1]) for x in seqlen_pairs]
        y_fit = np.polyval(coeffs, x_fit)
        total_cost = np.sum(y_fit)
        predicted_costs[s_t] = total_cost

    # find min of predicted_costs
    min_predicted_cost = min(predicted_costs.items(), key=lambda x: x[1])

    if logging.level_debug():
        logging.debug(
            f"{mode}: Recommended seqlen pair by regression prediction: {min_predicted_cost[0]} with predicted total {cost_metric} cost: {min_predicted_cost[1]:.4f}"
        )
        logging.debug(
            f"{mode}: All seqlen pairs (len={len(seqlen_pairs)}): {seqlen_pairs}"
        )
        logging.debug(f"{mode}: All predicted costs: {predicted_costs}")

    return min_predicted_cost[0]


def _compute_optimal_configs_for_pair(
    config: NPUFleetConfig,
    input_seqlen: int,
    output_seqlen: int,
    mode: str,
) -> list[LLMConfig]:
    """Helper for Ray-parallelized optimal config lookup per seqlen pair."""
    config = deepcopy(config)
    config.workload_config.llm_config.input_seqlen = input_seqlen
    config.workload_config.llm_config.output_seqlen = output_seqlen
    return get_optimal_vPod_config_with_seqlen_fallback(config, mode)


def get_seqlen_to_configs_mapping(
    seqlen_pairs: Sequence[tuple[int, int]],
    config: NPUFleetConfig,
    mode: str,
) -> dict[tuple[int, int], list[LLMConfig]]:
    """
    Get the mapping from sequence length pairs to the top-k optimal vPod configurations.

    Args:
        seqlen_pairs: A list of pairs, where each pair is (input_seq_len, output_seq_len).
        config: The configuration object containing workload parameters.
        mode: A string, either 'prefill' or 'decode'.

    Returns:
        A dictionary mapping each (input_seq_len, output_seq_len) pair to a list of its optimal LLMConfigs.
    """
    seqlen_to_cfgs_mapping: dict[tuple[int, int], list[LLMConfig]] = {}
    unique_seqlen_pairs = set(seqlen_pairs)
    uncached_pairs = [
        p
        for p in unique_seqlen_pairs
        if (p[0], p[1], mode) not in _seqlen_to_cfgs_cache
    ]

    # Return cached results for known pairs
    for p in unique_seqlen_pairs:
        cache_key = (p[0], p[1], mode)
        if cache_key in _seqlen_to_cfgs_cache:
            seqlen_to_cfgs_mapping[p] = deepcopy(_seqlen_to_cfgs_cache[cache_key])

    # Only do the expensive lookup for new pairs
    if uncached_pairs:
        # Use Ray to parallelize lookups when available and worthwhile
        use_ray = False
        if (
            len(uncached_pairs) > 16
        ):  # a threshold for when parallelization is likely beneficial; can be tuned
            use_ray = True
            # try:
            #     import ray
            #     use_ray = ray.is_initialized()
            # except ImportError:
            #     pass

        if use_ray:
            import ray

            logging.info(
                "Using Ray to parallelize optimal config lookups for %d uncached seqlen pairs.",
                len(uncached_pairs),
            )
            # Pass PYTHONPATH via runtime_env so Ray workers can import neusim
            npusim_home = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            runtime_env = {"env_vars": {"PYTHONPATH": npusim_home}}
            compute_remote = ray.remote(_compute_optimal_configs_for_pair).options(
                runtime_env=runtime_env
            )
            futures = [
                compute_remote.remote(config, iseq, oseq, mode)
                for iseq, oseq in uncached_pairs
            ]
            results = ray.get(futures)
            for (iseq, oseq), opt_cfgs in zip(uncached_pairs, results, strict=False):
                cache_key = (iseq, oseq, mode)
                _seqlen_to_cfgs_cache[cache_key] = opt_cfgs
                seqlen_to_cfgs_mapping[(iseq, oseq)] = deepcopy(opt_cfgs)
        else:
            config = deepcopy(config)
            for input_seqlen, output_seqlen in uncached_pairs:
                config.workload_config.llm_config.input_seqlen = input_seqlen
                config.workload_config.llm_config.output_seqlen = output_seqlen
                opt_cfgs = get_optimal_vPod_config_with_seqlen_fallback(config, mode)
                cache_key = (input_seqlen, output_seqlen, mode)
                _seqlen_to_cfgs_cache[cache_key] = opt_cfgs
                seqlen_to_cfgs_mapping[(input_seqlen, output_seqlen)] = deepcopy(
                    opt_cfgs
                )

    return seqlen_to_cfgs_mapping


def get_seqlen_to_config_mapping(
    seqlen_pairs: Sequence[tuple[int, int]],
    config: NPUFleetConfig,
    mode: str,
) -> dict[tuple[int, int], LLMConfig]:
    """
    Get the mapping from sequence length pairs to optimal vPod configurations.

    Args:
        seqlen_pairs: A list of pairs, where each pair is (input_seq_len, output_seq_len).
        config: The configuration object containing workload parameters.
        mode: A string, either 'prefill' or 'decode'.

    Returns:
        A dictionary mapping each (input_seq_len, output_seq_len) pair to its optimal LLMConfig.
    """
    seqlen_to_cfgs_mapping = get_seqlen_to_configs_mapping(seqlen_pairs, config, mode)
    seqlen_to_cfg_mapping: dict[tuple[int, int], LLMConfig] = {
        seqlen: cfgs[0] for seqlen, cfgs in seqlen_to_cfgs_mapping.items()
    }
    return seqlen_to_cfg_mapping


def get_config_to_seqlen_set_mapping(
    seqlen_pairs: Sequence[tuple[int, int]],
    config: NPUFleetConfig,
    mode: str,
) -> list[tuple[LLMConfig, set[tuple[int, int]]]]:
    """
    Get the mapping from optimal vPod configurations to the set of sequence length pairs they cover.

    Args:
        seqlen_pairs: A list of pairs, where each pair is (input_seq_len, output_seq_len).
        config: The configuration object containing workload parameters.
        mode: A string, either 'prefill' or 'decode'.

    Returns:
        A list of tuples, where each tuple contains an LLMConfig and a list of (input_seq_len, output_seq_len) pairs it covers.
        We need to use a list of tuples instead of a dict since LLMConfig is unhashable.
        However, the returned list guarantees that each LLMConfig only appears once.
    """
    seqlen_to_cfg_mapping = get_seqlen_to_config_mapping(seqlen_pairs, config, mode)

    # Invert the mapping to get config to seqlen pairs
    cfgs_to_seqlen: list[tuple[LLMConfig, set[tuple[int, int]]]] = []
    for seqlen, cfg in seqlen_to_cfg_mapping.items():
        found = False
        for cfg_, seqlens in cfgs_to_seqlen:
            if cfg_ == cfg:
                seqlens.add(seqlen)
                found = True
                break
        if not found:
            cfgs_to_seqlen.append((cfg, {seqlen}))

    return cfgs_to_seqlen


def get_seqlens_for_config(
    seqlen_pairs: Sequence[tuple[int, int]],
    config: LLMConfig,
    fleet_config: NPUFleetConfig,
    mode: str,
) -> set[tuple[int, int]]:
    """
    Get the list of sequence length pairs that a given vPod configuration can handle.

    Args:
        seqlen_pairs: A list of pairs, where each pair is (input_seq_len, output_seq_len).
        config: The LLMConfig for which to find the sequence lengths.
        fleet_config: The NPUFleetConfig containing workload parameters.
        mode: A string, either 'prefill' or 'decode'.

    Returns:
        A list of (input_seq_len, output_seq_len) pairs that the config can handle.
    """
    seqlen_to_cfgs_mapping = get_seqlen_to_configs_mapping(
        seqlen_pairs, fleet_config, mode
    )

    if logging.level_debug():
        logging.debug(
            f"seqlen_to_cfgs_mapping: {[(i, o, [(cfg.name, cfg.num_chips) for cfg in cfgs]) for (i, o), cfgs in seqlen_to_cfgs_mapping.items()]}"
        )

    # Deployed vPod configs deliberately carry dummy sequence lengths because
    # those lengths describe requests rather than vPod identity. Match using the
    # stable hardware/parallelism identity so a deployed 32/32 placeholder still
    # resolves to lookup configs generated for the real request sequence.
    deployed_key = config.get_chip_version_and_parallelism_degree_tuple()
    relevant_seqlens: set[tuple[int, int]] = set()
    for seqlen, cfgs in seqlen_to_cfgs_mapping.items():
        for cfg in cfgs:
            if cfg.get_chip_version_and_parallelism_degree_tuple() == deployed_key:
                relevant_seqlens.add(seqlen)
                break

    return relevant_seqlens


def get_seqlen_range_for_config(
    seqlen_pairs: Sequence[tuple[int, int]],
    config: LLMConfig,
    fleet_config: NPUFleetConfig,
    mode: str,
) -> tuple[int, int]:
    """
    Get the range of sequence lengths that a given vPod configuration can handle.

    Args:
        seqlen_pairs: A list of pairs, where each pair is (input_seq_len, output_seq_len).
        config: The LLMConfig for which to find the sequence length range.
        fleet_config: The NPUFleetConfig containing workload parameters.
        mode: A string, either 'prefill' or 'decode'.

    Returns:
        A tuple representing the range (min_seqlen, max_seqlen) that the config can handle.
        If mode == 'prefill', the range is based on input_seq_len.
        If mode == 'decode', the range is based on input_seq_len + output_seq_len.
        If the config does not map to any sequence lengths, returns (-1, -1).
    """
    if logging.level_debug():
        seqlen_to_cfg_mapping = get_seqlen_to_config_mapping(
            seqlen_pairs, fleet_config, mode
        )
        logging.debug(
            f"seqlen_to_cfg_mapping: {[(i, o, cfg.name, cfg.num_chips) for (i, o), cfg in seqlen_to_cfg_mapping.items()]}"
        )

    # Find all seqlens that map to the given config
    relevant_seqlens = get_seqlens_for_config(seqlen_pairs, config, fleet_config, mode)

    if len(relevant_seqlens) == 0:
        return (-1, -1)

    if mode == "prefill":
        seqlen_values = [s[0] for s in relevant_seqlens]
    elif mode == "decode":
        seqlen_values = [s[0] + s[1] for s in relevant_seqlens]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return (min(seqlen_values), max(seqlen_values))


def find_best_fit_config(
    eff_seqlen: int,
    config_to_seqlen_range: dict[LLMConfig, tuple[int, int]],
) -> LLMConfig | None:
    """Find the closest vPod config for an unmatched request's effective seqlen.

    Args:
        eff_seqlen: The effective sequence length of the request.
            For prefill, this is input_seqlen. For decode, input_seqlen + output_seqlen.
        config_to_seqlen_range: Mapping from LLMConfig to (min_seqlen, max_seqlen) range.
            Ranges of (-1, -1) indicate configs with no mapped seqlens and are skipped.

    Returns:
        The best-fit LLMConfig, or None if no valid config exists.

    Priority:
      1. Covering config: config whose range [min, max] contains eff_seqlen
         (prefer smallest range for tightest fit)
      2. Nearest config: config with smallest distance to eff_seqlen,
         preferring larger configs (that can safely handle the seqlen)
         over smaller ones.
    """
    best_covering_cfg = None
    best_covering_span = float("inf")
    best_nearest_cfg = None
    best_nearest_distance = float("inf")
    best_nearest_is_larger = False

    for cfg, (range_min, range_max) in config_to_seqlen_range.items():
        if range_min == -1:
            continue

        if range_min <= eff_seqlen <= range_max:
            # Covering config - prefer tightest fit (smallest span)
            span = range_max - range_min
            if span < best_covering_span:
                best_covering_span = span
                best_covering_cfg = cfg
        else:
            # Non-covering config - track closest
            distance = min(abs(eff_seqlen - range_min), abs(eff_seqlen - range_max))
            is_larger = range_min > eff_seqlen
            # Prefer larger configs over smaller ones at equal distance
            if distance < best_nearest_distance or (
                distance == best_nearest_distance
                and is_larger
                and not best_nearest_is_larger
            ):
                best_nearest_distance = distance
                best_nearest_cfg = cfg
                best_nearest_is_larger = is_larger

    return best_covering_cfg if best_covering_cfg is not None else best_nearest_cfg
