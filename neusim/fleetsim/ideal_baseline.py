### Get results for the Ideal Baseline without performing event-driven simulation
### as this is unnecessary and takes long to simulate frequent ideal auto-scaling.


import csv
import json
import os
from copy import deepcopy
from functools import partial

import numpy as np
import tqdm
from absl import logging
from parallelbar import progress_starmap

import neusim.fleetsim.npusim_backend_interface as sim_backend
import neusim.fleetsim.util as util
from neusim.configs.systems.NPUFleetConfig import NPUFleetConfig
from neusim.fleetsim import cost_model
from neusim.fleetsim import vPodAutoScaler_lib as autoscaler_lib
from neusim.fleetsim.LoadGenerator import LLMRequest, LoadGenerator


def run_request_prefill_or_decode(
    req: LLMRequest, config: NPUFleetConfig, prefill_or_decode: str
):
    config = deepcopy(config)
    config.workload_config.llm_config.input_seqlen = autoscaler_lib.pad_seqlen(
        req.input_seqlen,
        config.workload_config.input_seqlen_padding_factors,
        config.workload_config.input_seqlen_padding_steps,
    )
    config.workload_config.llm_config.output_seqlen = autoscaler_lib.pad_seqlen(
        req.output_seqlen if prefill_or_decode == "decode" else 4,
        config.workload_config.output_seqlen_padding_factors,
        config.workload_config.output_seqlen_padding_steps,
    )
    optimal_configs = autoscaler_lib.get_optimal_vPod_config_with_seqlen_fallback(
        config, prefill_or_decode
    )
    if len(optimal_configs) == 0:
        raise ValueError(
            f"No valid config for {config.workload_config.model_name} "
            f"seqlen={config.workload_config.llm_config.input_seqlen}_{config.workload_config.llm_config.output_seqlen} "
            f"{prefill_or_decode}; this seqlen may exceed HBM capacity for all chip versions."
        )
    recommended_config = optimal_configs[0]

    requests = [
        deepcopy(req)
        for _ in range(
            recommended_config.microbatch_size_ici
            if prefill_or_decode == "prefill"
            else (
                min(
                    # limit batch size for decode for now to match with the trace
                    # since we do not have sufficient requests to batch
                    recommended_config.microbatch_size_ici,
                    config.workload_config.max_decode_batch_size,
                )
                if config.workload_config.max_decode_batch_size != -1
                else recommended_config.microbatch_size_ici
            )
        )
    ]

    # invoke backend
    prefill_ops, decode_ops = sim_backend.run_inference_request_batch(
        recommended_config,
        requests,
    )
    real_batch_size = len(requests)

    # fill out the stats
    if prefill_or_decode == "prefill":
        prefill_time = (
            sum(op.stats.execution_time_ns * op.stats.count for op in prefill_ops)
            * recommended_config.pipeline_parallelism_degree
        )
        prefill_energy = (
            sum(op.stats.total_energy_J * op.stats.count for op in prefill_ops)
            * recommended_config.num_chips
        ) / real_batch_size
        prefill_cost_dollars = (
            cost_model.pipeline_batch_monetary_cost_dollars(
                prefill_ops, recommended_config
            )
            / real_batch_size
        )

        req.current_decode_step = 1
        req.prefill_start_timestamp = req.enqueue_timestamp
        req.prefill_end_timestamp = req.prefill_start_timestamp + prefill_time
        req.decode_start_timestamp = req.prefill_end_timestamp
        req.ideal_TTFT_ns = req.TTFT_ns()
        req.prefill_energy_J = prefill_energy
        req.prefill_cost_dollars = prefill_cost_dollars
        req.config_prefill_batch_size = real_batch_size
        req.config_prefill_input_seqlen = recommended_config.input_seqlen
        req.config_prefill_npu_type = recommended_config.name
        req.config_prefill_num_chips = recommended_config.num_chips
        req.config_prefill_pcfg = util.get_pstr(recommended_config)

    elif prefill_or_decode == "decode":
        num_decode_iterations = req.output_seqlen - 1
        decode_time = (  # this is per decode iteration time
            sum(op.stats.execution_time_ns * op.stats.count for op in decode_ops)
            * recommended_config.pipeline_parallelism_degree
        )
        decode_energy_per_token = (
            sum(op.stats.total_energy_J * op.stats.count for op in decode_ops)
            * recommended_config.num_chips
        ) / real_batch_size
        decode_cost_dollars_per_token = (
            cost_model.pipeline_batch_monetary_cost_dollars(
                decode_ops, recommended_config
            )
            / real_batch_size
        )
        req.current_decode_step = req.output_seqlen
        req.decode_end_timestamp = (
            req.decode_start_timestamp + decode_time * num_decode_iterations
        )
        req.ideal_TPOT_ns = req.TPOT_ns()
        req.decode_energy_per_token_J = [
            decode_energy_per_token
        ] * num_decode_iterations
        req.decode_cost_per_token_dollars = [
            decode_cost_dollars_per_token
        ] * num_decode_iterations
        req.config_decode_batch_sizes = [real_batch_size] * num_decode_iterations
        # req.config_decode_input_seqlens = [recommended_config.input_seqlen] * num_decode_iterations
        # req.config_decode_output_seqlens = [recommended_config.output_seqlen] * num_decode_iterations
        req.config_decode_npu_types = [recommended_config.name] * num_decode_iterations
        req.config_decode_num_chips = [
            recommended_config.num_chips
        ] * num_decode_iterations
        req.config_decode_pcfgs = [
            util.get_pstr(recommended_config)
        ] * num_decode_iterations

    else:
        raise ValueError("Unknown prefill_or_decode value.")


def run_request(req: LLMRequest, config: NPUFleetConfig) -> LLMRequest | None:
    try:
        run_request_prefill_or_decode(req, config, "prefill")
        run_request_prefill_or_decode(req, config, "decode")
    except Exception as e:
        logging.warning(
            f"Error occurred while running request: {e}. Skipping request id {req.id} for now."
        )
        return None
    return deepcopy(req)


def get_request_stats_summary(request_trace: list[LLMRequest]) -> dict:
    """
    Return a summary of the request trace statistics in a dictionary.
    """
    if not request_trace:
        raise ValueError(
            "No completed requests are available for the Ideal baseline summary."
        )

    stats = {}
    stats["total_requests"] = len(request_trace)

    # min, max, average of input sequence length
    input_seqlens = [request.input_seqlen for request in request_trace]
    stats["input_seqlen_min"] = min(input_seqlens)
    stats["input_seqlen_max"] = max(input_seqlens)
    stats["input_seqlen_avg"] = np.mean(input_seqlens)

    # min, max, average of output sequence length
    output_seqlens = [request.output_seqlen for request in request_trace]
    decode_token_counts = [output_seqlen - 1 for output_seqlen in output_seqlens]
    stats["output_seqlen_min"] = min(output_seqlens)
    stats["output_seqlen_max"] = max(output_seqlens)
    stats["output_seqlen_avg"] = np.mean(output_seqlens)

    # min, max, average of TTFT in seconds
    TTFTs = [request.TTFT_ns() for request in request_trace]
    stats["TTFT_min_s"] = np.min(TTFTs) / 1e9
    stats["TTFT_max_s"] = np.max(TTFTs) / 1e9
    stats["TTFT_avg_s"] = np.mean(TTFTs) / 1e9

    # min, max, average of TPOT in seconds
    TPOTs = [request.TPOT_ns() for request in request_trace]
    stats["TPOT_min_s"] = np.min(TPOTs) / 1e9
    stats["TPOT_max_s"] = np.max(TPOTs) / 1e9
    stats["TPOT_avg_s"] = np.mean(TPOTs) / 1e9

    # request rate in requests per second
    first_enqueue_time = min(req.enqueue_timestamp for req in request_trace)
    last_enqueue_time = max(req.enqueue_timestamp for req in request_trace)
    arrival_window_ns = last_enqueue_time - first_enqueue_time
    stats["request_rate_rps"] = (
        len(request_trace) / (arrival_window_ns / 1e9)
        if arrival_window_ns > 0
        else float("inf")
    )

    # Throughput includes request execution time, unlike the arrival rate.
    last_completion_time = max(req.decode_end_timestamp for req in request_trace)
    completion_window_ns = last_completion_time - first_enqueue_time
    if completion_window_ns <= 0:
        raise ValueError(
            "Completed requests must have decode_end_timestamp after the first enqueue."
        )
    completion_window_s = completion_window_ns / 1e9
    stats["throughput_rps"] = len(request_trace) / completion_window_s
    # prefill throughput in tokens per second
    stats["prefill_throughput_tps"] = sum(input_seqlens) / completion_window_s
    # Decode iterations produce tokens 2..N; token 1 is emitted by prefill.
    stats["decode_throughput_tps"] = sum(decode_token_counts) / completion_window_s

    # cost efficiencies
    stats["prefill_token_per_joule"] = sum(input_seqlens) / sum(
        req.prefill_energy_J for req in request_trace
    )
    stats["decode_token_per_joule"] = sum(decode_token_counts) / sum(
        sum(req.decode_energy_per_token_J) for req in request_trace
    )
    stats["total_token_per_joule"] = (sum(input_seqlens) + sum(output_seqlens)) / (
        sum(req.prefill_energy_J for req in request_trace)
        + sum(sum(req.decode_energy_per_token_J) for req in request_trace)
    )
    stats["prefill_token_per_dollar"] = sum(input_seqlens) / sum(
        req.prefill_cost_dollars for req in request_trace
    )
    stats["decode_token_per_dollar"] = sum(decode_token_counts) / sum(
        sum(req.decode_cost_per_token_dollars) for req in request_trace
    )
    stats["total_token_per_dollar"] = (sum(input_seqlens) + sum(output_seqlens)) / (
        sum(req.prefill_cost_dollars for req in request_trace)
        + sum(sum(req.decode_cost_per_token_dollars) for req in request_trace)
    )

    return stats


def run(sim_config: NPUFleetConfig, name: str, n_cpu: int | None = None):
    """Entry function. Call this with the config to simulate the Ideal baseline."""
    logging.info("Running Ideal simulation without eventsim...")

    loadgen = LoadGenerator(sim_config.workload_config)
    requests = loadgen.generate()
    # truncate requests based on max_num_requests and max_timestamp
    if sim_config.workload_config.max_num_requests != -1:
        requests = requests[: sim_config.workload_config.max_num_requests]
    if sim_config.workload_config.max_timestamp != -1:
        requests = [
            req
            for req in requests
            if req.enqueue_timestamp <= sim_config.workload_config.max_timestamp
        ]

    # run sim
    if sim_config.tqdm:
        requests_iter = tqdm.tqdm(requests)
    else:
        requests_iter = requests
    # final_reqs: list[LLMRequest] = []
    # for req in requests_iter:
    #     result = run_request(req, sim_config)
    #     if result is not None:
    #         final_reqs.append(result)
    final_reqs = progress_starmap(
        func=partial(run_request, config=sim_config),
        tasks=[(req,) for req in requests_iter],
        n_cpu=n_cpu or os.cpu_count(),
    )

    # filter out None results due to errors
    final_reqs_filtered: list[LLMRequest] = [
        req for req in final_reqs if req is not None
    ]  # type: ignore

    logging.info(f"{name}: Ideal simulation complete.")

    # dump stats
    os.makedirs(sim_config.output_dir, exist_ok=True)

    requests = final_reqs_filtered
    stats = get_request_stats_summary(requests)
    logging.info("Request trace summary:")
    logging.info(json.dumps(stats, indent=4))

    # dump to JSON file
    output_filepath = os.path.join(sim_config.output_dir, "stats.json")
    with open(output_filepath, "w") as f:
        json.dump(stats, f, indent=4)
    logging.info("Simulation stats summary dumped to %s", output_filepath)

    # dump request trace to CSV file
    filepath = os.path.join(sim_config.output_dir, "request_trace.csv")
    with open(filepath, "w") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(
            [
                "request_id",
                "input_seqlen",
                "output_seqlen",
                "enqueue_timestamp",
                "prefill_start_timestamp",
                "prefill_end_timestamp",
                "decode_start_timestamp",
                "decode_end_timestamp",
                "prefill_queuing_delay_ns",
                "prefill_latency_ns",
                "TTFT_ns",
                "decode_queuing_delay_per_iteration_ns",
                "TPOT_ns",
                "ideal_TTFT_ns",
                "ideal_TPOT_ns",
                "prefill_energy_J",
                "decode_energy_per_token_J",
                "prefill_cost_dollars",
                "decode_cost_per_token_dollars",
                "config_prefill_npu_type",
                "config_prefill_input_seqlen",
                "config_prefill_batch_size",
                "config_prefill_num_chips",
                "config_prefill_pcfg",
                "config_decode_npu_types",
                "config_decode_input_seqlens",
                "config_decode_output_seqlens",
                "config_decode_batch_sizes",
                "config_decode_num_chips",
                "config_decode_pcfgs",
            ]
        )
        # Write data
        for request in requests:
            # compute decode energy per token
            total_energy = sum(request.decode_energy_per_token_J)
            num_decode_iterations = request.output_seqlen - 1
            decode_energy_per_token_J = total_energy / num_decode_iterations
            total_cost = sum(request.decode_cost_per_token_dollars)
            decode_cost_per_token_dollars = total_cost / num_decode_iterations

            writer.writerow(
                [
                    request.id,
                    request.input_seqlen,
                    request.output_seqlen,
                    request.enqueue_timestamp,
                    request.prefill_start_timestamp,
                    request.prefill_end_timestamp,
                    request.decode_start_timestamp,
                    request.decode_end_timestamp,
                    request.prefill_queuing_delay_ns(),
                    request.prefill_latency_ns(),
                    request.TTFT_ns(),
                    request.decode_queuing_delay_per_iteration_ns(),
                    request.TPOT_ns(),
                    request.ideal_TTFT_ns,
                    request.ideal_TPOT_ns,
                    request.prefill_energy_J,
                    decode_energy_per_token_J,
                    request.prefill_cost_dollars,
                    decode_cost_per_token_dollars,
                    request.config_prefill_npu_type,
                    request.config_prefill_input_seqlen,
                    request.config_prefill_batch_size,
                    request.config_prefill_num_chips,
                    request.config_prefill_pcfg,
                    "/".join(request.config_decode_npu_types),
                    "/".join(map(str, request.config_decode_input_seqlens)),
                    "/".join(map(str, request.config_decode_output_seqlens)),
                    "/".join(map(str, request.config_decode_batch_sizes)),
                    "/".join(map(str, request.config_decode_num_chips)),
                    "/".join(request.config_decode_pcfgs),
                ]
            )
        logging.info("Request trace dumped to %s", filepath)
