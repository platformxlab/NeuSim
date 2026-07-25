import math
from types import SimpleNamespace

import pytest

import neusim.npusim.frontend.query_results_helper_lib as results_lib
from neusim.configs.systems.NPUFleetConfig import NPUFleetConfig
from neusim.fleetsim import ideal_baseline
from neusim.fleetsim.ideal_baseline import get_request_stats_summary
from neusim.fleetsim.LoadGenerator import LLMRequest


def _completed_request() -> LLMRequest:
    request = LLMRequest(input_seqlen=10, output_seqlen=2, timestamp=0)
    request.mark_prefill_started(0)
    request.mark_prefill_finished(10)
    request.mark_decode_iteration_started(10)
    request.mark_decode_iteration_finished(20, num_iterations=1)
    request.prefill_energy_J = 2.0
    request.decode_energy_per_token_J = [1.0]
    request.prefill_cost_dollars = 0.5
    request.decode_cost_per_token_dollars = [0.25]
    return request


def test_one_request_at_zero_has_infinite_arrival_rate_and_finite_throughput() -> None:
    stats = get_request_stats_summary([_completed_request()])

    assert math.isinf(stats["request_rate_rps"])
    assert stats["throughput_rps"] == pytest.approx(1 / (20 / 1e9))
    assert stats["prefill_throughput_tps"] == pytest.approx(10 / (20 / 1e9))
    assert stats["decode_throughput_tps"] == pytest.approx(1 / (20 / 1e9))
    assert stats["decode_token_per_joule"] == pytest.approx(1.0)
    assert stats["decode_token_per_dollar"] == pytest.approx(4.0)


def test_empty_summary_has_clear_error() -> None:
    with pytest.raises(ValueError, match="No completed requests"):
        get_request_stats_summary([])


def test_ideal_decode_accounts_for_only_tokens_after_first(monkeypatch) -> None:
    config = NPUFleetConfig()
    recommended = config.workload_config.llm_config.model_copy(
        update={
            "name": "4",
            "num_chips": 1,
            "microbatch_size_ici": 1,
            "pipeline_parallelism_degree": 1,
        }
    )
    op = SimpleNamespace(
        stats=SimpleNamespace(execution_time_ns=10, total_energy_J=2.0, count=1)
    )
    monkeypatch.setattr(
        ideal_baseline.autoscaler_lib,
        "get_optimal_vPod_config_with_seqlen_fallback",
        lambda *_args, **_kwargs: [recommended],
    )
    monkeypatch.setattr(
        ideal_baseline.sim_backend,
        "run_inference_request_batch",
        lambda *_args, **_kwargs: ([op], [op]),
    )

    request = LLMRequest(input_seqlen=32, output_seqlen=5, timestamp=0)
    request.mark_prefill_started(0)
    request.mark_prefill_finished(100)
    request.decode_start_timestamp = 100

    ideal_baseline.run_request_prefill_or_decode(request, config, "decode")

    assert request.current_decode_step == 5
    assert request.decode_end_timestamp == 140
    assert request.TPOT_ns() == 10
    assert request.ideal_TPOT_ns == 10
    assert len(request.decode_energy_per_token_J) == 4
    assert len(request.decode_cost_per_token_dollars) == 4
    assert len(request.config_decode_batch_sizes) == 4


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_ideal_pipeline_cost_uses_stage_time_but_latency_uses_all_stages(
    monkeypatch, phase: str
) -> None:
    config = NPUFleetConfig()
    recommended = config.workload_config.llm_config.model_copy(
        update={
            "name": "4",
            "num_chips": 8,
            "microbatch_size_ici": 2,
            "pipeline_parallelism_degree": 4,
        }
    )
    op = SimpleNamespace(
        stats=SimpleNamespace(execution_time_ns=10, total_energy_J=2.0, count=1)
    )
    monkeypatch.setattr(
        ideal_baseline.autoscaler_lib,
        "get_optimal_vPod_config_with_seqlen_fallback",
        lambda *_args, **_kwargs: [recommended],
    )
    monkeypatch.setattr(
        ideal_baseline.sim_backend,
        "run_inference_request_batch",
        lambda *_args, **_kwargs: ([op], [op]),
    )

    request = LLMRequest(input_seqlen=32, output_seqlen=5, timestamp=0)
    request.mark_prefill_started(0)
    request.mark_prefill_finished(100)
    request.decode_start_timestamp = 100

    ideal_baseline.run_request_prefill_or_decode(request, config, phase)

    expected_cost = (
        10
        / 1e9
        * results_lib.VERSION_TO_COST[recommended.name]
        / 3600
        * recommended.num_chips
        / recommended.microbatch_size_ici
    )
    if phase == "prefill":
        assert request.TTFT_ns() == 40
        assert request.prefill_cost_dollars == pytest.approx(expected_cost)
    else:
        assert request.TPOT_ns() == 40
        assert request.decode_cost_per_token_dollars == pytest.approx(
            [expected_cost] * 4
        )
