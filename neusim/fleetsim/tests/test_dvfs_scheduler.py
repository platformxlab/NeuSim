"""Regression tests for static FleetSim service-level DVFS."""

import json
from collections import deque
from types import SimpleNamespace

import pytest

from neusim.fleetsim import dvfs_scheduler
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.fleetsim.npusim_backend_interface import FrozenLLMConfig, PhaseMetrics


@pytest.fixture(autouse=True)
def _clean_lookup_cache():
    dvfs_scheduler.reset_dvfs_lookup_cache()
    yield
    dvfs_scheduler.reset_dvfs_lookup_cache()


def _write_grouped_cache(tmp_path):
    cache_file = tmp_path / "llama3-70b" / "32_16" / "5p" / "prefill" / "bs1.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        json.dumps(
            {
                "metadata": {
                    "model": "llama3-70b",
                    "input_seqlen": 32,
                    "output_seqlen": 16,
                    "version": "5p",
                    "phase": "prefill",
                    "batch_size": 1,
                },
                # Deliberately unsorted: the loader normalizes this once.
                "points": {
                    "0.2": {
                        "time_ns_per_stage": 120,
                        "energy_J_per_chip": 7.0,
                    },
                    "0.0": {
                        "time_ns_per_stage": 100,
                        "energy_J_per_chip": 10.0,
                    },
                    "0.1": {
                        "time_ns_per_stage": 105,
                        "energy_J_per_chip": 8.0,
                    },
                },
            }
        )
    )


def test_grouped_cache_is_sorted_once_and_selects_feasible_minimum(tmp_path) -> None:
    _write_grouped_cache(tmp_path)
    assert dvfs_scheduler.load_dvfs_lookup_cache(str(tmp_path), strict=True) == 3

    result = dvfs_scheduler._lookup_dvfs_energy(
        "llama3-70b", 32, 16, "5p", "prefill", 1, 0.2, 1, 110
    )

    assert result == (105, 8.0, 0.1)
    assert dvfs_scheduler.get_dvfs_lookup_stats() == {
        "entries": 1,
        "points": 3,
        "hits": 1,
        "misses": 0,
        "plans_applied": 0,
        "plans_rejected_nonbeneficial": 0,
    }


def test_strict_cache_load_and_runtime_miss_are_fatal(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        dvfs_scheduler.load_dvfs_lookup_cache(str(tmp_path / "missing"), strict=True)

    request = LLMRequest(32, 16, timestamp=0)
    frozen = FrozenLLMConfig(
        model_name="llama3-70b",
        name="5p",
        input_seqlen=32,
        output_seqlen=16,
        global_batch_size=1,
        microbatch_size_ici=1,
        microbatch_size_dcn=1,
    )
    metrics_server = SimpleNamespace(
        slo_targets=[object()],
        dvfs_locked_to_peak=False,
        assign_ttft_target=lambda _seqlen: 1.0,
    )
    workload = SimpleNamespace(
        dvfs_max_perf_degrad=1.0,
        dvfs_require_cache_hit=True,
        dvfs_policy="DVFSC",
    )

    with pytest.raises(KeyError, match="cache point is missing"):
        dvfs_scheduler.compute_dvfs_plan_for_batch(
            phase_metrics=PhaseMetrics(100, 10.0),
            frozen_config=frozen,
            requests=[request],
            event_timestamp=0,
            baseline_time_ns=100,
            num_pipeline_stages=1,
            num_chips=1,
            prefill_or_decode="prefill",
            num_iterations=1,
            metrics_server=metrics_server,
            request_queue=deque(),
            workload_config=workload,
        )
    assert dvfs_scheduler.get_dvfs_lookup_stats()["misses"] == 1


def test_prefill_target_and_queue_safeguard_match_original_algorithm() -> None:
    active = LLMRequest(32, 16, timestamp=100)
    queued = LLMRequest(32, 16, timestamp=0)
    target = dvfs_scheduler.compute_t_target_prefill([active], 200, lambda _: 1e-6)
    assert target == 900
    assert (
        dvfs_scheduler.apply_safeguard1(
            target, 100, deque([queued]), 200, lambda _: 0.5e-6, "prefill", 1e9
        )
        == 200
    )
