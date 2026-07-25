import math
from types import SimpleNamespace

import pytest

from neusim.configs.models.LLMConfig import LLMConfig
from neusim.fleetsim import npusim_backend_interface as backend
from neusim.fleetsim.LoadGenerator import LLMRequest


def test_batch_padding_fields_copy_and_cached_result_isolation(monkeypatch) -> None:
    captured = []
    shared_result = ([{"phase": "prefill"}], [{"phase": "decode"}])

    def fake_cached(config):
        captured.append(config)
        return shared_result

    monkeypatch.setattr(
        backend, "run_llm_inference_request_seqlen_cached", fake_cached
    )
    config = LLMConfig(
        input_seqlen=11,
        output_seqlen=7,
        global_batch_size=9,
        microbatch_size_ici=9,
        microbatch_size_dcn=9,
    )
    requests = [LLMRequest(33, 9), LLMRequest(37, 13)]

    first = backend.run_inference_request_batch(
        config,
        requests,
        input_padding_factors=[8, 16],
        input_padding_steps=[64],
        output_padding_factors=[4, 8],
        output_padding_steps=[16],
    )
    frozen = captured[-1]
    assert frozen.input_seqlen == 40
    assert frozen.output_seqlen == 16
    assert frozen.global_batch_size == 2
    assert frozen.microbatch_size_ici == 2
    assert frozen.microbatch_size_dcn == 2
    assert config.input_seqlen == 11
    assert config.output_seqlen == 7
    assert config.global_batch_size == 9

    first[0][0]["phase"] = "mutated"
    second = backend.run_inference_request_batch(config, requests)
    assert second[0][0]["phase"] == "prefill"
    assert shared_result[0][0]["phase"] == "prefill"


def test_empty_batch_has_clear_error() -> None:
    with pytest.raises(ValueError, match="at least one"):
        backend.run_inference_request_batch(LLMConfig(), [])


def test_disk_cache_is_opt_in(tmp_path) -> None:
    backend.set_npusim_backend_cache_dir(None)
    assert backend.location is None
    assert backend.memory.location is None

    cache_dir = tmp_path / "npusim-cache"
    backend.set_npusim_backend_cache_dir(cache_dir)
    assert backend.location == str(cache_dir)
    assert backend.memory.location == str(cache_dir)

    backend.set_npusim_backend_cache_dir(None)


def test_decode_counts_are_normalized_and_must_be_divisible(monkeypatch) -> None:
    valid_prefill = [SimpleNamespace(stats=SimpleNamespace(count=1))]
    valid_decode = [
        SimpleNamespace(stats=SimpleNamespace(count=8)),
        SimpleNamespace(stats=SimpleNamespace(count=16)),
    ]

    class FakeGenerator:
        def __init__(self, config):
            self.config = config

        def generate(self, **kwargs):
            return valid_prefill + valid_decode, valid_prefill, valid_decode

    monkeypatch.setattr(backend, "LLMOpsGenerator", FakeGenerator)
    frozen = backend.FrozenLLMConfig(output_seqlen=8)
    prefill, decode = backend.run_llm_inference_request_seqlen(frozen)
    assert prefill is valid_prefill
    assert [op.stats.count for op in decode] == [1, 2]

    invalid_decode = [SimpleNamespace(stats=SimpleNamespace(count=9))]

    class InvalidGenerator(FakeGenerator):
        def generate(self, **kwargs):
            return invalid_decode, [], invalid_decode

    monkeypatch.setattr(backend, "LLMOpsGenerator", InvalidGenerator)
    with pytest.raises(ValueError, match="positive multiple"):
        backend.run_llm_inference_request_seqlen(frozen)


def test_real_neusim_backend_returns_analyzed_operators(capsys) -> None:
    """Exercise the actual FleetSim -> neusim.npusim frontend/backend path."""
    config = LLMConfig(
        model_name="tiny-llm",
        input_seqlen=32,
        output_seqlen=4,
        d_model=128,
        num_heads=4,
        num_kv_heads=4,
        d_head=32,
        d_ff=256,
        num_layers=1,
        global_batch_size=1,
        microbatch_size_ici=1,
        microbatch_size_dcn=1,
    )

    prefill, decode = backend.run_inference_request_batch(
        config, [LLMRequest(input_seqlen=32, output_seqlen=4)]
    )

    assert prefill
    assert decode
    for op in prefill + decode:
        assert math.isfinite(op.stats.execution_time_ns)
        assert op.stats.execution_time_ns > 0
        assert math.isfinite(op.stats.total_energy_J)
        assert op.stats.total_energy_J > 0
        assert isinstance(op.stats.count, int)
        assert op.stats.count > 0
    assert capsys.readouterr().out == ""
