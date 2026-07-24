"""FleetSim adapter for NeuSim's NPU simulation frontend."""

import time as _time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from absl import logging
from joblib import Memory
from pydantic import BaseModel, ConfigDict

from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig
from neusim.fleetsim.LoadGenerator import LLMRequest
from neusim.fleetsim.util import pad_seqlen
from neusim.npusim.frontend.llm_ops_generator import (
    DeepSeekOpsGenerator,
    LLMOpsGenerator,
)
from neusim.npusim.frontend.Operator import Operator

enable_profile = False
"""Whether to report unusually slow backend cache misses."""


@dataclass(frozen=True, slots=True)
class PhaseMetrics:
    """Read-only per-chip totals for one prefill or decode service interval."""

    stage_time_ns: int
    energy_per_chip_J: float


@dataclass(frozen=True, slots=True)
class InferenceBatchMetrics:
    """Cached phase totals consumed by FleetSim's static scheduler."""

    prefill: PhaseMetrics
    decode: PhaseMetrics


def _summarize_phase(ops: Sequence[Operator]) -> PhaseMetrics:
    return PhaseMetrics(
        stage_time_ns=sum(op.stats.execution_time_ns * op.stats.count for op in ops),
        energy_per_chip_J=sum(op.stats.total_energy_J * op.stats.count for op in ops),
    )


def set_enable_profile(value: bool) -> None:
    global enable_profile
    enable_profile = bool(value)


def _as_hashable(value: Any) -> Any:
    """Recursively convert Pydantic data to an all-field hash key."""
    if isinstance(value, BaseModel):
        return _as_hashable(value.model_dump(mode="python"))
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (
                    (_as_hashable(key), _as_hashable(item))
                    for key, item in value.items()
                ),
                key=repr,
            )
        )
    if isinstance(value, list | tuple):
        return tuple(_as_hashable(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(sorted((_as_hashable(item) for item in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


class FrozenLLMConfig(LLMConfig):
    """Immutable, all-field-hashable LLM configuration used as a cache key."""

    model_config = ConfigDict(frozen=True)
    __eq__ = BaseModel.__eq__

    def __hash__(self) -> int:
        return hash(_as_hashable(self.model_dump(mode="python")))


class FrozenDeepSeekConfig(DeepSeekConfig):
    """Immutable, all-field-hashable DeepSeek configuration used as a cache key."""

    model_config = ConfigDict(frozen=True)
    __eq__ = BaseModel.__eq__

    def __hash__(self) -> int:
        return hash(_as_hashable(self.model_dump(mode="python")))


def make_frozen_config(
    config: LLMConfig | DeepSeekConfig,
) -> FrozenLLMConfig | FrozenDeepSeekConfig:
    """Return the immutable config type used by the backend cache."""
    if isinstance(config, DeepSeekConfig):
        return FrozenDeepSeekConfig(**config.model_dump())
    return FrozenLLMConfig(**config.model_dump())


def _config_for_batch(
    config: LLMConfig | DeepSeekConfig,
    requests: Sequence[LLMRequest],
    input_padding_factors: Sequence[int],
    input_padding_steps: Sequence[int],
    output_padding_factors: Sequence[int],
    output_padding_steps: Sequence[int],
) -> LLMConfig | DeepSeekConfig:
    batch = tuple(requests)
    if not batch:
        raise ValueError("requests must contain at least one LLMRequest")

    backend_config = config.model_copy(deep=True)
    backend_config.input_seqlen = pad_seqlen(
        max(request.input_seqlen for request in batch),
        input_padding_factors,
        input_padding_steps,
    )
    backend_config.output_seqlen = pad_seqlen(
        max(request.output_seqlen for request in batch),
        output_padding_factors,
        output_padding_steps,
    )
    backend_config.global_batch_size = len(batch)
    backend_config.microbatch_size_ici = len(batch)
    backend_config.microbatch_size_dcn = len(batch)
    return backend_config


def make_frozen_config_for_batch(
    config: LLMConfig | DeepSeekConfig,
    requests: Sequence[LLMRequest],
    input_padding_factors: Sequence[int] = (
        32,
        64,
        128,
        512,
        1024,
        4096,
        8192,
        16384,
        32768,
    ),
    input_padding_steps: Sequence[int] = (
        128,
        512,
        1024,
        8192,
        16384,
        65536,
        131072,
        262144,
    ),
    output_padding_factors: Sequence[int] = (4, 16, 32, 64, 128, 256, 512, 1024),
    output_padding_steps: Sequence[int] = (32, 64, 128, 512, 1024, 2048, 8192),
) -> FrozenLLMConfig | FrozenDeepSeekConfig:
    """Build exactly the frozen cache key used for a padded request batch."""
    return make_frozen_config(
        _config_for_batch(
            config,
            requests,
            input_padding_factors,
            input_padding_steps,
            output_padding_factors,
            output_padding_steps,
        )
    )


def run_inference_request_batch(
    config: LLMConfig | DeepSeekConfig,
    requests: Sequence[LLMRequest],
    input_padding_factors: Sequence[int] = (
        32,
        64,
        128,
        512,
        1024,
        4096,
        8192,
        16384,
        32768,
    ),
    input_padding_steps: Sequence[int] = (
        128,
        512,
        1024,
        8192,
        16384,
        65536,
        131072,
        262144,
    ),
    output_padding_factors: Sequence[int] = (4, 16, 32, 64, 128, 256, 512, 1024),
    output_padding_steps: Sequence[int] = (32, 64, 128, 512, 1024, 2048, 8192),
) -> InferenceBatchMetrics:
    """Return immutable per-chip phase totals for one padded request batch."""
    batch = tuple(requests)
    backend_config = _config_for_batch(
        config,
        batch,
        input_padding_factors,
        input_padding_steps,
        output_padding_factors,
        output_padding_steps,
    )
    frozen_config = make_frozen_config(backend_config)

    if logging.level_debug():
        logging.debug(
            "Running inference request batch with config: %s\nrequests: %s",
            backend_config,
            batch,
        )

    start = _time.perf_counter() if enable_profile else 0.0
    result = run_llm_inference_request_seqlen_cached(frozen_config)
    if enable_profile:
        elapsed = _time.perf_counter() - start
        if elapsed > 1.0:
            logging.warning(
                "Slow NeuSim backend call (%.1fs, likely cache miss): "
                "model=%s, chips=%d, iseq=%d, oseq=%d, bs=%d",
                elapsed,
                backend_config.name,
                backend_config.num_chips,
                backend_config.input_seqlen,
                backend_config.output_seqlen,
                backend_config.global_batch_size,
            )

    return result


def run_llm_inference_request_seqlen(
    frozen_config: FrozenLLMConfig | FrozenDeepSeekConfig,
) -> InferenceBatchMetrics:
    """Generate operators once and cache only immutable phase totals."""
    if isinstance(frozen_config, FrozenDeepSeekConfig):
        config: LLMConfig | DeepSeekConfig = DeepSeekConfig(
            **frozen_config.model_dump()
        )
        ops_generator = DeepSeekOpsGenerator(config)
    else:
        config = LLMConfig(**frozen_config.model_dump())
        ops_generator = LLMOpsGenerator(config)

    generated = ops_generator.generate(
        dump_to_file=False,
        separate_prefill_decode=True,
        analyze_energy=True,
    )
    if not isinstance(generated, tuple) or len(generated) != 3:
        raise TypeError("NeuSim LLM generator did not return split operator lists")
    _, prefill_ops, decode_ops = generated
    if not isinstance(prefill_ops, list) or not isinstance(decode_ops, list):
        raise TypeError("NeuSim LLM generator returned non-list operator collections")

    for op in decode_ops:
        count = op.stats.count
        if count < config.output_seqlen or count % config.output_seqlen:
            raise ValueError(
                f"decode op count {count} must be a positive multiple of "
                f"output_seqlen {config.output_seqlen}; op={op}"
            )
        op.stats.count = count // config.output_seqlen

    return InferenceBatchMetrics(
        prefill=_summarize_phase(prefill_ops),
        decode=_summarize_phase(decode_ops),
    )


location: str | None = None
"""Current joblib cache directory; ``None`` means import-time memory-only mode."""

memory = Memory(location=None, verbose=0)


def _build_cached_backend(*args, **kwargs):
    disk_cached = memory.cache(run_llm_inference_request_seqlen, *args, **kwargs)
    return disk_cached, lru_cache(maxsize=None)(disk_cached)


(
    run_llm_inference_request_seqlen_cached_disk,
    run_llm_inference_request_seqlen_cached,
) = _build_cached_backend()


def set_npusim_backend_cache_dir(cache_dir: str | Path | None, *args, **kwargs) -> None:
    """Replace the cache stack, enabling disk caching only for a nonempty path."""
    global location
    global memory
    global run_llm_inference_request_seqlen_cached_disk
    global run_llm_inference_request_seqlen_cached

    old_cached = run_llm_inference_request_seqlen_cached
    old_cached.cache_clear()

    location = str(Path(cache_dir).expanduser()) if cache_dir else None
    memory = Memory(location=location, verbose=0)
    (
        run_llm_inference_request_seqlen_cached_disk,
        run_llm_inference_request_seqlen_cached,
    ) = _build_cached_backend(*args, **kwargs)
