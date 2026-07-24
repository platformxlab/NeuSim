"""Ordered batching for independent millisecond-DVFS candidate analyses."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

DEFAULT_MS_CANDIDATE_BATCH_SIZE = 8
"""Candidates evaluated serially inside one millisecond-DVFS Ray task."""

MS_CANDIDATE_BATCH_SIZE_ENV = "DVFS_MS_CANDIDATE_BATCH_SIZE"
ORDERED_RAY_CANDIDATE_BATCH_MODE = "ordered_ray_candidate_batches"
SERIAL_CANDIDATE_MODE = "serial_candidate_loop"


def configured_ms_candidate_batch_size() -> int:
    """Return the validated millisecond candidate batch size."""
    raw = os.environ.get(MS_CANDIDATE_BATCH_SIZE_ENV)
    if raw is None:
        return DEFAULT_MS_CANDIDATE_BATCH_SIZE
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"{MS_CANDIDATE_BATCH_SIZE_ENV} must be a positive integer; got {raw!r}"
        ) from error
    if value < 1:
        raise ValueError(
            f"{MS_CANDIDATE_BATCH_SIZE_ENV} must be a positive integer; got {raw!r}"
        )
    return value


def analyze_operator_energy_batch(
    jobs: Sequence[Any],
    config: Any,
    pg_config: Any,
    dvfs_config: Any,
) -> list[Any]:
    """Evaluate one candidate batch serially in its original order."""
    from neusim.npusim.frontend.power_analysis_lib import analyze_operator_energy

    return [
        analyze_operator_energy(
            job,
            config,
            pg_config,
            dvfs_config,
            False,
            False,
        )
        for job in jobs
    ]


def analyze_operator_energy_candidates(
    jobs: Sequence[Any],
    config: Any,
    pg_config: Any,
    dvfs_config: Any,
    *,
    serial: bool,
    remote_batch: Any = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Evaluate all candidates exactly once and return them in input order.

    Ray preserves the order of a list passed to ``ray.get``. Each batch helper
    also uses a plain left-to-right Python loop, so flattening the completed
    batches reproduces the former candidate-major result order exactly. The
    outer trace tasks reserve zero CPUs; these inner batch tasks retain Ray's
    default one-CPU reservation and therefore preserve nested-Ray scheduling.
    """
    batch_size = configured_ms_candidate_batch_size()
    candidate_count = len(jobs)
    if serial:
        return analyze_operator_energy_batch(
            jobs,
            config,
            pg_config,
            dvfs_config,
        ), {
            "candidate_evaluation_mode": SERIAL_CANDIDATE_MODE,
            "candidate_batch_size": batch_size,
            "candidate_batch_size_env": MS_CANDIDATE_BATCH_SIZE_ENV,
            "candidate_count": candidate_count,
            "submitted_candidate_tasks": 0,
            "candidate_result_order": "input_order",
        }

    import ray

    remote = remote_batch or ray.remote(analyze_operator_energy_batch)
    batches = [
        jobs[start : start + batch_size]
        for start in range(0, candidate_count, batch_size)
    ]
    evaluated_batches = ray.get(
        [remote.remote(batch, config, pg_config, dvfs_config) for batch in batches]
    )
    evaluated = [candidate for batch in evaluated_batches for candidate in batch]
    if len(evaluated) != candidate_count:
        raise RuntimeError(
            "millisecond candidate batching returned "
            f"{len(evaluated)} results for {candidate_count} inputs"
        )
    return evaluated, {
        "candidate_evaluation_mode": ORDERED_RAY_CANDIDATE_BATCH_MODE,
        "candidate_batch_size": batch_size,
        "candidate_batch_size_env": MS_CANDIDATE_BATCH_SIZE_ENV,
        "candidate_count": candidate_count,
        "submitted_candidate_tasks": len(batches),
        "candidate_result_order": "input_order",
    }
