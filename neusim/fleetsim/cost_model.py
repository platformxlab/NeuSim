"""Shared FleetSim cost-accounting helpers."""

from collections.abc import Sequence

import neusim.npusim.frontend.query_results_helper_lib as results_lib
from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig
from neusim.npusim.frontend.Operator import Operator


def pipeline_batch_monetary_cost_dollars(
    ops: Sequence[Operator], config: LLMConfig | DeepSeekConfig
) -> float:
    """Return steady-state chip-hour cost for one pipeline microbatch.

    A pipelined vPod accepts a new microbatch after one stage service interval,
    even though that microbatch's end-to-end latency spans every pipeline stage.
    Charging every overlapping microbatch for the full end-to-end latency would
    count the same allocated chip time once per stage.  The monetary optimizer
    likewise ranks configurations by stage throughput, so runtime accounting
    must use the same service interval.
    """
    stage_time_ns = sum(op.stats.execution_time_ns * op.stats.count for op in ops)
    return (
        stage_time_ns
        / 1e9
        * results_lib.VERSION_TO_COST[config.name]
        / 3600
        * config.num_chips
    )
