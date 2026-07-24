from types import SimpleNamespace

import pytest

import neusim.npusim.frontend.query_results_helper_lib as results_lib
from neusim.configs.systems.NPUFleetConfig import NPUFleetConfig
from neusim.fleetsim import cost_model


def test_pipeline_batch_cost_uses_one_stage_service_interval() -> None:
    base = NPUFleetConfig(
        workload_config={
            "static_vpod_allocation": {
                "prefill": {
                    "count": 1, "npu_type": "4", "num_chips": 1, "batch_size": 1
                },
                "decode": {
                    "count": 1, "npu_type": "4", "num_chips": 1, "batch_size": 1
                },
            }
        }
    ).workload_config.llm_config
    config = base.model_copy(
        update={
            "name": "4",
            "num_chips": 8,
            "pipeline_parallelism_degree": 4,
        }
    )
    ops = [
        SimpleNamespace(stats=SimpleNamespace(execution_time_ns=10, count=3)),
        SimpleNamespace(stats=SimpleNamespace(execution_time_ns=20, count=2)),
    ]

    expected_stage_time_ns = 70
    expected = (
        expected_stage_time_ns
        / 1e9
        * results_lib.VERSION_TO_COST[config.name]
        / 3600
        * config.num_chips
    )
    actual = cost_model.pipeline_batch_monetary_cost_dollars(ops, config)

    assert actual == pytest.approx(expected)
    assert actual != pytest.approx(expected * config.pipeline_parallelism_degree)
