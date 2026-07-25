from copy import deepcopy

from neusim.configs.models.LLMConfig import LLMConfig
from neusim.fleetsim.vPodAutoScaler import MultiPoolAutoScaler


def test_pool_routing_ignores_deployed_dummy_sequence_lengths() -> None:
    candidate = LLMConfig(
        name="6e",
        input_seqlen=2560,
        output_seqlen=4,
        num_chips=32,
        tensor_parallelism_degree=8,
        pipeline_parallelism_degree=4,
    )
    deployed = deepcopy(candidate)
    deployed.input_seqlen = 32
    deployed.output_seqlen = 32

    autoscaler = object.__new__(MultiPoolAutoScaler)
    autoscaler.prefill_or_decode = "prefill"
    autoscaler.boundaries = [2560.0]
    autoscaler.pool_configs = [[candidate]]

    assert candidate != deployed
    assert autoscaler.get_seqlens_for_config([(2560, 4)], deployed) == {(2560, 4)}
