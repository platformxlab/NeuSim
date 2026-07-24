### Utility functions.

from collections.abc import Sequence

from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig


def get_pstr(config: LLMConfig | DeepSeekConfig) -> str:
    """
    Get the performance string for the given LLM configuration.
    """
    dp = config.data_parallelism_degree
    tp = config.tensor_parallelism_degree
    pp = config.pipeline_parallelism_degree
    ppdcn = config.pipeline_parallel_degree_dcn
    bs = config.microbatch_size_ici
    pstr = f"bs{bs}-dp{dp}-tp{tp}-pp{pp}"
    if isinstance(config, DeepSeekConfig):
        ep = config.expert_parallelism_degree
        pstr += f"-ep{ep}"
    if ppdcn > 1:
        pstr += f"-ppdcn{ppdcn}"
    return pstr


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
