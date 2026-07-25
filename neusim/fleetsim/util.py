### Utility functions.

import math
from collections.abc import Sequence

import numpy as np

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


def num_chips_to_shape_2D(num_chips: int) -> list[int]:
    """
    Convert number of chips to a 2D shape (dimX, dimY).
    The shape is chosen to be as square as possible.
    """
    dimX = math.isqrt(num_chips)
    while num_chips % dimX != 0:
        dimX -= 1
    dimY = num_chips // dimX
    return [dimX, dimY]


def num_chips_to_shape_3D(num_chips: int) -> list[int]:
    """
    Convert number of chips to a 3D shape (dimX, dimY, dimZ).
    The shape is chosen to be as cube-like as possible.
    Examples:
        1 -> [1, 1, 1];
        4 -> [1, 2, 2];
        4096 -> [16, 16, 16];
    """
    dimX = int(math.cbrt(num_chips))
    while num_chips % dimX != 0:
        dimX -= 1
    num_chips //= dimX
    dimY = int(math.sqrt(num_chips))
    while num_chips % dimY != 0:
        dimY -= 1
    dimZ = num_chips // dimY
    return [dimX, dimY, dimZ]


def shape_to_num_chips(shape: Sequence[int]) -> int:
    """
    Convert a shape to the number of chips.
    """
    return math.prod(shape)


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


def sample_from_list[_T](my_list: Sequence[_T], n_samples: int) -> list[_T]:
    """
    Sample n_samples elements from my_list uniformly.
    """
    indices = np.linspace(0, len(my_list) - 1, n_samples, dtype=int)
    sampled_list = [my_list[i] for i in indices]
    return sampled_list


class ListMap[_KeyType, _ValueType]:
    """
    A map that is implemented using a list of key-value pairs.
    This is useful when the key is unhashable (e.g., a Pydantic BaseModel that is not frozen).
    Be aware that the lookup time is O(n).
    """

    def __init__(self, pairs: Sequence[tuple[_KeyType, _ValueType]] | None = None):
        self._data: list[tuple[_KeyType, _ValueType]] = list(pairs) if pairs else []

    def __getitem__(self, key: _KeyType) -> _ValueType:
        for k, v in self._data:
            if k == key:
                return v
        raise KeyError(f"Key {key} not found.")

    def __setitem__(self, key: _KeyType, value: _ValueType):
        for i, (k, _value) in enumerate(self._data):
            if k == key:
                self._data[i] = (key, value)
                return
        self._data.append((key, value))

    def __delitem__(self, key: _KeyType):
        self._data = [(k, v) for k, v in self._data if k != key]

    def __contains__(self, key: _KeyType) -> bool:
        for k, _value in self._data:
            if k == key:
                return True
        return False

    def __len__(self) -> int:
        return len(self._data)

    def __str__(self) -> str:
        return str(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def items(self) -> list[tuple[_KeyType, _ValueType]]:
        return self._data

    def keys(self) -> list[_KeyType]:
        return [k for k, v in self._data]

    def values(self) -> list[_ValueType]:
        return [v for k, v in self._data]
