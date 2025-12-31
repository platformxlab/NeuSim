### Utility functions for ops generators.

from copy import deepcopy
from functools import lru_cache
from math import sqrt

import os
import zipfile

import numpy as np

from neusim.configs.models.ModelConfig import ModelConfig
from neusim.npusim.frontend.Operator import Operator


@lru_cache(maxsize=None)
def get_factors(n: int) -> list[int]:
    '''
    Get all factors of a number n.
    '''
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors


@lru_cache(maxsize=None)
def prime_factorize(n: int) -> list[int]:
    '''
    Returns the prime factors of n in ascending order.
    '''

    pfactors = []

    # Print the number of two's that divide n
    while n % 2 == 0:
        pfactors.append(2)
        n = n // 2

    # n must be odd at this point
    # so a skip of 2 (i = i + 2) can be used
    for i in range(3, int(sqrt(n)) + 1, 2):
        # while i divides n , print i ad divide n
        while n % i == 0:
            pfactors.append(i)
            n = n // i

    # Condition if n is a prime
    # number greater than 2
    if n > 2:
        pfactors.append(n)

    return pfactors


@lru_cache(maxsize=None)
def split_parallelism_degree(
    p_degree: int, n_axes: int
) -> list[int]:
    '''
    Use heuristic to split @p_degree to @n_axes axes.
    The heuristic tries to make the shape as square as possible,
    since this gives the best bisection bandwidth.
    '''
    assert n_axes > 0, "Number of axes must be positive."
    assert p_degree > 0, "Parallelism degree must be positive."
    if n_axes == 1:
        return [p_degree]

    ## First, initialize axes to the smallest prime factors of @p_degree.
    ## Then, distribute the remaining factors to the axes.
    factors = prime_factorize(p_degree)
    axes = deepcopy(factors[:n_axes])
    factors = factors[n_axes:]
    for f in factors:
        min_axis = axes.index(min(axes))
        axes[min_axis] *= f

    return axes


def get_ICI_topology_from_num_chips(config: ModelConfig) -> list[int]:
    """
    Get the ICI topology (x, y[, z]) from the number of chips.
    """
    num_axes = 2 if "2D" in config.ICI_topology else 3
    pdegree = config.num_chips

    topology = split_parallelism_degree(pdegree, num_axes)

    return topology


def get_bisection_bw_per_chip_GBps(config: ModelConfig) -> tuple[float, list[int]]:
    """
    Get the bisection BW per chip based on the config.
    Returns (bisection_bw_GBps, topology).
    """
    topology = get_ICI_topology_from_num_chips(config)
    ici_bw_GBps = config.ici_bw_GBps  # BW of two links (bisection bw of one row/column)

    # For 1D torus, @config.ici_bw_GBps is already the bisection bw
    if len(topology) == 1:
        return ici_bw_GBps / config.num_chips, list(topology)

    # For N-D torus, bisection bw is determined by the minimum cut,
    # which removes the max dim from torus
    bisect_topology = sorted(topology)[:-1]
    bisection_surface = int(np.prod(bisect_topology))
    bisection_bw = ici_bw_GBps * bisection_surface / config.num_chips
    # scale to match the TPUv4 paper
    # The factor is fit from the TPUv4 ISCA'23 paper
    bisection_bw = bisection_bw * (
        len(topology) - 1
    )  # ** (len(topology) / (len(topology) - 1))
    return bisection_bw, list(topology)


def compute_component_slack_for_op(op: Operator) -> tuple[dict[str, int], dict[str, float]]:
    """
    Compute per-operator DVFS slack (extra time) for each component.

    Returns:
        extras: {"sa","vu","hbm","ici","vmem"} -> extra time (ns), >= 0.
        ratios: {"sa","vu","hbm","ici","vmem"} -> extra/active ratio, >= 0.0.
    """
    et: int = op.stats.execution_time_ns
    comp_times: dict[str, int] = {
        "sa": op.stats.sa_time_ns,
        "vu": op.stats.vu_time_ns,
        "hbm": op.stats.memory_time_ns,
        "ici": op.stats.ici_time_ns,
        "vmem": op.stats.vmem_time_ns,
    }

    extras: dict[str, int] = {}
    ratios: dict[str, float] = {}

    for k, t in comp_times.items():
        if t <= 0:
            extras[k] = 0
            ratios[k] = 0.0
            continue
        extra = et - t
        if extra < 0:
            extra = 0
        extras[k] = int(extra)
        ratios[k] = float(extra / t)

    return extras, ratios


def open_zip(
    file, mode = "r", add_extension_in_filename = True
):
    if isinstance(file, str):
        if not file.endswith(".zip") and not os.path.exists(file):
            if add_extension_in_filename:
                file += ".zip"  # add .zip extension if creating a new file
    return zipfile.ZipFile(file=file, mode=mode, compression=zipfile.ZIP_DEFLATED)  # type: ignore
