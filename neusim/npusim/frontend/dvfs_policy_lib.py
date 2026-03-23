"""Backward-compatible facade — re-exports from backend."""

from neusim.npusim.backend.dvfs_policy_lib import (  # noqa: F401
    SA_VF_TABLE,
    VU_VF_TABLE,
    SRAM_VF_TABLE,
    HBM_VF_TABLE,
    ICI_VF_TABLE,
    slowdown_freq,
    pick_v_from_freq,
    comp,
    get_dvfs_policy_None,
    get_dvfs_policy_Ideal,
    get_dvfs_config,
)
