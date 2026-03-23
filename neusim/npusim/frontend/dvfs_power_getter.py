"""Backward-compatible facade — re-exports from backend."""

from neusim.npusim.backend.dvfs_power_getter import (  # noqa: F401
    VfPoint,
    VBWPoint,
    PowerEfficiencyPoint,
    Row,
    Groups,
    SA_POINTS,
    VU_POINTS,
    SRAM_POINTS,
    HBM_POINTS,
    ICI_POINTS,
    DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
    FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
    get_power_from_dvfs,
    get_all_dvfs_configs_for_component,
    get_all_dvfs_configs_for_op,
)
