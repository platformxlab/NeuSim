"""Core power/energy modeling functions for NPU components."""

from math import ceil

import numpy as np

import neusim.npusim.frontend.Operator as Operator
from neusim.npusim.frontend.Operator import DVFSPolicy, DVFSConfig, ComponentDVFSConfig
from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.power_gating.PowerGatingConfig import PowerGatingConfig
from neusim.npusim.backend.dvfs_power_getter import (
    get_power_from_dvfs,
    DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
    FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
)


def compute_peak_sa_flops_per_sec_from_chip_config(config: ChipConfig) -> float:
    freq = config.freq_GHz * 1e9
    num_sa = config.num_sa
    sa_dim_size = config.sa_dim
    return 2 * (sa_dim_size**2) * freq * num_sa


def compute_peak_vu_flops_per_sec_from_chip_config(config: ChipConfig) -> float:
    freq = config.freq_GHz * 1e9
    num_vu = config.num_vu
    vu_num_alus = 128 * 8  # TODO: make this a parameter in chip config
    return vu_num_alus * freq * num_vu

def compute_peak_sa_flops_per_sec_from_dvfs_config(config: ChipConfig, dvfs: ComponentDVFSConfig) -> float:
    if not dvfs.frequency_GHz or dvfs.frequency_GHz <= 0:
        return compute_peak_sa_flops_per_sec_from_chip_config(config)
    freq = dvfs.frequency_GHz * 1e9
    num_sa = config.num_sa
    sa_dim_size = config.sa_dim
    return 2 * (sa_dim_size**2) * freq * num_sa


def compute_peak_vu_flops_per_sec_from_dvfs_config(config: ChipConfig, dvfs: ComponentDVFSConfig) -> float:
    if not dvfs.frequency_GHz or dvfs.frequency_GHz <= 0:
        return compute_peak_vu_flops_per_sec_from_chip_config(config)
    freq = dvfs.frequency_GHz * 1e9
    num_vu = config.num_vu
    vu_num_alus = 128 * 8  # TODO: make this a parameter in chip config
    return vu_num_alus * freq * num_vu

def compute_sa_flops_util(op: Operator.Operator, config: ChipConfig, dvfs: ComponentDVFSConfig) -> float:
    """
    Compute SA flops utilization for an operator.
    """
    peak_sa_flops_per_sec = compute_peak_sa_flops_per_sec_from_dvfs_config(config, dvfs)
    sa_time_ns = op.stats.sa_time_ns
    if sa_time_ns > 0:  # op.op_type == Operator.OpType.MXU:
        # assert sa_time_ns > 0, f"SA time is 0 for op: {op.to_csv_dict()}"
        sa_flops_util = min(
            (op.stats.flop_count / sa_time_ns * 1e9) / peak_sa_flops_per_sec,
            1.0,
        )
    else:
        sa_flops_util = 0
    return sa_flops_util


def compute_vu_flops_util(op: Operator.Operator, config: ChipConfig, dvfs: ComponentDVFSConfig) -> float:
    """
    Compute VU flops utilization for an operator.
    """
    peak_vu_flops_per_sec = compute_peak_vu_flops_per_sec_from_dvfs_config(config, dvfs)
    vu_time_ns = op.stats.vu_time_ns
    if op.op_type == Operator.OpType.MXU:
        # assert peak_vu_flops_per_sec > 0, f"Peak VU FLOPS is {peak_vu_flops_per_sec} for op: {op.to_csv_dict()}"
        # assert vu_time_ns > 0, f"VU time is {vu_time_ns} for op: {op.to_csv_dict()}"
        # assumes vu flops is at least 1/8 of sa flops for accmulation
        vu_flops_util = min(
            (op.stats.flop_count / 8 / vu_time_ns * 1e9) / peak_vu_flops_per_sec,
            1.0,
        )
    else:
        if peak_vu_flops_per_sec > 0 and vu_time_ns > 0:
            vu_flops_util = min(
                (op.stats.flop_count / vu_time_ns * 1e9) / peak_vu_flops_per_sec,
                1.0,
            )
        else:
            vu_flops_util = 0
    return vu_flops_util


def cycle_to_ns(cycles: int, freq_GHz: float) -> float:
    """
    Convert cycles to nanoseconds.
    """
    return cycles / freq_GHz


def ns_to_cycle(ns: float, freq_GHz: float) -> float:
    """
    Convert nanoseconds to cycles.
    """
    return ns * freq_GHz


def scale_dvfs_component_time(op: Operator.Operator, config: ChipConfig) -> Operator.Operator:
    """
    Scale per-component active times based on the DVFS frequency set in op.dvfs_*.
    Original times are assumed to be measured at base 1.7 GHz.
    """
    base_freq_GHz = config.freq_GHz

    def _scale_time_with_dvfs(time_ns: int | float, dvfs: ComponentDVFSConfig) -> int:
        """
        Scale the active time when a DVFS frequency is specified.
        If no DVFS freq, keep original time.
        """
        if time_ns <= 0 or dvfs.frequency_GHz is None:
            return int(float(time_ns))
        if dvfs.frequency_GHz <= 0:
            return int(float(time_ns))
        # Original times assumed at base_freq_Hz.
        return ceil(time_ns * base_freq_GHz / dvfs.frequency_GHz)

    # Apply DVFS freq to component times (performance effect only)
    op.stats.sa_time_ns     = _scale_time_with_dvfs(op.stats.sa_time_ns,     op.dvfs_sa)
    op.stats.vu_time_ns     = _scale_time_with_dvfs(op.stats.vu_time_ns,     op.dvfs_vu)
    op.stats.vmem_time_ns   = _scale_time_with_dvfs(op.stats.vmem_time_ns,   op.dvfs_sram)
    op.stats.ici_time_ns    = _scale_time_with_dvfs(op.stats.ici_time_ns,    op.dvfs_ici)
    op.stats.memory_time_ns = _scale_time_with_dvfs(op.stats.memory_time_ns, op.dvfs_hbm)

    return op


def analyze_dynamic_energy(
    op: Operator.Operator, config: ChipConfig
) -> Operator.Operator:
    """
    Analyze dynamic power and energy for an operator.

    Assumes:
      - Static energy & execution/component times (possibly DVFS/PG-adjusted)
        have already been computed.
      - `configure_dvfs_for_op` has been called to populate op.dvfs_*.

    Behavior:
      - Uses get_power_from_dvfs(...) to obtain dynamic power.
      - Computes dynamic energy for each component as P_dyn * active_time.
    """

    # Recompute FLOPS utils with updated times
    sa_flops_util = compute_sa_flops_util(op, config, op.dvfs_sa)
    vu_flops_util = compute_vu_flops_util(op, config, op.dvfs_vu)

    # Dynamic powers
    if config.enable_dvfs:
        sa_dyn_W, _ = get_power_from_dvfs("SA", op.dvfs_sa)
        vu_dyn_W, _ = get_power_from_dvfs("VU", op.dvfs_vu)
        sram_dyn_W, _ = get_power_from_dvfs("SRAM", op.dvfs_sram)
        hbm_dyn_W, _ = get_power_from_dvfs("HBM", op.dvfs_hbm)
        ici_dyn_W, _ = get_power_from_dvfs("ICI", op.dvfs_ici)
    else:
        sa_dyn_W = config.dynamic_power_sa_W
        vu_dyn_W = config.dynamic_power_vu_W
        sram_dyn_W = config.dynamic_power_vmem_W
        hbm_dyn_W = config.dynamic_power_hbm_W
        ici_dyn_W = config.dynamic_power_ici_W

    # 'other' still uses config (no DVFS enabled)
    other_dyn_W = config.dynamic_power_other_W

    # Dynamic energy per component
    exe_time_ns = op.stats.execution_time_ns
    sa_time_ns = op.stats.sa_time_ns
    vu_time_ns = op.stats.vu_time_ns
    vmem_time_ns = op.stats.vmem_time_ns
    ici_time_ns = op.stats.ici_time_ns
    hbm_time_ns = op.stats.memory_time_ns

    op.stats.dynamic_energy_sa_J = sa_dyn_W * sa_time_ns * config.num_sa / 1e9 * sa_flops_util
    op.stats.dynamic_energy_vu_J = vu_dyn_W * vu_time_ns * config.num_vu / 1e9 * vu_flops_util
    op.stats.dynamic_energy_sram_J = sram_dyn_W * vmem_time_ns / 1e9
    op.stats.dynamic_energy_ici_J = ici_dyn_W * ici_time_ns / 1e9
    op.stats.dynamic_energy_hbm_J = hbm_dyn_W * hbm_time_ns / 1e9
    op.stats.dynamic_energy_other_J = other_dyn_W * exe_time_ns / 1e9

    return op


def analyze_sa_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for SA.
    """
    if config.enable_dvfs:
        _, static_sa_W = get_power_from_dvfs("SA", op.dvfs_sa)
    else:
        static_sa_W = config.static_power_sa_W
    static_sa_W *= config.num_sa
    pg_power_W = static_sa_W * pg_config.sa_power_level_factors[-1]

    # No power-gating
    if not pg_config.SA_PG_enabled:
        op.stats.static_energy_sa_J = static_sa_W * op.stats.execution_time_ns / 1e9
        return op

    if op.stats.sa_time_ns > 0:
        # assumes in the worst case, idle intervals are evenly distributed
        # over the entire execution time
        worst_case_sa_idle_interval_ns = ceil(
            (op.stats.execution_time_ns - 1) / (op.stats.sa_time_ns / config.sa_dim)
        )
        if worst_case_sa_idle_interval_ns == 0:
            # if SA is not idle (op is SA-bound), then no power gating
            op.stats.static_energy_sa_J = static_sa_W * op.stats.execution_time_ns / 1e9
            return op
    else:
        worst_case_sa_idle_interval_ns = 0

    sa_flops_util = compute_sa_flops_util(op, config, op.dvfs_sa)

    # calculate PG delay overhead and update op stats
    if (
        op.stats.sa_time_ns > 0
        and pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PE
    ):  # used by HW and Full
        assert isinstance(op.stats, (Operator.EinsumStatistics, Operator.FlashAttentionStatistics))
        overhead_ns_1 = ceil(
            op.stats.sa_time_ns / config.sa_dim * cycle_to_ns(
                pg_config.sa_pe_pg_delay_cycles, config.freq_GHz
            )
        )
        overhead_ns_2 = ceil(
            op.stats.num_sa_ops * cycle_to_ns(pg_config.sa_pe_pg_delay_cycles, config.freq_GHz)
        )
        overhead_ns = min(overhead_ns_1, overhead_ns_2)
        op.stats.sa_time_ns += overhead_ns
    elif (
        op.stats.sa_time_ns > 0
        and pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.COMPONENT
    ):  # used by Base (idle-detect policy)
        if worst_case_sa_idle_interval_ns > 4 * cycle_to_ns(
            pg_config.sa_pg_delay_cycles, config.freq_GHz
        ):
            pg_delay_ns = ceil(
                cycle_to_ns(pg_config.sa_pg_delay_cycles, config.freq_GHz)
            )
            # sa_time_ns/sa_dim is the worst case number of idle intervals
            op.stats.sa_time_ns += ceil(pg_delay_ns * (op.stats.sa_time_ns / config.sa_dim))
    # if op.stats.sa_time_ns > op.stats.execution_time_ns:
    #     op.stats.execution_time_ns = op.stats.sa_time_ns
    #     op.stats.bounded_by = "Compute"

    sa_time_ns = op.stats.sa_time_ns
    exe_time_ns = max(op.stats.execution_time_ns, sa_time_ns)

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.COMPONENT
    ):
        pg_energy = pg_power_W * (exe_time_ns - sa_time_ns) / 1e9
        static_energy = static_sa_W * sa_time_ns / 1e9

        # Base policy: do not power gate if idle interval is smaller than 2x pg delay time
        if sa_time_ns > 0 and worst_case_sa_idle_interval_ns < 2 * cycle_to_ns(
            pg_config.sa_pg_delay_cycles, config.freq_GHz
        ):
            pg_energy = 0
            static_energy = static_sa_W * exe_time_ns / 1e9

        op.stats.static_energy_sa_J = pg_energy + static_energy
        return op

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PE
    ):
        pg_energy = (
            pg_power_W * sa_time_ns / 1e9 * (1 - sa_flops_util)
            + pg_power_W * (exe_time_ns - sa_time_ns) / 1e9
        )
        static_energy = static_sa_W * sa_time_ns / 1e9 * sa_flops_util
        op.stats.static_energy_sa_J = pg_energy + static_energy
        return op

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.COMPONENT
    ):
        if op.op_type == Operator.OpType.MXU:
            op.stats.static_energy_sa_J = static_sa_W * exe_time_ns / 1e9
        else:
            op.stats.static_energy_sa_J = 0
        return op

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PE
    ):
        op.stats.static_energy_sa_J = static_sa_W * exe_time_ns / 1e9 * sa_flops_util
        return op

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.COMPONENT
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.SA_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.SA_spatial_granularity
        == PowerGatingConfig.SASpatialGranularity.PE
    ):
        raise NotImplementedError()  # TODO

    # should not reach here
    raise ValueError("Unsupported/Unknown/Invalid SA power gating configuration")


def analyze_vu_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for VU.
    """
    if config.enable_dvfs:
        _, static_vu_W = get_power_from_dvfs("VU", op.dvfs_vu)
    else:
        static_vu_W = config.static_power_vu_W
    static_vu_W *= config.num_vu
    pg_power_W = static_vu_W * pg_config.vu_power_level_factors[-1]

    # No power-gating
    if not pg_config.VU_PG_enabled:
        op.stats.static_energy_vu_J = static_vu_W * op.stats.execution_time_ns / 1e9
        return op

    if op.stats.vu_time_ns > 0:
        # assumes in the worst case, idle intervals are evenly distributed
        # over the entire execution time
        worst_case_vu_idle_interval_ns = ceil(
            (op.stats.execution_time_ns - 1) / op.stats.vu_time_ns
        )
        if worst_case_vu_idle_interval_ns == 0:
            # if VU is not idle (op is VU-bound), then no power gating
            op.stats.static_energy_vu_J = static_vu_W * op.stats.execution_time_ns / 1e9
            return op
    else:
        worst_case_vu_idle_interval_ns = 0

    vu_flops_util = compute_vu_flops_util(op, config, op.dvfs_vu)

    # calculate PG delay overhead and update op stats
    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.VU_PG_policy == PowerGatingConfig.PowerGatingPolicy.HW
    ):  # used by Base and HW (idle-detect policy)
        if worst_case_vu_idle_interval_ns > 4 * cycle_to_ns(
            pg_config.vu_pg_delay_cycles, config.freq_GHz
        ):
            pg_delay_ns = ceil(
                cycle_to_ns(pg_config.vu_pg_delay_cycles, config.freq_GHz)
            )
            # vu_time_ns is the worst case number of idle intervals
            op.stats.vu_time_ns += pg_delay_ns * op.stats.vu_time_ns
            # if op.stats.vu_time_ns > op.stats.execution_time_ns:
            #     op.stats.execution_time_ns = op.stats.vu_time_ns
            #     op.stats.bounded_by = "Compute"

    vu_time_ns = op.stats.vu_time_ns
    exe_time_ns = max(op.stats.execution_time_ns, vu_time_ns)

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.COMPONENT
    ):  # used by Base, HW, and Full pg_config
        pg_energy = pg_power_W * (exe_time_ns - vu_time_ns) / 1e9
        static_energy = static_vu_W * vu_time_ns / 1e9

        # HW policy: do not power gate if idle interval is smaller than 2x pg delay time
        if pg_config.VU_PG_policy == PowerGatingConfig.PowerGatingPolicy.HW:
            if worst_case_vu_idle_interval_ns < 2 * cycle_to_ns(
                pg_config.vu_pg_delay_cycles, config.freq_GHz
            ):
                pg_energy = 0
                static_energy = static_vu_W * exe_time_ns / 1e9

        op.stats.static_energy_vu_J = pg_energy + static_energy

        # Full policy: calculate number of setpm instructions
        if pg_config.VU_PG_policy == PowerGatingConfig.PowerGatingPolicy.SW:
            if exe_time_ns == vu_time_ns:  # VU bound; no VU idle intervals
                op.stats.num_setpm_vu = 0
            elif vu_time_ns == 0:  # VU idle; set VUs to be PG'ed only once
                op.stats.num_setpm_vu = 1
            else:  # use the number of idle intervals as an estimate
                op.stats.num_setpm_vu = min(
                    round(exe_time_ns / worst_case_vu_idle_interval_ns),
                    (exe_time_ns - vu_time_ns) // 32,  # 32 cycles BET for VU with wake-up delay of 2 cycles on TPUv5p
                                                       # This division estimates the max number of setpm instructions
                                                       # TODO: make this a parameter in PG config
                )
        return op

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.ALU
    ):  # only used by Ideal pg_config for now
        pg_energy = (
            pg_power_W * vu_time_ns / 1e9 * (1 - vu_flops_util)
            + pg_power_W * (exe_time_ns - vu_time_ns) / 1e9
        )
        static_energy = static_vu_W * vu_time_ns / 1e9 * vu_flops_util
        op.stats.static_energy_vu_J = pg_energy + static_energy
        return op

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.COMPONENT
    ):
        if vu_time_ns > 0:
            op.stats.static_energy_vu_J = static_vu_W * exe_time_ns / 1e9
        else:
            op.stats.static_energy_vu_J = 0
        return op

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.ALU
    ):
        op.stats.static_energy_vu_J = static_vu_W * exe_time_ns / 1e9 * vu_flops_util
        return op

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.COMPONENT
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.PARTITION
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.VU_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.VU_spatial_granularity
        == PowerGatingConfig.VUSpatialGranularity.ALU
    ):
        raise NotImplementedError()  # TODO

    # should not reach here
    raise ValueError("Unsupported/Unknown/Invalid VU power gating configuration")


def analyze_vmem_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for vmem.
    """
    if config.enable_dvfs:
        _, static_vmem_W = get_power_from_dvfs("SRAM", op.dvfs_sram)
    else:
        static_vmem_W = config.static_power_vmem_W
    pg_power_W = static_vmem_W * pg_config.vmem_power_level_factors[-1]

    # No power-gating
    if not pg_config.vmem_PG_enabled:
        op.stats.static_energy_sram_J = static_vmem_W * op.stats.execution_time_ns / 1e9
        return op

    partition_granularity = pg_config.vmem_partition_size_bytes
    if (
        pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
    ):
        partition_granularity = 4 * 1024  # 4KB

    vmem_size = config.vmem_size_MB * 1024 * 1024

    # compute vmem capacity utilization
    if op.op_type == Operator.OpType.MXU:
        assert isinstance(
            op.stats, (Operator.EinsumStatistics, Operator.FlashAttentionStatistics)
        ), f"op_name: {op.name} :: op_type: {op.op_type}, opcode_type: {op.opcode_type}, opcode: {op.opcode} not supported for op.stats type {type(op.stats)}\nconfig: {op.config_str}"
        max_vmem_demand = op.stats.max_vmem_demand_bytes
        vmem_capacity_util = min(max_vmem_demand / vmem_size, 1.0)
    else:
        # only use 2MB per core (4MB in total) for operators w/o data reuse
        vmem_capacity_util = 4 / config.vmem_size_MB
    vmem_demand_ceiled = (
        int(np.ceil(vmem_capacity_util * vmem_size / partition_granularity))
        * partition_granularity
    )
    vmem_capacity_util = vmem_demand_ceiled / vmem_size

    exe_time_ns = op.stats.execution_time_ns

    # calculate PG delay overhead and update op stats
    if pg_config.vmem_PG_policy == PowerGatingConfig.PowerGatingPolicy.HW:
        pg_delay_overhead = ceil(
            op.stats.execution_time_ns
            / cycle_to_ns(pg_config.vmem_HW_drowsy_period_cycles, config.freq_GHz)
            * cycle_to_ns(pg_config.vmem_partition_pg_delay_cycles, config.freq_GHz)
        )
        exe_time_ns += pg_delay_overhead


    vmem_time_ns = op.stats.vmem_time_ns


    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        pg_energy = (
            pg_power_W * vmem_time_ns / 1e9 * (1 - vmem_capacity_util)
            + pg_power_W * (exe_time_ns - vmem_time_ns) / 1e9
        )
        static_energy = static_vmem_W * vmem_time_ns / 1e9 * vmem_capacity_util
        op.stats.static_energy_sram_J = pg_energy + static_energy

        # Full policy: calculate number of setpm instructions
        if pg_config.vmem_PG_policy == PowerGatingConfig.PowerGatingPolicy.SW:
            # only set once per operator as we assume fixed tile size per operator for now
            op.stats.num_setpm_sram = 1

        return op

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        pg_energy = (
            pg_power_W * vmem_time_ns / 1e9 * (1 - vmem_capacity_util)
            + pg_power_W * (exe_time_ns - vmem_time_ns) / 1e9
        )
        static_energy = static_vmem_W * vmem_time_ns / 1e9 * vmem_capacity_util
        op.stats.static_energy_sram_J = pg_energy + static_energy
        return op

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        pg_energy = (
            pg_power_W * vmem_time_ns / 1e9 * (1 - vmem_capacity_util)
            + pg_power_W * (exe_time_ns - vmem_time_ns) / 1e9
        )
        static_energy = static_vmem_W * vmem_time_ns / 1e9 * vmem_capacity_util
        op.stats.static_energy_sram_J = pg_energy + static_energy
        return op

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        pg_energy = (
            pg_power_W * vmem_time_ns / 1e9 * (1 - vmem_capacity_util)
            + pg_power_W * (exe_time_ns - vmem_time_ns) / 1e9
        )
        static_energy = static_vmem_W * vmem_time_ns / 1e9 * vmem_capacity_util
        op.stats.static_energy_sram_J = pg_energy + static_energy
        return op

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.TWO_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.vmem_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.vmem_spatial_granularity
        == PowerGatingConfig.VmemSpatialGranularity.PARTITION
        and pg_config.vmem_voltage_granularity
        == PowerGatingConfig.VoltageGranularity.MULTI_LEVEL
    ):
        raise NotImplementedError()  # TODO

    # should not reach here
    raise ValueError("Unsupported/Unknown/Invalid Vmem power gating configuration")


def analyze_ici_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for ICI.
    """
    if config.enable_dvfs:
        _, static_ici_W = get_power_from_dvfs("ICI", op.dvfs_ici)
    else:
        static_ici_W = config.static_power_ici_W
    pg_power_W = static_ici_W * pg_config.ici_power_level_factors[-1]

    # No power-gating
    if not pg_config.ici_PG_enabled:
        op.stats.static_energy_ici_J = static_ici_W * op.stats.execution_time_ns / 1e9
        return op

    # assert (
    #     pg_config.ici_PG_policy == PowerGatingConfig.PowerGatingPolicy.HW
    # ), "Only HW-managed power gating is supported for ICI"

    # calculate PG delay overhead and update op stats
    if op.stats.ici_time_ns > 0:
        pg_delay_ns = ceil(
            2 * cycle_to_ns(pg_config.ici_pg_delay_cycles, config.freq_GHz)
        )
        op.stats.ici_time_ns += pg_delay_ns
        # if op.stats.ici_time_ns > op.stats.execution_time_ns:
        #     op.stats.execution_time_ns = op.stats.ici_time_ns
        #     op.stats.bounded_by = "ICI/NVLink"

    ici_time_ns = op.stats.ici_time_ns
    exe_time_ns = max(op.stats.execution_time_ns, ici_time_ns)

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.LINK
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.INSTRUCTION
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.COMPONENT
    ):
        pg_energy = pg_power_W * (exe_time_ns - ici_time_ns) / 1e9
        static_energy = static_ici_W * ici_time_ns / 1e9
        op.stats.static_energy_ici_J = pg_energy + static_energy
        return op

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.LINK
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.OPERATOR
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.COMPONENT
    ):
        if ici_time_ns > 0:
            op.stats.static_energy_ici_J = static_ici_W * exe_time_ns / 1e9
        else:
            op.stats.static_energy_ici_J = pg_power_W * exe_time_ns / 1e9
        return op

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.LINK
    ):
        raise NotImplementedError()  # TODO

    if (
        pg_config.ici_temporal_granularity
        == PowerGatingConfig.TemporalGranularity.APPLICATION
        and pg_config.ici_spatial_granularity
        == PowerGatingConfig.ICISpatialGranularity.COMPONENT
    ):
        raise NotImplementedError()  # TODO

    # should not reach here
    raise ValueError("Unsupported/Unknown/Invalid ICI power gating configuration")


def analyze_hbm_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for HBM.
    """
    if config.enable_dvfs:
        _, static_hbm_W = get_power_from_dvfs("HBM", op.dvfs_hbm)
    else:
        static_hbm_W = config.static_power_hbm_W
    pg_power_W = static_hbm_W * pg_config.hbm_power_level_factors[-1]

    # No power-gating
    if not pg_config.hbm_PG_enabled:
        op.stats.static_energy_hbm_J = (
            static_hbm_W * op.stats.execution_time_ns / 1e9
        )
        return op

    # assume 4MB DMA size if memory traffic is larger than this
    if op.stats.memory_traffic_bytes < 4 * 1024 * 1024:
        active_length_ns = op.stats.memory_time_ns
    else:
        active_length_ns = (
            4 * 1024 * 1024 / (config.hbm_bw_GBps * 1024 * 1024 * 1024) * 1e9
        )
    hbm_util = op.stats.memory_time_ns / op.stats.execution_time_ns
    num_periods = ceil(
        op.stats.memory_time_ns / active_length_ns
    )
    idle_length_ns = ceil(
        op.stats.execution_time_ns / num_periods - active_length_ns
    )

    # break-even time
    BET_ns = config.hbm_latency_ns * 2 + ceil(cycle_to_ns(pg_config.hbm_pg_delay_cycles, config.freq_GHz)) * 4
    idle_detect_timeout_ns = BET_ns * 4

    if pg_config.hbm_PG_policy == PowerGatingConfig.PowerGatingPolicy.SW:
        if idle_length_ns >= BET_ns:
            # power gate HBM
            pg_energy = pg_power_W * (op.stats.execution_time_ns - op.stats.memory_time_ns) / 1e9
            static_energy = op.stats.memory_time_ns / 1e9 * static_hbm_W
            op.stats.static_energy_hbm_J = pg_energy + static_energy
        else:
            # do not power gate HBM
            op.stats.static_energy_hbm_J = static_hbm_W * op.stats.execution_time_ns / 1e9
    elif pg_config.hbm_PG_policy == PowerGatingConfig.PowerGatingPolicy.HW:
        if idle_length_ns >= idle_detect_timeout_ns:
            # power gate HBM
            pg_energy = pg_power_W * (op.stats.execution_time_ns - op.stats.memory_time_ns) / 1e9
            static_energy = op.stats.memory_time_ns / 1e9 * static_hbm_W
            op.stats.static_energy_hbm_J = pg_energy + static_energy

            # calculate PG delay overhead and update op stats
            pg_delay_ns = ceil(
                cycle_to_ns(pg_config.hbm_pg_delay_cycles, config.freq_GHz) * num_periods
            )
            op.stats.memory_time_ns += pg_delay_ns
        else:
            # do not power gate HBM
            op.stats.static_energy_hbm_J = static_hbm_W * op.stats.execution_time_ns / 1e9
    else:
        raise NotImplementedError("Unknown HBM power gating policy")

    return op


def analyze_other_static_energy(
    op: Operator.Operator, config: ChipConfig, pg_config: PowerGatingConfig
) -> Operator.Operator:
    """
    Static power/energy analysis for other.
    """
    assert pg_config.other_PG_enabled is False, "Other power gating is not supported"

    op.stats.static_energy_other_J = (
        config.static_power_other_W * op.stats.execution_time_ns / 1e9
    )
    return op


def add_op_dvfs_exe_time_overhead(op: Operator.Operator, config: ChipConfig) -> Operator.Operator:
    """
    Add DVFS latency overhead to op execution times and
    set voltage conversion efficiency in the DVFSConfig for each component.
    """
    # helpers
    def _get_dvfs_config_for_component(comp: str) -> ComponentDVFSConfig:
        if comp == "sa":
            return op.dvfs_sa
        elif comp == "vu":
            return op.dvfs_vu
        elif comp == "hbm":
            return op.dvfs_hbm
        elif comp == "ici":
            return op.dvfs_ici
        elif comp == "vmem":
            return op.dvfs_sram
        else:
            raise ValueError(f"Unknown component '{comp}'")

    def _apply_exe_time_overhead_for_component(comp: str, st_ns: int):
        if comp == "sa":
            op.stats.sa_time_ns += st_ns
        elif comp == "vu":
            op.stats.vu_time_ns += st_ns
        elif comp == "hbm":
            op.stats.memory_time_ns += st_ns
        elif comp == "ici":
            op.stats.ici_time_ns += st_ns
        elif comp == "vmem":
            op.stats.vmem_time_ns += st_ns
        else:
            raise ValueError(f"Unknown component '{comp}'")

    def _get_activity_factor_for_component(comp: str):
        if comp in ["sa", "vu"]:
            # for SA and VU, activity factor is spatial utilization (e.g., FLOPS)
            return min(1, op.stats.flops_util)
        elif comp in ["hbm", "ici", "vmem"]:
            # for other components, activity factor is time utilization
            time_ns = {
                "hbm": op.stats.memory_time_ns,
                "ici": op.stats.ici_time_ns,
                "vmem": op.stats.vmem_time_ns,
            }[comp]
            util = min(1, time_ns / op.stats.execution_time_ns)
            return util
        else:
            raise ValueError(f"Unknown component '{comp}'")

    def _lookup_efficiency_from_table(scaling_time_ns: int, activity: float, voltage: float, policy: DVFSPolicy) -> float:
        if policy == DVFSPolicy.NONE:
            table = FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE
        else:
            table = DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE

        # filter out scaling time
        rows = [r for r in table if r.scaling_time_ns == scaling_time_ns]
        assert len(rows) > 0, f"No DVFS overhead table entry for scaling_time_ns={scaling_time_ns}"

        # pick the nearest voltage that is greater than or equal to requested voltage
        voltages = sorted(set(r.voltage_V for r in rows))
        v_snap = None
        for v in voltages:
            if v >= voltage:
                v_snap = v
                break
        assert v_snap is not None, f"No DVFS overhead table entry for voltage_V >= {voltage}, scaling_time_ns={scaling_time_ns}, activity_factor={activity}"
        rows = [r for r in rows if r.voltage_V == v_snap]

        # pick the row with smallest activity factor >= requested activity
        rows = sorted(rows, key=lambda r: r.activity_factor)
        chosen_row = None
        for r in rows:
            if r.activity_factor >= activity:
                chosen_row = r
                break
        assert chosen_row is not None, f"No DVFS overhead table entry for activity_factor >= {activity}, V={v_snap}, scaling_time_ns={scaling_time_ns}"

        return chosen_row.power_efficiency_percent

    # component execution times
    comp_times: dict[str, int] = {
        "sa":   op.stats.sa_time_ns,
        "vu":   op.stats.vu_time_ns,
        "hbm":  op.stats.memory_time_ns,
        "ici":  op.stats.ici_time_ns,
        "vmem": op.stats.vmem_time_ns,
    }

    for comp, t_ns in comp_times.items():
        dvfs_config = _get_dvfs_config_for_component(comp)
        dvfs_policy = dvfs_config.policy

        activity_factor = _get_activity_factor_for_component(comp)
        voltage_V = dvfs_config.voltage_V or 0.7  # default to 0.7V for now

        chosen_eff= _lookup_efficiency_from_table(
            scaling_time_ns=dvfs_config.voltage_regulator_scaling_time_ns,
            activity=activity_factor,
            voltage=voltage_V,
            policy=dvfs_policy,
        )

        if t_ns > 0 and dvfs_policy not in [DVFSPolicy.NONE, DVFSPolicy.IDEAL]:
            # if component is unused, do not change execution time
            _apply_exe_time_overhead_for_component(comp, dvfs_config.voltage_regulator_scaling_time_ns)
        dvfs_config.voltage_conversion_power_efficiency_percent = chosen_eff

    # update e2e execution time
    exe_time_ns = max(
        op.stats.sa_time_ns,
        op.stats.vu_time_ns,
        op.stats.vmem_time_ns,
        op.stats.ici_time_ns,
        op.stats.memory_time_ns,
    )
    op.stats.execution_time_ns = exe_time_ns
    bounded_by = None
    for t, label in [
        (op.stats.sa_time_ns,   "Compute"),
        (op.stats.vu_time_ns,   "Compute"),
        (op.stats.vmem_time_ns, "Compute"),
        (op.stats.memory_time_ns, "Memory"),
        (op.stats.ici_time_ns,  "ICI/NVLink"),
    ]:
        if t == exe_time_ns:
            bounded_by = label
            break
    assert bounded_by, "Failed to determine bounded_by after DVFS overhead addition"
    op.stats.bounded_by = bounded_by

    return op


def apply_regulator_efficiency(op: Operator.Operator) -> Operator.Operator:
    """
    Apply regulator efficiency losses to per-component energies.
    Multiply both dynamic and static energies by (100/efficiency_percent).

    Notes:
      - We scale energies (J) post computation, which is equivalent to
        scaling power for the already-integrated durations.
      - Components covered: SA, VU, SRAM (vmem), HBM, ICI. 'other' is unchanged.
    """

    # SA
    op.stats.dynamic_energy_sa_J *= 100.0 / op.dvfs_sa.voltage_conversion_power_efficiency_percent
    op.stats.static_energy_sa_J *= 100.0 / op.dvfs_sa.voltage_conversion_power_efficiency_percent

    # VU
    op.stats.dynamic_energy_vu_J *= 100.0 / op.dvfs_vu.voltage_conversion_power_efficiency_percent
    op.stats.static_energy_vu_J *= 100.0 / op.dvfs_vu.voltage_conversion_power_efficiency_percent

    # SRAM (vmem)
    op.stats.dynamic_energy_sram_J *= 100.0 / op.dvfs_sram.voltage_conversion_power_efficiency_percent
    op.stats.static_energy_sram_J *= 100.0 / op.dvfs_sram.voltage_conversion_power_efficiency_percent

    # HBM
    op.stats.dynamic_energy_hbm_J *= 100.0 / op.dvfs_hbm.voltage_conversion_power_efficiency_percent
    op.stats.static_energy_hbm_J *= 100.0 / op.dvfs_hbm.voltage_conversion_power_efficiency_percent

    # ICI
    op.stats.dynamic_energy_ici_J *= 100.0 / op.dvfs_ici.voltage_conversion_power_efficiency_percent
    op.stats.static_energy_ici_J *= 100.0 / op.dvfs_ici.voltage_conversion_power_efficiency_percent

    return op
