from enum import Enum
from math import ceil

from pydantic import BaseModel


class PowerGatingConfig(BaseModel):
    """
    Power-gating configuration.
    """

    class TemporalGranularity(Enum):
        INSTRUCTION = 1
        OPERATOR = 2
        APPLICATION = 3

    class SASpatialGranularity(Enum):
        PE = 1
        PARTITION = 2
        COMPONENT = 3

    class VUSpatialGranularity(Enum):
        ALU = 1
        PARTITION = 2
        COMPONENT = 3

    class VmemSpatialGranularity(Enum):
        REGISTER_SIZE = 1
        PARTITION = 2

    class ICISpatialGranularity(Enum):
        LINK = 1
        COMPONENT = 2

    class VoltageGranularity(Enum):
        TWO_LEVEL = 1  # only on/off
        MULTI_LEVEL = 2  # on/off + sleep modes

    class PowerGatingPolicy(Enum):
        HW = 1  # HW-managed (auto mode)
        SW = 2  # SW-managed

    name: str = "PowerGatingConfig"
    SA_PG_enabled: bool = False
    SA_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    SA_temporal_granularity: TemporalGranularity = TemporalGranularity.INSTRUCTION
    SA_spatial_granularity: SASpatialGranularity = SASpatialGranularity.COMPONENT
    sa_partition_shapes: list[int] = [128, 128]
    """partition shapes in number of PEs (128*128 by default)"""
    sa_power_level_factors: list[float] = [1.0, 0.0]
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""
    sa_pe_pg_delay_cycles: int = 1
    """Delay in cycles of power gating and waking up a single PE."""
    sa_pg_delay_cycles: int = 10
    """Delay in cycles of power gating and waking up the entire SA."""

    VU_PG_enabled: bool = False
    VU_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    VU_temporal_granularity: TemporalGranularity = TemporalGranularity.INSTRUCTION
    VU_spatial_granularity: VUSpatialGranularity = VUSpatialGranularity.COMPONENT
    vu_partition_shapes: list[int] = [8, 128]
    """partition shapes in number of ALUs (8*128 by default)"""
    vu_power_level_factors: list[float] = [1.0, 0.0]
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""
    vu_pg_delay_cycles: int = 2
    """Delay in cycles of power gating and waking up a VU."""

    vmem_PG_enabled: bool = False
    vmem_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    vmem_temporal_granularity: TemporalGranularity = TemporalGranularity.INSTRUCTION
    vmem_spatial_granularity: VmemSpatialGranularity = VmemSpatialGranularity.PARTITION
    vmem_voltage_granularity: VoltageGranularity = VoltageGranularity.TWO_LEVEL
    vmem_power_level_factors: list[float] = [1.0, 0.0]
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""
    vmem_partition_size_bytes: int = 2 * 1024 * 1024
    """partition size in bytes (2MB by default if spatial granularity is PARTITION)"""
    vmem_partition_pg_delay_cycles: int = 10
    """Delay in cycles of power gating and waking up a vmem partition."""
    vmem_HW_drowsy_period_cycles: int = 2000
    """Period at which all vmem partitions are put into sleep."""

    ici_PG_enabled: bool = False
    ici_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    ici_temporal_granularity: TemporalGranularity = TemporalGranularity.INSTRUCTION
    ici_spatial_granularity: ICISpatialGranularity = ICISpatialGranularity.COMPONENT
    ici_power_level_factors: list[float] = [1.0, 0.0]
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""
    ici_pg_delay_cycles: int = 10
    """Delay in cycles of power gating and waking up an ICI."""

    hbm_PG_enabled: bool = False
    hbm_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    hbm_power_level_factors: list[float] = [1.0, 0.1]  # 0.1 takes into account the auto refresh cost
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""
    hbm_refresh_interval_ns: int = 3900
    hbm_refresh_delay_ns: int = 400  # for 12H device
    hbm_pg_delay_cycles: int = 60

    other_PG_enabled: bool = False
    other_PG_policy: PowerGatingPolicy = PowerGatingPolicy.HW
    other_power_level_factors: list[float] = [1.0, 0.0]
    """power consumption (0~1) at each voltage level (from highest power to lowest power)"""


def get_power_gating_config(pg_config_name: str) -> PowerGatingConfig:
    """
    'disabled', 'NoPG': no power gating. \n
    'ideal_inst_component': ideal power gating with instruction-level temporal granularity and component-level spatial granularity. \n
    'ideal_op_component': ideal power gating with operator-level temporal granularity and component-level spatial granularity. \n
    'ideal_inst_PE_ALU', 'Ideal': ideal power gating with instruction-level temporal granularity and PE/ALU-level spatial granularity. This should result in the most power savings. \n
    'Full': Same as 'Ideal' but with non-zero power-gating factor (power_level_factors) and delay cycles. \n
    '\\<base_config\\>\\_vary_Vth_\\<value\\>_\\<value_sram\\>': vary Vth_low and Vth_sram for sensitivity analysis. The values are the percentage over Vdd. \n
    '\\<base_config\\>\\_vary_PG_delay_\\<value\\>': vary PG delay for sensitivity analysis. The value is specified as the ratio over base config. \n
    """
    pg_config = PowerGatingConfig
    if pg_config_name in ["disabled", "NoPG"]:
        pg_config = PowerGatingConfig(name="NoPG")
    elif pg_config_name == "ideal_inst_component":
        pg_config = PowerGatingConfig(
            name="ideal_inst_component",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.COMPONENT,
            sa_partition_shapes=[128, 128],
            VU_PG_enabled=True,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.COMPONENT,
            vmem_PG_enabled=True,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.PARTITION,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.0],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
        )
    elif pg_config_name == "ideal_op_component":
        pg_config = PowerGatingConfig(
            name="ideal_op_component",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.OPERATOR,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.COMPONENT,
            sa_partition_shapes=[128, 128],
            VU_PG_enabled=True,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.OPERATOR,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.COMPONENT,
            vmem_PG_enabled=True,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.OPERATOR,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.PARTITION,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.0],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.OPERATOR,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
        )
    elif pg_config_name in ["ideal_inst_PE_ALU", "Ideal"]:
        pg_config = PowerGatingConfig(
            name="Ideal",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.PE,
            sa_partition_shapes=[128, 128],
            sa_power_level_factors=[1.0, 0.0],
            sa_pe_pg_delay_cycles=0,
            sa_pg_delay_cycles=0,
            VU_PG_enabled=True,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.ALU,
            vu_power_level_factors=[1.0, 0.0],
            vu_pg_delay_cycles=0,
            vmem_PG_enabled=True,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.0],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            vmem_partition_pg_delay_cycles=0,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
            ici_power_level_factors=[1.0, 0.0],
            ici_pg_delay_cycles=0,
            hbm_PG_enabled=True,
            hbm_PG_policy=PowerGatingConfig.PowerGatingPolicy.SW,
            hbm_pg_delay_cycles=0,
        )
    elif pg_config_name.startswith("Base"):
        pg_config = PowerGatingConfig(
            name="Base",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.COMPONENT,
            sa_partition_shapes=[128, 128],
            # 0.03 -> 0.05 accounts for the fact that weight registers cannot be power gated
            sa_power_level_factors=[1.0, 0.05],
            sa_pe_pg_delay_cycles=1,
            VU_PG_enabled=True,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.COMPONENT,
            vu_power_level_factors=[1.0, 0.03],
            vu_pg_delay_cycles=2,
            vmem_PG_enabled=True,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.25],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            vmem_partition_pg_delay_cycles=4,
            vmem_HW_drowsy_period_cycles=2000,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
            ici_power_level_factors=[1.0, 0.03],
            ici_pg_delay_cycles=60,
            hbm_PG_enabled=True,
            hbm_PG_policy=PowerGatingConfig.PowerGatingPolicy.HW,
        )
    elif pg_config_name.startswith("HW"):
        pg_config = PowerGatingConfig(
            name="HW",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.PE,
            sa_partition_shapes=[128, 128],
            sa_power_level_factors=[1.0, 0.03],
            sa_pe_pg_delay_cycles=1,
            VU_PG_enabled=True,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.COMPONENT,
            vu_power_level_factors=[1.0, 0.03],
            vu_pg_delay_cycles=2,
            vmem_PG_enabled=True,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.25],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            vmem_partition_pg_delay_cycles=4,
            vmem_HW_drowsy_period_cycles=2000,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
            ici_power_level_factors=[1.0, 0.03],
            ici_pg_delay_cycles=60,
            hbm_PG_enabled=True,
            hbm_PG_policy=PowerGatingConfig.PowerGatingPolicy.HW,
        )
    elif pg_config_name.startswith("Full"):
        pg_config = PowerGatingConfig(
            name="Full",
            SA_PG_enabled=True,
            SA_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            SA_spatial_granularity=PowerGatingConfig.SASpatialGranularity.PE,
            sa_partition_shapes=[128, 128],
            sa_power_level_factors=[1.0, 0.03],
            sa_pe_pg_delay_cycles=1,
            VU_PG_enabled=True,
            VU_PG_policy=PowerGatingConfig.PowerGatingPolicy.SW,
            VU_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            VU_spatial_granularity=PowerGatingConfig.VUSpatialGranularity.COMPONENT,
            vu_power_level_factors=[1.0, 0.03],
            vu_pg_delay_cycles=2,
            vmem_PG_enabled=True,
            vmem_PG_policy=PowerGatingConfig.PowerGatingPolicy.SW,
            vmem_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            vmem_spatial_granularity=PowerGatingConfig.VmemSpatialGranularity.REGISTER_SIZE,
            vmem_voltage_granularity=PowerGatingConfig.VoltageGranularity.TWO_LEVEL,
            vmem_power_level_factors=[1.0, 0.0002],
            vmem_partition_size_bytes=2 * 1024 * 1024,
            vmem_partition_pg_delay_cycles=10,
            ici_PG_enabled=True,
            ici_temporal_granularity=PowerGatingConfig.TemporalGranularity.INSTRUCTION,
            ici_spatial_granularity=PowerGatingConfig.ICISpatialGranularity.COMPONENT,
            ici_power_level_factors=[1.0, 0.03],
            ici_pg_delay_cycles=60,
            hbm_PG_enabled=True,
            hbm_PG_policy=PowerGatingConfig.PowerGatingPolicy.SW,
        )
    else:
        raise ValueError(f"Unsupported power gating configuration: {pg_config_name}")

    # vary Vth_low and PG delay for sensitivity analysis
    if "vary_Vth" in pg_config_name:
        # name scheme: "<base_config_name>_vary_Vth_<value>_<value_sram>"
        pg_config.name = pg_config_name
        Vth = float(pg_config_name.split("_")[-2])
        Vth_sram = float(pg_config_name.split("_")[-1])
        pg_config.sa_power_level_factors[-1] = Vth
        pg_config.vu_power_level_factors[-1] = Vth
        pg_config.vmem_power_level_factors[-1] = Vth_sram
        pg_config.ici_power_level_factors[-1] = Vth
        pg_config.hbm_power_level_factors[-1] = Vth
        pg_config.other_power_level_factors[-1] = Vth
    if "vary_PG_delay" in pg_config_name:
        # name scheme: "<base_config_name>_vary_PG_delay_<value>"
        # <value> is the extra delay ratio: new delay = old delay * <value>
        # This do not apply to PEs in the SA.
        pg_config.name = pg_config_name
        pg_delay = float(pg_config_name.split("_")[-1])
        pg_config.sa_pg_delay_cycles = ceil(pg_config.sa_pg_delay_cycles * pg_delay)
        pg_config.vu_pg_delay_cycles = ceil(pg_config.vu_pg_delay_cycles * pg_delay)
        pg_config.vmem_partition_pg_delay_cycles = ceil(
            pg_config.vmem_partition_pg_delay_cycles * pg_delay
        )
        pg_config.ici_pg_delay_cycles = ceil(pg_config.ici_pg_delay_cycles * pg_delay)

    return pg_config
