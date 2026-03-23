"""Power analysis orchestration facade.

Re-exports modeling functions from backend and configs, and keeps
orchestration helpers that tie everything together.
"""

from absl import logging

import neusim.npusim.frontend.Operator as Operator
from neusim.npusim.frontend.Operator import DVFSPolicy, DVFSConfig, ComponentDVFSConfig
from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.models.ModelConfig import ModelConfig

# Re-export PowerGatingConfig from configs
from neusim.configs.power_gating.PowerGatingConfig import (  # noqa: F401
    PowerGatingConfig,
    get_power_gating_config,
)

# Re-export DVFS policy from backend
from neusim.npusim.backend.dvfs_policy_lib import get_dvfs_config  # noqa: F401

# Re-export DVFS power getter names from backend
from neusim.npusim.backend.dvfs_power_getter import (  # noqa: F401
    get_all_dvfs_configs_for_op,
    get_power_from_dvfs,
    DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
    FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
)

# Re-export all modeling functions from backend power model
from neusim.npusim.backend.power_model import (  # noqa: F401
    compute_peak_sa_flops_per_sec_from_chip_config,
    compute_peak_vu_flops_per_sec_from_chip_config,
    compute_peak_sa_flops_per_sec_from_dvfs_config,
    compute_peak_vu_flops_per_sec_from_dvfs_config,
    compute_sa_flops_util,
    compute_vu_flops_util,
    cycle_to_ns,
    ns_to_cycle,
    scale_dvfs_component_time,
    analyze_dynamic_energy,
    analyze_sa_static_energy,
    analyze_vu_static_energy,
    analyze_vmem_static_energy,
    analyze_ici_static_energy,
    analyze_hbm_static_energy,
    analyze_other_static_energy,
    add_op_dvfs_exe_time_overhead,
    apply_regulator_efficiency,
)


# =====================================================================
# Orchestration helpers (stay in frontend)
# =====================================================================

def get_global_dvfs_config_helper(dvfs_config: str | DVFSConfig | DVFSPolicy | None = None) -> DVFSConfig:
    if not dvfs_config:
        dvfs_config = DVFSConfig()
    elif isinstance(dvfs_config, DVFSPolicy):
        dvfs_config = DVFSConfig(policy=dvfs_config)
    elif isinstance(dvfs_config, str):
        if dvfs_config != "None":
            dvfs_str_split = dvfs_config.split("_")
            policy = DVFSPolicy.from_str(dvfs_str_split[0])
            if len(dvfs_str_split) == 1:
                dvfs_config = DVFSConfig(policy=policy)
            else:
                perf_degrad_factor = float(dvfs_str_split[1])
                dvfs_config = DVFSConfig(policy=policy, performance_degradation_percentage=perf_degrad_factor)
        else:
            dvfs_config = DVFSConfig(policy=DVFSPolicy.NONE)

    return dvfs_config


def configure_dvfs_for_op(
    op: Operator.Operator,
    config: ChipConfig,
    dvfs_config: DVFSConfig,
) -> Operator.Operator:
    """
    Initialize per-component DVFSConfig on `op` based on a JSON DVFS policy.
    """
    dvfs_configs = get_dvfs_config(op, config, dvfs_config)

    op.dvfs_sa = dvfs_configs["sa"]
    op.dvfs_vu = dvfs_configs["vu"]
    op.dvfs_sram = dvfs_configs["sram"]
    op.dvfs_hbm = dvfs_configs["hbm"]
    op.dvfs_ici = dvfs_configs["ici"]

    return op


def analyze_operator_energy(
    op: Operator.Operator,
    config: ChipConfig,
    pg_config: str | PowerGatingConfig | None = None,
    dvfs_config: str | DVFSConfig | DVFSPolicy | None = None,
    set_dvfs_config_for_op: bool = True,
) -> Operator.Operator:
    """Top-level operator energy analysis.

    Workflow:
      1. Resolve power-gating config and dvfs config.
      2. Configure DVFS for this operator (populate op.dvfs_*).
      3. Run dynamic energy analysis with DVFS:
           - scales active times by frequency,
           - scales dynamic power by V^2 * f,
           - recomputes execution_time_ns and bounded_by internally.
      4. Run static energy analyses (DVFS-aware leakage + power gating),
         which may adjust component times.
      5. Apply regulator efficiency losses.
      6. Final consistency pass:
           - ensure execution_time_ns >= all component times,
           - set bounded_by according to the true critical component.
    """
    # 1) Resolve power-gating config and dvfs config
    if not pg_config:
        pg_config = config.pg_config
    if isinstance(pg_config, str):
        pg_config = get_power_gating_config(pg_config)

    # 2) Configure DVFS for this operator
    if set_dvfs_config_for_op:
        dvfs_config = get_global_dvfs_config_helper(dvfs_config)
        configure_dvfs_for_op(op, config, dvfs_config)

    # 3) Dynamic power/energy with DVFS
    scale_dvfs_component_time(op, config)
    add_op_dvfs_exe_time_overhead(op, config)

    analyze_dynamic_energy(op, config)

    # 4) Static power/energy (DVFS-aware leakage + PG)
    analyze_sa_static_energy(op, config, pg_config)
    analyze_vu_static_energy(op, config, pg_config)
    analyze_vmem_static_energy(op, config, pg_config)
    analyze_ici_static_energy(op, config, pg_config)
    analyze_hbm_static_energy(op, config, pg_config)
    analyze_other_static_energy(op, config, pg_config)

    # 5) Apply regulator efficiency losses
    apply_regulator_efficiency(op)

    # 6) Final consistency: execution time & bounded_by
    exe_time_ns = op.stats.execution_time_ns
    bounded_by = op.stats.bounded_by

    candidates = [
        (op.stats.sa_time_ns,     "Compute"),
        (op.stats.vu_time_ns,     "Compute"),
        (op.stats.vmem_time_ns,   "Compute"),
        (op.stats.memory_time_ns, "Memory"),
        (op.stats.ici_time_ns,    "ICI/NVLink"),
    ]

    for t, label in candidates:
        if t > exe_time_ns:
            exe_time_ns = t
            bounded_by = label

    op.stats.execution_time_ns = exe_time_ns
    op.stats.bounded_by = bounded_by

    return op


def configure_dvfs_for_ops(
    ops: list[Operator.Operator],
    config: ModelConfig,
    dvfs_config: DVFSConfig,
) -> list[Operator.Operator]:
    """
    Configure DVFS for all operators.
    """
    logging.set_verbosity(logging.INFO)

    return [configure_dvfs_for_op(op, config, dvfs_config) for op in ops]


def analyze_all_operator_energy(
    ops: list[Operator.Operator],
    config: ModelConfig,
    pg_config: str | PowerGatingConfig | None = None,
    dvfs_config: str | DVFSConfig | DVFSPolicy | None = None,
) -> list[Operator.Operator]:
    """
    Analyze energy for all operators.
    """
    dvfs_config = get_global_dvfs_config_helper(dvfs_config)
    configure_dvfs_for_ops(ops, config, dvfs_config)

    for op in ops:
        analyze_operator_energy(op, config, pg_config, dvfs_config, set_dvfs_config_for_op=False)

    return ops
