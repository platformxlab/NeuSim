"""Power analysis orchestration facade.

Re-exports modeling functions from backend and configs, and keeps
orchestration helpers that tie everything together.
"""

import time

import neusim.npusim.frontend.Operator as Operator
from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.models.ModelConfig import ModelConfig

# Re-export PowerGatingConfig from configs
from neusim.configs.power_gating.PowerGatingConfig import (  # noqa: F401
    PowerGatingConfig,
    get_power_gating_config,
)

# Re-export DVFS power getter names from backend
from neusim.npusim.backend.dvfs_power_getter import (  # noqa: F401
    DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
    FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
    get_all_dvfs_configs_for_op,
    get_power_from_dvfs,
)

# Re-export all modeling functions from backend power model
from neusim.npusim.backend.power_model import (  # noqa: F401
    add_op_dvfs_exe_time_overhead,
    analyze_dynamic_energy,
    analyze_hbm_static_energy,
    analyze_ici_static_energy,
    analyze_other_static_energy,
    analyze_sa_static_energy,
    analyze_vmem_static_energy,
    analyze_vu_static_energy,
    apply_regulator_efficiency,
    compute_peak_sa_flops_per_sec_from_chip_config,
    compute_peak_sa_flops_per_sec_from_dvfs_config,
    compute_peak_vu_flops_per_sec_from_chip_config,
    compute_peak_vu_flops_per_sec_from_dvfs_config,
    compute_sa_flops_util,
    compute_vu_flops_util,
    cycle_to_ns,
    ns_to_cycle,
    scale_dvfs_component_time,
)

# Request-level optimizer entry points. Lazy imports inside the optimizer avoid
# a cycle when Pareto evaluation calls analyze_operator_energy below.
from neusim.npusim.frontend.dvfs_optimizer import (  # noqa: F401
    PAPER_MS_INTERVAL_NS,
    configure_dvfs_c_ms_all_budgets,
    configure_dvfs_c_ms_with_regions,
    configure_dvfs_c_no_pareto_all_budgets,
    configure_dvfs_c_with_degradation,
    configure_dvfs_for_op,
    configure_dvfs_for_ops,
    generate_pareto_energy_latency_points_for_all_ops,
    generate_pareto_energy_latency_points_for_op,
    generate_pareto_energy_latency_points_for_op_exhaustive_search,
    generate_pareto_energy_latency_points_for_op_greedy_search,
    get_global_dvfs_config_helper,
)
from neusim.npusim.frontend.Operator import DVFSConfig, DVFSPolicy

# =====================================================================
# Orchestration helpers (stay in frontend)
# =====================================================================

def analyze_operator_energy(
    op: Operator.Operator,
    config: ChipConfig,
    pg_config: str | PowerGatingConfig | None = None,
    dvfs_config: str | DVFSConfig | DVFSPolicy | None = None,
    set_dvfs_config_for_op: bool = True,
    ignore_vr_power_loss: bool = False,
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

    # 5) Apply regulator efficiency losses unless the caller requests the
    # regulator-independent energy used by optimizer comparisons.
    if not ignore_vr_power_loss:
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


def analyze_all_operator_energy(
    ops: list[Operator.Operator],
    config: ModelConfig,
    pg_config: str | PowerGatingConfig | None = None,
    dvfs_config: str | DVFSConfig | DVFSPolicy | None = None,
    ignore_vr_power_loss: bool = False,
    dump_pareto_points_to_file: bool = False,
    timing_result: dict | None = None,
) -> list[Operator.Operator]:
    """
    Analyze energy for all operators.
    """
    dvfs_config = get_global_dvfs_config_helper(dvfs_config)
    plan_start_s = time.perf_counter()
    configure_dvfs_for_ops(
        ops,
        config,
        dvfs_config,
        dump_pareto_points_to_file,
        pg_config=pg_config,
        timing_result=timing_result,
    )
    analyze_all_operator_energy.last_dvfs_plan_time_s = (
        time.perf_counter() - plan_start_s
    )

    for op in ops:
        analyze_operator_energy(
            op,
            config,
            pg_config,
            dvfs_config,
            set_dvfs_config_for_op=False,
            ignore_vr_power_loss=ignore_vr_power_loss,
        )

    return ops
