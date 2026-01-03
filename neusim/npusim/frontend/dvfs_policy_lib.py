### Pre-defined DVFS policies, and helper functions to apply DVFS policies to Operators.

from neusim.configs.chips.ChipConfig import ChipConfig
import neusim.npusim.frontend.Operator as Operator
from neusim.npusim.frontend.Operator import ComponentDVFSConfig, DVFSPolicy, DVFSConfig
from neusim.npusim.frontend.util import compute_component_slack_for_op


# ----- SA voltage-frequency bands -----
SA_VF_TABLE = [
    (0.45, 0.600240096),
    (0.50, 0.850340136),
    (0.55, 1.149425287),
    (0.60, 1.351351351),
    (0.65, 1.602564103),
    (0.70, 1.700680272),
]

# ----- VU voltage-frequency bands -----
VU_VF_TABLE = [
    (0.45, 0.600240096),
    (0.50, 0.850340136),
    (0.55, 1.149425287),
    (0.60, 1.351351351),
    (0.65, 1.602564103),
    (0.70, 1.700680272),
]

# ----- SRAM voltage-frequency bands -----
SRAM_VF_TABLE = [
    (0.45, 0.500000000),
    (0.50, 0.750750751),
    (0.55, 1.050420168),
    (0.60, 1.250000000),
    (0.65, 1.501501502),
    (0.70, 1.700680272),
]

# ----- HBM voltage-bandwidth-frequency proxy table -----
# (v,  f_proxy_ghz)
HBM_VF_TABLE = [
    # (1.00,  1.459),
    # (1.05,  1.544),
    # (1.10,  1.629),
    # (1.15,  1.712),
    # (1.20,  1.793),
    (0.45, 1.311252269),
    (0.50, 1.388384755),
    (0.55, 1.463666062),
    (0.60, 1.543883848),
    (0.65, 1.622250454),
    (0.70, 1.7),
]

ICI_VF_TABLE = [
    (0.45, 1.308720),
    (0.50, 1.388110),
    (0.55, 1.461829),
    (0.60, 1.541220),
    (0.65, 1.620610),
    (0.70, 1.7),
]


def slowdown_freq(ratio: float, base_freq_GHz: float, min_freq_GHz: float = 0.05) -> float:
    """
    Given ratio = extra / active_time, use all slack:
        new_time = active_time * (1 + ratio)
        new_freq = base_freq / (1 + ratio)
    Clamp into [min_freq_GHz, base_freq_GHz].
    """
    if ratio <= 0:
        return base_freq_GHz
    f = base_freq_GHz / (1.0 + ratio)
    if f < min_freq_GHz:
        f = min_freq_GHz
    if f > base_freq_GHz:
        f = base_freq_GHz
    return f


def pick_v_from_freq(f_ghz: float, table: list[tuple[float, float]]) -> float:
    '''
    @return: voltage for a given frequency @f_ghz in the (V, f) @table.
    Assume table is sorted ascending.
    '''
    if f_ghz <= 0:
        return 0.0

    v0, f0 = table[0]
    if f_ghz <= f0:
        return v0

    for i in range(len(table) - 1):
        v_curr, f_curr = table[i]
        v_next, f_next = table[i + 1]
        if f_ghz > f_curr and f_ghz <= f_next:
            return v_next

    return table[-1][0]


def comp(policy: DVFSPolicy, v: float, f_ghz: float, scaling_time_ns: int = 20) -> ComponentDVFSConfig:
    '''Helper to build component entries'''
    return ComponentDVFSConfig(
        policy=policy,
        voltage_V=v,
        frequency_GHz=f_ghz,
        voltage_regulator_scaling_time_ns=scaling_time_ns,
    )


def get_dvfs_policy_None(
    op: Operator.Operator | None = None, config: ChipConfig | None = None, dvfs_cfg: DVFSConfig | None = None, # unused
) -> dict[str, ComponentDVFSConfig]:
    plan = {
        "sa":   comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        "vu":   comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        "sram": comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        "hbm":  comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        "ici":  comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
    }
    return plan


def get_dvfs_policy_Ideal(
    op: Operator.Operator, config: ChipConfig, dvfs_cfg: DVFSConfig,
) -> dict[str, ComponentDVFSConfig]:

    # Fixed base frequency for DVFS planning (GHz)
    base_freq_ghz = config.freq_GHz

    # Per-operator slack ratios
    extras, ratios = compute_component_slack_for_op(op)

    # SA
    if op.stats.sa_time_ns > 0:
        f_sa_ghz = slowdown_freq(ratios["sa"], base_freq_ghz)
        v_sa = pick_v_from_freq(f_sa_ghz, SA_VF_TABLE)
    else:
        f_sa_ghz, v_sa = 0.05, 0.45 # min freq/v

    # VU
    if op.stats.vu_time_ns > 0:
        f_vu_ghz = slowdown_freq(ratios["vu"], base_freq_ghz)
        v_vu = pick_v_from_freq(f_vu_ghz, VU_VF_TABLE)
    else:
        f_vu_ghz, v_vu = 0.05, 0.45 # min freq/v

    # SRAM / Vmem (use vmem slack)
    if op.stats.vmem_time_ns > 0:
        f_sram_ghz = slowdown_freq(ratios["vmem"], base_freq_ghz)
        v_sram = pick_v_from_freq(f_sram_ghz, SRAM_VF_TABLE)
    else:
        f_sram_ghz, v_sram = 0.05, 0.45 # min freq/v

    # HBM
    if op.stats.memory_time_ns > 0:
        f_hbm_ghz = slowdown_freq(ratios["hbm"], base_freq_ghz)
        v_hbm = pick_v_from_freq(f_hbm_ghz, HBM_VF_TABLE)
    else:
        f_hbm_ghz, v_hbm = 0.0, 0.45 # min freq/v

    # ICI
    if op.stats.ici_time_ns > 0:
        f_ici_ghz = slowdown_freq(ratios["ici"], base_freq_ghz)
        v_ici = pick_v_from_freq(f_ici_ghz, ICI_VF_TABLE)
    else:
        f_ici_ghz, v_ici = 0.0, 0.45 # min freq/v

    plan = {
        # use 200ns scaling time for IDEAL DVFS since it is most power efficient
        # when calculating time overhead, we ignore this scaling time for ideal DVFS
        "sa":   comp(DVFSPolicy.IDEAL, v_sa,   f_sa_ghz,    200),
        "vu":   comp(DVFSPolicy.IDEAL, v_vu,   f_vu_ghz,    200),
        "sram": comp(DVFSPolicy.IDEAL, v_sram, f_sram_ghz,  200),
        "hbm":  comp(DVFSPolicy.IDEAL, v_hbm,  f_hbm_ghz,   200),
        "ici":  comp(DVFSPolicy.IDEAL, v_ici,  f_ici_ghz,   200),
    }
    return plan


def get_dvfs_config(
    op: Operator.Operator,
    config: ChipConfig,
    dvfs_cfg: DVFSConfig,
) -> dict[str, ComponentDVFSConfig]:
    """
    Build the DVFSConfigs for each component for this operator based on dvfs_mode.
    Only SA, VU, SRAM/VMEM, HBM, ICI are controlled.
    """

    plan: dict[str, ComponentDVFSConfig] = {}
    # For NONE, configure as the default numbers.
    if dvfs_cfg.policy == DVFSPolicy.NONE:
        plan = get_dvfs_policy_None(op, config)

    # For IDEAL, use the computed per-op settings.
    elif dvfs_cfg.policy == DVFSPolicy.IDEAL:
        plan = get_dvfs_policy_Ideal(op, config, dvfs_cfg)

    else:
        raise ValueError(f"Unsupported DVFSPolicy: {dvfs_cfg.policy}")

    return plan
