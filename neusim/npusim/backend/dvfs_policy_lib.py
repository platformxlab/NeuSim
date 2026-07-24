### Pre-defined DVFS policies, and helper functions to apply DVFS policies to Operators.

import neusim.npusim.frontend.Operator as Operator
from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend.dvfs_custom_policy import (
    couple_compute_domains,
    get_dvfs_policy_custom,
)
from neusim.npusim.backend.dvfs_power_getter import (
    DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
    HBM_DIE_POINTS,
    HBM_IO_POINTS,
    HBM_MC_POINTS,
    ICI_MC_POINTS,
    SA_POINTS,
    SRAM_POINTS,
    VU_POINTS,
    VBWPoint,
    VfPoint,
    _baseline_bw_hbm,
)
from neusim.npusim.frontend.Operator import ComponentDVFSConfig, DVFSConfig, DVFSPolicy
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


def _snap_freq_up(points: list[VfPoint], target_f_ghz: float) -> tuple[float, float]:
    """
    Snap to the smallest discrete (v, f) point with f >= target_f_ghz.
    If target_f_ghz <= 0, return the min-frequency point.
    If no point has f >= target_f_ghz, return the max-frequency point.
    Returns (voltage_V, frequency_GHz).
    """
    sorted_pts = sorted(points, key=lambda p: p.frequency_GHz)
    if target_f_ghz <= 0:
        return sorted_pts[0].voltage_V, sorted_pts[0].frequency_GHz
    for p in sorted_pts:
        if p.frequency_GHz >= target_f_ghz:
            return p.voltage_V, p.frequency_GHz
    return sorted_pts[-1].voltage_V, sorted_pts[-1].frequency_GHz


def _snap_bw_up(points: list[VBWPoint], target_f_ghz: float, base_freq_ghz: float) -> tuple[float, float]:
    """
    For HBM/ICI: convert target frequency to bandwidth, snap voltage from
    the smallest discrete bandwidth point >= target, keep continuous frequency.
    Returns (voltage_V, frequency_GHz).
    """
    base_bw = max(p.bandwidth_GBs for p in points)
    if target_f_ghz <= 0:
        p = min(points, key=lambda pp: pp.bandwidth_GBs)
        return p.voltage_V, 0.0
    bw_target = base_bw * (target_f_ghz / base_freq_ghz)
    sorted_pts = sorted(points, key=lambda p: p.bandwidth_GBs)
    for p in sorted_pts:
        if p.bandwidth_GBs >= bw_target:
            return p.voltage_V, target_f_ghz
    return sorted_pts[-1].voltage_V, target_f_ghz


def _vr_efficiency(voltage_V: float, scaling_time_ns: int = 20, activity: float = 0.5) -> float:
    """Approximate VR efficiency for energy-aware voltage selection."""
    table = DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE
    rows = [r for r in table if r.scaling_time_ns == scaling_time_ns]
    if not rows:
        # Fall back to nearest available scaling time
        available = sorted(set(r.scaling_time_ns for r in table))
        nearest = min(available, key=lambda t: abs(t - scaling_time_ns))
        rows = [r for r in table if r.scaling_time_ns == nearest]
    # snap voltage up
    voltages = sorted(set(r.voltage_V for r in rows))
    v_snap = voltages[-1]
    for v in voltages:
        if v >= voltage_V:
            v_snap = v
            break
    rows = [r for r in rows if r.voltage_V == v_snap]
    # snap activity up
    rows = sorted(rows, key=lambda r: r.activity_factor)
    for r in rows:
        if r.activity_factor >= activity:
            return r.power_efficiency_percent
    return rows[-1].power_efficiency_percent


def _snap_freq_energy_optimal(
    points: list[VfPoint], target_f_ghz: float,
    scaling_time_ns: int = 200, activity: float = 0.5,
) -> tuple[float, float]:
    """
    Snap to the discrete (v, f) point that minimizes energy (VR-aware)
    among all points with f >= target_f_ghz.
    """
    sorted_pts = sorted(points, key=lambda p: p.frequency_GHz)
    if target_f_ghz <= 0:
        return sorted_pts[0].voltage_V, sorted_pts[0].frequency_GHz

    candidates = [p for p in sorted_pts if p.frequency_GHz >= target_f_ghz]
    if not candidates:
        return sorted_pts[-1].voltage_V, sorted_pts[-1].frequency_GHz

    best_p = min(candidates, key=lambda p: (
        (p.static_power_W + p.dynamic_power_W) * (100.0 / _vr_efficiency(p.voltage_V, scaling_time_ns, activity))
    ))
    return best_p.voltage_V, best_p.frequency_GHz


def _snap_bw_energy_optimal(
    points: list[VBWPoint], target_f_ghz: float, base_freq_ghz: float,
    scaling_time_ns: int = 200, activity: float = 0.5,
) -> tuple[float, float]:
    """
    For HBM/ICI MC: find the voltage that minimizes energy (accounting for VR efficiency)
    among all BW points with bandwidth >= target.
    Returns (voltage_V, frequency_GHz).
    """
    base_bw = max(p.bandwidth_GBs for p in points)
    if target_f_ghz <= 0:
        p = min(points, key=lambda pp: pp.bandwidth_GBs)
        return p.voltage_V, 0.0

    bw_target = base_bw * (target_f_ghz / base_freq_ghz)
    candidates = [p for p in points if p.bandwidth_GBs >= bw_target]
    if not candidates:
        # fallback: pick max-BW point
        p = max(points, key=lambda pp: pp.bandwidth_GBs)
        return p.voltage_V, target_f_ghz

    # pick the candidate that minimizes energy score
    best_p = min(candidates, key=lambda p: (
        (p.static_power_W + p.dynamic_power_W) * (100.0 / _vr_efficiency(p.voltage_V, scaling_time_ns, activity))
    ))
    return best_p.voltage_V, target_f_ghz


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
        "sa":      comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        "vu":      comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        "sram":    comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        "hbm_mc":  comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        "hbm_die": comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        "hbm_io":  comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        "ici_mc":  comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        "ici_phy": comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
    }
    # Compatibility aliases used by pre-split NeuSim callers.
    plan["hbm"] = plan["hbm_mc"]
    plan["ici"] = plan["ici_mc"]
    return plan


def get_dvfs_policy_Ideal(
    op: Operator.Operator, config: ChipConfig, dvfs_cfg: DVFSConfig,
) -> dict[str, ComponentDVFSConfig]:

    # Fixed base frequency for DVFS planning (GHz)
    base_freq_ghz = config.freq_GHz

    # Per-operator slack ratios
    extras, ratios = compute_component_slack_for_op(op)

    # Ideal DVFS scaling time (ns) — used for VR efficiency lookup
    ideal_scaling_time_ns = 200
    exec_time = op.stats.execution_time_ns if op.stats.execution_time_ns > 0 else 1

    # SA — snap to discrete V/f point (VR-aware)
    if op.stats.sa_time_ns > 0:
        f_sa_cont = slowdown_freq(ratios["sa"], base_freq_ghz)
        sa_activity = min(1.0, op.stats.flops_util) if op.stats.flops_util > 0 else 0.5
        v_sa, f_sa_ghz = _snap_freq_energy_optimal(SA_POINTS, f_sa_cont, ideal_scaling_time_ns, sa_activity)
    else:
        v_sa, f_sa_ghz = _snap_freq_energy_optimal(SA_POINTS, 0.0)

    # VU — snap to discrete V/f point (VR-aware)
    if op.stats.vu_time_ns > 0:
        f_vu_cont = slowdown_freq(ratios["vu"], base_freq_ghz)
        vu_activity = min(1.0, op.stats.flops_util) if op.stats.flops_util > 0 else 0.5
        v_vu, f_vu_ghz = _snap_freq_energy_optimal(VU_POINTS, f_vu_cont, ideal_scaling_time_ns, vu_activity)
    else:
        v_vu, f_vu_ghz = _snap_freq_energy_optimal(VU_POINTS, 0.0)

    # SRAM / Vmem — snap to discrete V/f point (VR-aware)
    if op.stats.vmem_time_ns > 0:
        f_sram_cont = slowdown_freq(ratios["vmem"], base_freq_ghz)
        sram_activity = min(1.0, op.stats.vmem_time_ns / exec_time)
        v_sram, f_sram_ghz = _snap_freq_energy_optimal(SRAM_POINTS, f_sram_cont, ideal_scaling_time_ns, sram_activity)
    else:
        v_sram, f_sram_ghz = _snap_freq_energy_optimal(SRAM_POINTS, 0.0)

    # HBM — bandwidth-based: snap voltage from discrete BW table (VR-aware)
    if op.stats.memory_time_ns > 0:
        f_hbm_cont = slowdown_freq(ratios["hbm"], base_freq_ghz)
        hbm_activity = min(1.0, op.stats.memory_time_ns / exec_time)
        v_hbm, f_hbm_ghz = _snap_bw_energy_optimal(HBM_MC_POINTS, f_hbm_cont, base_freq_ghz, ideal_scaling_time_ns, hbm_activity)
    else:
        v_hbm, f_hbm_ghz = _snap_bw_energy_optimal(HBM_MC_POINTS, 0.0, base_freq_ghz)

    # ICI — bandwidth-based: snap voltage from discrete BW table (VR-aware)
    if op.stats.ici_time_ns > 0:
        f_ici_cont = slowdown_freq(ratios["ici"], base_freq_ghz)
        ici_activity = min(1.0, op.stats.ici_time_ns / exec_time)
        v_ici, f_ici_ghz = _snap_bw_energy_optimal(ICI_MC_POINTS, f_ici_cont, base_freq_ghz, ideal_scaling_time_ns, ici_activity)
    else:
        v_ici, f_ici_ghz = _snap_bw_energy_optimal(ICI_MC_POINTS, 0.0, base_freq_ghz)

    # Couple HBM die & I/O to MC bandwidth (same approach as CustomAll)
    hbm_base_bw = _baseline_bw_hbm()
    mc_freq = f_hbm_ghz if f_hbm_ghz and f_hbm_ghz > 0 else 1.7
    target_bw = hbm_base_bw * (mc_freq / 1.7)

    die_candidates = sorted(
        [p for p in HBM_DIE_POINTS if abs(p.voltage_V - 1.2) < 0.02],
        key=lambda p: p.bandwidth_GBs,
    )
    die_point = die_candidates[-1]
    for p in die_candidates:
        if p.bandwidth_GBs >= target_bw - 1:
            die_point = p
            break
    v_hbm_die, f_hbm_die = die_point.voltage_V, mc_freq

    io_candidates = sorted(HBM_IO_POINTS, key=lambda p: p.bandwidth_GBs)
    io_point = io_candidates[-1]
    for p in io_candidates:
        if p.bandwidth_GBs >= target_bw - 1:
            io_point = p
            break
    v_hbm_io, f_hbm_io = io_point.voltage_V, mc_freq

    plan = {
        # use 20ns scaling time for IDEAL DVFS
        # when calculating time overhead, we ignore this scaling time for ideal DVFS
        "sa":      comp(DVFSPolicy.IDEAL, v_sa,   f_sa_ghz,    20),
        "vu":      comp(DVFSPolicy.IDEAL, v_vu,   f_vu_ghz,    20),
        "sram":    comp(DVFSPolicy.IDEAL, v_sram, f_sram_ghz,  20),
        "hbm_mc":  comp(DVFSPolicy.IDEAL, v_hbm,  f_hbm_ghz,  20),
        "hbm_die": comp(DVFSPolicy.IDEAL, v_hbm_die, f_hbm_die, 0),
        "hbm_io":  comp(DVFSPolicy.IDEAL, v_hbm_io,  f_hbm_io,  0),
        "ici_mc":  comp(DVFSPolicy.IDEAL, v_ici,  f_ici_ghz,   20),
        "ici_phy": comp(DVFSPolicy.NONE,  0.7,    1.7,         0),
    }
    return plan


def get_dvfs_policy_compute_only(
    op: Operator.Operator, config: ChipConfig, dvfs_cfg: DVFSConfig,
) -> dict[str, ComponentDVFSConfig]:
    ideal_plan = plan = get_dvfs_policy_Ideal(op, config, dvfs_cfg)
    f_sa_ghz = ideal_plan["sa"].frequency_GHz
    v_sa = ideal_plan["sa"].voltage_V
    f_vu_ghz = ideal_plan["vu"].frequency_GHz
    v_vu = ideal_plan["vu"].voltage_V
    f_sram_ghz = ideal_plan["sram"].frequency_GHz
    v_sram = ideal_plan["sram"].voltage_V
    f_hbm_ghz = ideal_plan["hbm_mc"].frequency_GHz
    v_hbm = ideal_plan["hbm_mc"].voltage_V
    f_ici_ghz = ideal_plan["ici_mc"].frequency_GHz
    v_ici = ideal_plan["ici_mc"].voltage_V
    assert f_sa_ghz is not None and v_sa is not None
    assert f_vu_ghz is not None and v_vu is not None
    assert f_sram_ghz is not None and v_sram is not None
    assert f_hbm_ghz is not None and v_hbm is not None
    assert f_ici_ghz is not None and v_ici is not None
    bounded_by = op.stats.bounded_by
    is_mem_ici_bounded = bounded_by in ("Memory", "ICI/NVLink")

    if is_mem_ici_bounded:
        # Use the smallest slowdown for all of SA/VU/SRAM, defaults for HBM/ICI
        # HBM and ICI use NONE so they do not suffer from voltage conversion loss.
        compute_domain_freq = max(f_sa_ghz, f_vu_ghz, f_sram_ghz)
        compute_domain_v = max(v_sa, v_vu, v_sram)
        plan = {
            "sa":      comp(DVFSPolicy.DVFS_C, compute_domain_v, compute_domain_freq),
            "vu":      comp(DVFSPolicy.DVFS_C, compute_domain_v, compute_domain_freq),
            "sram":    comp(DVFSPolicy.DVFS_C, compute_domain_v, compute_domain_freq),
            "hbm_mc":  comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
            "hbm_die": comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
            "hbm_io":  comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
            "ici_mc":  comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
            "ici_phy": comp(DVFSPolicy.NONE, 0.7,  1.7, 0),
        }
    else:
        # Not memory/ICI bounded (e.g., Compute or others) -> no DVFS
        plan = {
            "sa":      comp(DVFSPolicy.DVFS_C, 0.7, 1.7),
            "vu":      comp(DVFSPolicy.DVFS_C, 0.7, 1.7),
            "sram":    comp(DVFSPolicy.DVFS_C, 0.7, 1.7),
            "hbm_mc":  comp(DVFSPolicy.NONE, 0.7, 1.7, 0),
            "hbm_die": comp(DVFSPolicy.NONE, 0.7, 1.7, 0),
            "hbm_io":  comp(DVFSPolicy.NONE, 0.7, 1.7, 0),
            "ici_mc":  comp(DVFSPolicy.NONE, 0.7, 1.7, 0),
            "ici_phy": comp(DVFSPolicy.NONE, 0.7, 1.7, 0),
        }

    # Re-evaluate each member's voltage at the shared frequency. Merely taking
    # the voltages selected at their original frequencies can undervolt SRAM.
    couple_compute_domains(plan, "dom3")
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

    # For DVFS_C / DVFS_C_NO_PARETO / DVFS_C_ms, use default setting for HBM and ICI.
    elif dvfs_cfg.policy in (DVFSPolicy.DVFS_C, DVFSPolicy.DVFS_C_NO_PARETO, DVFSPolicy.DVFS_C_ms):
        plan = get_dvfs_policy_compute_only(op, config, dvfs_cfg)

    # our custom policy
    elif dvfs_cfg.policy == DVFSPolicy.CUSTOM:
        plan = get_dvfs_policy_custom(op, config, dvfs_cfg)

    # custom policy with full HBM DVFS (MC + Die + IO coupled)
    elif dvfs_cfg.policy in (DVFSPolicy.CUSTOM_ALL, DVFSPolicy.CUSTOM_ALL_ms):
        plan = get_dvfs_policy_custom(op, config, dvfs_cfg)

    else:
        raise ValueError(f"Unsupported DVFSPolicy: {dvfs_cfg.policy}")

    plan.setdefault("hbm", plan["hbm_mc"])
    plan.setdefault("ici", plan["ici_mc"])
    return plan
