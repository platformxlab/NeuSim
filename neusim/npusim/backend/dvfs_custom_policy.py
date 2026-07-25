from typing import Any, Sequence

from neusim.configs.chips.ChipConfig import ChipConfig
import neusim.npusim.frontend.Operator as Operator
from neusim.npusim.frontend.util import compute_component_slack_for_op
from neusim.npusim.frontend.Operator import (
    ComponentDVFSConfig,
    DVFSConfig,
    DVFSPolicy,
    OpcodeType,
)
from neusim.npusim.backend.dvfs_power_getter import (
    SA_POINTS,
    VU_POINTS,
    SRAM_POINTS,
    HBM_MC_POINTS,
    HBM_DIE_POINTS,
    HBM_IO_POINTS,
    ICI_MC_POINTS,
    HBM_POINTS,
    ICI_POINTS,
    VfPoint,
    _baseline_bw_hbm,
)


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


def get_dvfs_policy_custom(
    op: Operator.Operator,
    config: ChipConfig,
    dvfs_cfg: DVFSConfig,
) -> dict[str, ComponentDVFSConfig]:
    """
    Custom DVFS policy:

    - Use IDEAL-style slack-based continuous frequencies.
    - Snap each component to the next HIGHER reachable table point from
      dvfs_power_getter (SA/VU/SRAM/HBM/ICI).
    - For conv/matmul MXU ops, apply additional slowdown caps:

        SRAM:
          * At most 3x slowdown: f_sram >= base_freq / 3.
          * SRAM total BW >= HBM BW.

        If SA-bound (bounded_by == "Compute" and SA time >= VU time > 0):
          * VU at most 3x slowdown: f_vu >= base_freq / 3.

        If HBM-bound (bounded_by == "Memory"):
          * VU keeps its own slowdown (no extra cap).
          * SRAM still must provide BW >= HBM BW.

    Bounded-by is assumed to be one of:
      "Memory", "ICI/NVLink", "Compute".
    """
    assert dvfs_cfg.policy in (DVFSPolicy.CUSTOM, DVFSPolicy.CUSTOM_ALL, DVFSPolicy.CUSTOM_ALL_ms), f"dvfs_mode must be CUSTOM, CUSTOM_ALL, or CUSTOM_ALL_ms, got {dvfs_cfg.policy}"

    base_freq_ghz = config.freq_GHz
    VMEM_TOTAL_BW_GBPS = config.vmem_bw_GBps
    HBM_BASE_BW_GBPS = max(p.bandwidth_GBs for p in HBM_MC_POINTS)
    ICI_BASE_BW_GBPS = max(p.bandwidth_GBs for p in ICI_MC_POINTS)

    extras, ratios = compute_component_slack_for_op(op)

    # ---------- Basic helpers ----------

    def snap_vf_up_from_points(points: Sequence[VfPoint], target_f_ghz: float) -> tuple[float, float]:
        """
        Given (v, f, s, d) points:
          - Choose the point with the smallest f >= target_f_ghz.
          - If none, use the max-f point.
          - If target_f_ghz <= 0, use the min-f point.
        Returns:
          (v_selected, f_selected)
        """
        sorted_pts = sorted(points, key=lambda p: p.frequency_GHz)

        if target_f_ghz <= 0:
            v, f, _, _ = sorted_pts[0]
            return v, f

        for v, f, _, _ in sorted_pts:
            if f >= target_f_ghz:
                return v, f

        v, f, _, _ = sorted_pts[-1]
        return v, f

    def snap_hbm_up(target_f_ghz: float) -> tuple[float, float]:
        """
        HBM:
          - Use frequency as a proxy for bandwidth.
          - Only DVFS the memory controller (not the PHY, I/O bus, and DRAM arrays).
          - bw_target = bw_base * (f / base_freq).
          - Choose the smallest table bw >= bw_target.
          - Convert selected bw back to an effective frequency.
          - Default voltage is 0.7V.
        Returns:
          (v_selected, f_selected)
        """
        bw_base = HBM_BASE_BW_GBPS

        if target_f_ghz <= 0:
            v, bw, _, _ = min(HBM_MC_POINTS, key=lambda p: p.bandwidth_GBs)
            # f = base_freq_ghz * (bw / bw_base)
            return v, 0

        bw_target = bw_base * (target_f_ghz / base_freq_ghz)
        sorted_pts = sorted(HBM_MC_POINTS, key=lambda p: p.bandwidth_GBs)

        for v, bw, _, _ in sorted_pts:
            if bw >= bw_target:
                # f_sel = base_freq_ghz * (bw / bw_base)
                return v, target_f_ghz

        v, bw, _, _ = sorted_pts[-1]
        # f_sel = base_freq_ghz * (bw / bw_base)
        return v, target_f_ghz

    def snap_ici_up(target_f_ghz: float) -> tuple[float, float]:
        """
        ICI:
          - Use frequency as a proxy for bandwidth.
          - Only DVFS the ICI controller (not the PHY or I/O bus).
          - bw_target = bw_base * (f / base_freq).
          - Choose the smallest table bw >= bw_target.
          - Convert selected bw back to an effective frequency.
          - Default voltage is 0.7V.
        Returns:
          (v_selected, f_selected)
        """
        bw_base = ICI_BASE_BW_GBPS

        if target_f_ghz <= 0:
            v, bw_sel, _, _ = min(ICI_MC_POINTS, key=lambda p: p.bandwidth_GBs)
            # f = base_freq_ghz * (bw_sel / bw_base)
            return v, 0

        bw_target = bw_base * (target_f_ghz / base_freq_ghz)
        sorted_pts = sorted(ICI_MC_POINTS, key=lambda p: p.bandwidth_GBs)

        for v, bw, _, _ in sorted_pts:
            if bw >= bw_target:
                # f_sel = base_freq_ghz * (bw / bw_base)
                return v, target_f_ghz

        v, bw, _, _ = sorted_pts[-1]
        # f_sel = base_freq_ghz * (bw / bw_base)
        return v, target_f_ghz

        # candidate_bw = None
        # for _, bw, _, _ in sorted_pts:
        #     if bw >= bw_target:
        #         candidate_bw = bw
        #         break
        # if candidate_bw is None:
        #     candidate_bw = sorted_pts[-1].bandwidth_GBs

        # f_sel = base_freq_ghz * (candidate_bw / bw_base)
        # v_sel = 0.7 if f_sel > 0 else 0.0
        # return v_sel, f_sel

    def comp(policy: DVFSPolicy, v: float, f_ghz: float) -> ComponentDVFSConfig:
        return ComponentDVFSConfig(
            policy=policy,
            voltage_V=v,
            frequency_GHz=f_ghz,
        )

    # ---------- Operator category helpers ----------

    def is_conv(o: Operator.Operator) -> bool:
        """Conv-like MXU op."""
        return o.opcode_type == OpcodeType.CONV2D

    def is_matmul(o: Operator.Operator) -> bool:
        """Matmul/Einsum-like MXU op."""
        return o.opcode_type == OpcodeType.EINSUM

    def is_flash_attention(o: Operator.Operator) -> bool:
        """FlashAttention MXU op."""
        return o.opcode_type == OpcodeType.FLASH_ATTENTION

    def is_collective(o: Operator.Operator) -> bool:
        """Collective communication op."""
        return o.opcode_type in (
            OpcodeType.COLLECTIVE_REDUCE,
            OpcodeType.COLLECTIVE_NO_COMPUTE,
        )

    def is_elementwise(o: Operator.Operator) -> bool:
        """Elementwise op (e.g., Norm, Softmax, Add, Mul)."""
        return o.opcode_type == OpcodeType.ELEMENTWISE or o.opcode_type == OpcodeType.EMBEDDING_BAG

    def is_up_down_sample(o: Operator.Operator) -> bool:
        """Up/Down sample op."""
        return o.opcode_type == OpcodeType.UP_DOWN_SAMPLE

    def is_conv_matmul(o: Operator.Operator) -> bool:
        """
        Conv/Matmul family to which the DVFS caps apply.
        Currently:
          - Conv2D MXU ops
          - Einsum/MatMul MXU ops
        FlashAttention is kept separate (can be added here if desired).
        """
        return is_conv(o) or is_matmul(o)

    # ---------- Conv/Matmul caps ----------

    def apply_conv_matmul_caps(
        o: Operator.Operator,
        f_sa: float,
        f_vu: float,
        f_sram: float,
        f_hbm_discrete: float,
    ) -> tuple[float, float, float]:
        """
        Apply slowdown constraints for conv/matmul MXU ops.

        Uses:
          - o.stats.bounded_by: "Memory", "ICI/NVLink", or "Compute".
          - SA/VU times to decide SA-bound case when bounded_by == "Compute".
        """
        if not is_conv_matmul(o):
            return f_sa, f_vu, f_sram

        bounded = o.stats.bounded_by  # exact values from pipeline
        sa_t = o.stats.sa_time_ns
        vu_t = o.stats.vu_time_ns

        is_hbm_bound = (bounded == "Memory")
        is_compute_bound = (bounded == "Compute")
        is_sa_bound = is_compute_bound and sa_t == o.stats.execution_time_ns

        # Effective HBM BW from discrete HBM frequency.
        if HBM_BASE_BW_GBPS > 0 and f_hbm_discrete > 0:
            hbm_bw = HBM_BASE_BW_GBPS * (f_hbm_discrete / base_freq_ghz)
        else:
            hbm_bw = 0.0


        # (1) If SA-bound: VU at most 3x slowdown.
        if is_sa_bound :
            if f_vu > 0:
                f_vu_min_3x = base_freq_ghz / 3.0
                if f_vu < f_vu_min_3x:
                    f_vu = f_vu_min_3x
           # SRAM caps: at most 3x slowdown + SRAM BW >= HBM BW.
            if f_sram > 0:
                f_min_3x = base_freq_ghz / 3.0

                f_min_bw = 0.0
                if VMEM_TOTAL_BW_GBPS > 0 and hbm_bw > 0:
                    # Require: vmem_bw = VMEM_TOTAL_BW_GBPS * (f_sram / base_freq) >= hbm_bw
                    f_min_bw = (hbm_bw / VMEM_TOTAL_BW_GBPS) * base_freq_ghz

                f_sram = max(f_sram, f_min_3x, f_min_bw)
        # (2) If HBM-bound:
        #     - VU unchanged (already at its own slowdown).
        #     - SRAM constrained by HBM BW.
        elif is_hbm_bound:
            if f_sram > 0:
                f_min_bw = 0.0
                if VMEM_TOTAL_BW_GBPS > 0 and hbm_bw > 0:
                    f_min_bw = (hbm_bw / VMEM_TOTAL_BW_GBPS) * base_freq_ghz
                f_sram = max(f_sram, f_min_bw)

        # Clamp not to exceed base frequency.
        if f_sa > 0:
            f_sa = min(f_sa, base_freq_ghz)
        if f_vu > 0:
            f_vu = min(f_vu, base_freq_ghz)
        if f_sram > 0:
            f_sram = min(f_sram, base_freq_ghz)

        return f_sa, f_vu, f_sram

    def apply_collective_caps(
        o: Operator.Operator,
        f_sa: float,
        f_vu: float,
        f_sram: float,
        f_hbm_discrete: float,  # kept for symmetry; not used directly
    ) -> tuple[float, float, float]:
        """
        For collective ops:
          - Do NOT DVFS SA: keep at base_freq_ghz if active, else 0.
          - Leave VU/SRAM frequencies as computed by the generic policy.
            (i.e., default slack-based scaling still applies).
        """
        if not is_collective(o):
            return f_sa, f_vu, f_sram

        if o.stats.sa_time_ns > 0:
            f_sa = base_freq_ghz
        else:
            f_sa = 0.0

        return f_sa, f_vu, f_sram

    def apply_flash_attention_caps(
        o: Operator.Operator,
        f_sa: float,
        f_vu: float,
        f_sram: float,
        f_hbm_discrete: float,
    ) -> tuple[float, float, float]:
        """
        FlashAttention-specific DVFS:

        HBM-bounded case:
          - We only use discrete DVFS points for SA/VU (SA_POINTS, VU_POINTS).
          - Step 1: choose an SA frequency such that:
                  scaled_sa_time + vu_softmax_time
            is as close as possible to execution_time_ns, preferring
            solutions that do not EXCEED execution_time_ns.
          - Step 2: with that SA freq fixed, choose a VU frequency such that:
                  scaled_sa_time + scaled_vu_softmax_time
            is as close as possible to execution_time_ns, again preferring
            not to exceed it.
          - Ideal target:
                  scaled_sa_time + scaled_vu_softmax_time == execution_time_ns
          - Enforce:
                f_vu >= f_sa / 3
                f_sram >= f_sa / 3
                SRAM BW >= HBM BW (HBM-bound safety)
                all freqs <= base_freq_ghz

        Non-HBM-bound FlashAttention:
          - Fall back to generic behavior (we only apply SA-bound override for
            the pure SA compute-bound case as before).

        Returns:
            (f_sa, f_vu, f_sram)
        """

        if not is_flash_attention(o):
            return f_sa, f_vu, f_sram

        # ---------- HBM-bounded FlashAttention logic ----------
        if o.stats.bounded_by == "Memory":
            assert isinstance(
                o, Operator.FlashAttentionOperator
            ), f"node_cost must be a FlashAttentionOperator, got {type(o)}"

            et = float(o.stats.execution_time_ns)
            sa_time = float(o.stats.sa_time_ns)
            vu_softmax_time = float(getattr(o.stats, "vu_softmax_time_ns", 0) or 0)

            # If we don't have the pieces we need, just keep existing values.
            if et <= 0 or sa_time <= 0 or vu_softmax_time <= 0:
                return f_sa, f_vu, f_sram

            # Helper: scaled time assuming stats are at base_freq_ghz.
            def scaled_time(base_time_ns: float, freq_ghz: float) -> float:
                if freq_ghz <= 0.0 or base_freq_ghz <= 0.0:
                    return float("inf")
                # time ∝ 1 / freq
                return base_time_ns * (base_freq_ghz / freq_ghz)

            # -------- Step 1: choose SA freq (discrete) --------
            sa_candidates = sorted(
                {f for (_, f, _, _) in SA_POINTS if 0.0 < f <= base_freq_ghz}
            )
            best_sa_f = f_sa if (0.0 < f_sa <= base_freq_ghz) else base_freq_ghz
            best_gap = float("inf")

            for cand_f in sa_candidates:
                t_sa = scaled_time(sa_time, cand_f)
                total = t_sa + vu_softmax_time

                # Prefer not to exceed et if possible.
                if total <= et:
                    gap = et - total
                else:
                    # Only consider as fallback if no non-exceeding candidate exists.
                    gap = float("inf")

                if gap < best_gap or (gap == best_gap and cand_f < best_sa_f):
                    best_gap = gap
                    best_sa_f = cand_f

            # If all candidates exceeded et (unlikely in HBM-bound), pick the one
            # with minimal absolute diff, preferring lower freq for energy.
            if best_gap == float("inf"):
                best_sa_f = f_sa if (0.0 < f_sa <= base_freq_ghz) else base_freq_ghz
                best_gap = abs(
                    scaled_time(sa_time, best_sa_f) + vu_softmax_time - et
                )
                for cand_f in sa_candidates:
                    t_sa = scaled_time(sa_time, cand_f)
                    total = t_sa + vu_softmax_time
                    gap = abs(total - et)
                    if gap < best_gap or (gap == best_gap and cand_f < best_sa_f):
                        best_gap = gap
                        best_sa_f = cand_f

            f_sa = best_sa_f
            t_sa_scaled = scaled_time(sa_time, f_sa)

            # -------- Step 2: choose VU freq (discrete) --------
            vu_candidates = sorted(
                {f for (_, f, _, _) in VU_POINTS if 0.0 < f <= base_freq_ghz}
            )

            # Start from current (or base) as fallback.
            best_vu_f = f_vu if (0.0 < f_vu <= base_freq_ghz) else base_freq_ghz
            # Baseline gap with SA scaled, VU at baseline.
            base_total = t_sa_scaled + vu_softmax_time
            best_gap = et - base_total if base_total <= et else abs(base_total - et)

            # Enforce VU >= SA/3 during search.
            min_vu_f = f_sa / 3.0

            feasible_found = False
            for cand_f in vu_candidates:
                if cand_f < min_vu_f:
                    continue

                t_vu = scaled_time(vu_softmax_time, cand_f)
                total = t_sa_scaled + t_vu

                # Prefer total <= et if possible.
                if total <= et:
                    gap = et - total
                    feasible = True
                else:
                    gap = abs(total - et)
                    feasible = False

                # First prefer feasible (<= et), then smaller gap, then lower freq.
                if feasible:
                    if (not feasible_found) or gap < best_gap or (
                        gap == best_gap and cand_f < best_vu_f
                    ):
                        feasible_found = True
                        best_gap = gap
                        best_vu_f = cand_f
                elif not feasible_found:
                    # Only compete with other "exceeding" choices if we never
                    # found a non-exceeding one.
                    if gap < best_gap or (gap == best_gap and cand_f < best_vu_f):
                        best_gap = gap
                        best_vu_f = cand_f

            f_vu = best_vu_f

            # -------- SRAM constraints (HBM-bound) --------

            # 1) At least f_sa / 3 if active.
            if f_sa > 0.0:
                f_sram = max(f_sram, f_sa / 3.0) if f_sram > 0.0 else (f_sa / 3.0)

            # 2) SRAM BW >= HBM BW (using discrete HBM freq proxy).
            if (
                HBM_BASE_BW_GBPS > 0.0
                and f_hbm_discrete > 0.0
                and VMEM_TOTAL_BW_GBPS > 0.0
            ):
                hbm_bw = HBM_BASE_BW_GBPS * (f_hbm_discrete / base_freq_ghz)
                required_f_sram = (hbm_bw / VMEM_TOTAL_BW_GBPS) * base_freq_ghz
                if required_f_sram > 0.0:
                    if f_sram <= 0.0:
                        f_sram = required_f_sram
                    else:
                        f_sram = max(f_sram, required_f_sram)

            # 3) Clamp all to base_freq_ghz.
            if f_sa > 0.0:
                f_sa = min(f_sa, base_freq_ghz)
            if f_vu > 0.0:
                f_vu = min(f_vu, base_freq_ghz)
            if f_sram > 0.0:
                f_sram = min(f_sram, base_freq_ghz)

            return f_sa, f_vu, f_sram

        # ---------- Non-HBM-bound FlashAttention ----------

        # SA-bound compute case: keep at base freq (existing behavior).
        if (
            o.stats.bounded_by == "Compute"
            and o.stats.sa_time_ns == o.stats.execution_time_ns
            and o.stats.sa_time_ns > 0
        ):
            if o.stats.sa_time_ns > 0:
                f_sa = base_freq_ghz
            if o.stats.vu_time_ns > 0:
                f_vu = base_freq_ghz
            if o.stats.vmem_time_ns > 0:
                f_sram = base_freq_ghz

        return f_sa, f_vu, f_sram

    # ---------- Continuous frequency targets ----------

    sa_time = op.stats.sa_time_ns
    vu_time = op.stats.vu_time_ns
    vmem_time = op.stats.vmem_time_ns
    mem_time = op.stats.memory_time_ns
    ici_time = op.stats.ici_time_ns

    f_sa = slowdown_freq(ratios["sa"], base_freq_ghz) if sa_time > 0 else 0.0
    f_vu = slowdown_freq(ratios["vu"], base_freq_ghz) if vu_time > 0 else 0.0
    f_sram = slowdown_freq(ratios["vmem"], base_freq_ghz) if vmem_time > 0 else 0.0
    f_hbm_cont = slowdown_freq(ratios["hbm"], base_freq_ghz) if mem_time > 0 else 0.0
    f_ici_cont = slowdown_freq(ratios["ici"], base_freq_ghz) if ici_time > 0 else 0.0

    # ---------- Discretize HBM and ICI first ----------

    v_hbm, f_hbm = snap_hbm_up(f_hbm_cont)

    # if ici_time > 0:
    v_ici, f_ici = snap_ici_up(f_ici_cont)
    # else:
    #     v_ici, f_ici = 0.7, 0.0

    # ---------- Apply conv/matmul caps (uses discrete HBM freq) ----------
    if is_conv_matmul(op):
        f_sa, f_vu, f_sram = apply_conv_matmul_caps(op, f_sa, f_vu, f_sram, f_hbm)

    if is_collective(op):
        f_sa, f_vu, f_sram = apply_collective_caps(op, f_sa, f_vu, f_sram, f_hbm)

    if is_flash_attention(op):
        f_sa, f_vu, f_sram = apply_flash_attention_caps(op, f_sa, f_vu, f_sram, f_hbm)

    # ---------- Snap SA/VU/SRAM to discrete points ----------

    if sa_time > 0 or f_sa > 0:
        v_sa, f_sa = snap_vf_up_from_points(SA_POINTS, f_sa)
    else:
        v_sa, f_sa = snap_vf_up_from_points(SA_POINTS, 0.0)

    if vu_time > 0 or f_vu > 0:
        v_vu, f_vu = snap_vf_up_from_points(VU_POINTS, f_vu)
    else:
        v_vu, f_vu = snap_vf_up_from_points(VU_POINTS, 0.0)

    if vmem_time > 0 or f_sram > 0:
        v_sram, f_sram = snap_vf_up_from_points(SRAM_POINTS, f_sram)
    else:
        v_sram, f_sram = snap_vf_up_from_points(SRAM_POINTS, 0.0)

    # ---------- Build final plan ----------

    plan = {
        "sa":      comp(DVFSPolicy.CUSTOM, v_sa,   f_sa),
        "vu":      comp(DVFSPolicy.CUSTOM, v_vu,   f_vu),
        "sram":    comp(DVFSPolicy.CUSTOM, v_sram, f_sram),
        "hbm_mc":  comp(DVFSPolicy.CUSTOM, v_hbm,  f_hbm),
        "hbm_die": ComponentDVFSConfig(policy=DVFSPolicy.NONE, voltage_V=0.7, frequency_GHz=1.7, voltage_regulator_scaling_time_ns=0),
        "hbm_io":  ComponentDVFSConfig(policy=DVFSPolicy.NONE, voltage_V=0.7, frequency_GHz=1.7, voltage_regulator_scaling_time_ns=0),
        "ici_mc":  comp(DVFSPolicy.CUSTOM, v_ici,  f_ici),
        "ici_phy": ComponentDVFSConfig(policy=DVFSPolicy.NONE, voltage_V=0.7, frequency_GHz=1.7, voltage_regulator_scaling_time_ns=0),
    }

    if dvfs_cfg.policy in (DVFSPolicy.CUSTOM_ALL, DVFSPolicy.CUSTOM_ALL_ms):
        # Full-HBM policy: couple die and I/O bandwidth to the selected MC point.
        # Plain Custom intentionally leaves these analog domains at peak settings.
        mc_freq = plan["hbm_mc"].frequency_GHz or 1.7
        target_bw = _baseline_bw_hbm() * (mc_freq / 1.7)

        die_points = sorted(
            (p for p in HBM_DIE_POINTS if abs(p.voltage_V - 1.2) < 0.02),
            key=lambda p: p.bandwidth_GBs,
        )
        die_point = next(
            (p for p in die_points if p.bandwidth_GBs >= target_bw - 1),
            die_points[-1],
        )
        io_points = sorted(HBM_IO_POINTS, key=lambda p: p.bandwidth_GBs)
        io_point = next(
            (p for p in io_points if p.bandwidth_GBs >= target_bw - 1),
            io_points[-1],
        )
        plan["hbm_die"] = ComponentDVFSConfig(
            policy=DVFSPolicy.CUSTOM,
            voltage_V=die_point.voltage_V,
            frequency_GHz=mc_freq,
            voltage_regulator_scaling_time_ns=0,
        )
        plan["hbm_io"] = ComponentDVFSConfig(
            policy=DVFSPolicy.CUSTOM,
            voltage_V=io_point.voltage_V,
            frequency_GHz=mc_freq,
            voltage_regulator_scaling_time_ns=0,
        )

    # Collapse SA/VU/SRAM into shared V/f domains if the policy requests fewer
    # than 5 domains (HBM and ICI always stay their own domains).
    couple_compute_domains(plan, getattr(dvfs_cfg, "custom_compute_domain_mode", "dom5"))

    return plan


# ============================================================================
# Compute-domain coupling (variable V/f domain count for the Custom policy).
#
# The Custom policy normally optimizes SA, VU, SRAM, HBM(MC), ICI(MC) as 5
# independent V/f domains. To study how the *number* of V/f domains affects the
# energy/performance trade-off, we optionally merge the compute components
# (SA/VU/SRAM) into shared domains. HBM and ICI always remain separate.
#
# Physical model of a merged domain: all members share one voltage rail and one
# clock, so they run at the SAME (v, f). To avoid slowing any member down (and
# thus preserve performance), the shared frequency is the MAX of the members'
# individually-optimal frequencies — the fastest-needed member sets the rail.
# Because each component's individually-optimal frequency is already the lowest
# that keeps it off the operator's critical path, this max is exactly the
# coupled-optimal operating point. The shared voltage is the highest voltage any
# member needs to sustain that frequency. Slower members therefore run faster
# (and on a higher-voltage rail) than they would in 5-domain mode, spending more
# energy for the same performance — the cost of coarser domains.
# ============================================================================

# Domain groupings over the compute components only. HBM/ICI are implicit
# separate domains, so e.g. dom3 = (SA+VU+SRAM) + HBM + ICI = 3 total.
_COMPUTE_DOMAIN_GROUPS: dict[str, list[tuple[str, ...]]] = {
    "dom5":        [("sa",), ("vu",), ("sram",)],
    "dom3":        [("sa", "vu", "sram")],
    "dom4_savu":   [("sa", "vu"), ("sram",)],
    "dom4_sasram": [("sa", "sram"), ("vu",)],
    "dom4_vusram": [("vu", "sram"), ("sa",)],
}

_COMPUTE_POINTS = {"sa": SA_POINTS, "vu": VU_POINTS, "sram": SRAM_POINTS}


def get_compute_domain_groups(mode: str | None) -> list[tuple[str, ...]]:
    """Return the SA/VU/SRAM groupings for a domain mode (default 'dom5')."""
    m = (mode or "dom5").lower()
    if m not in _COMPUTE_DOMAIN_GROUPS:
        raise ValueError(
            f"Unknown compute domain mode {mode!r}; "
            f"valid options: {sorted(_COMPUTE_DOMAIN_GROUPS)}"
        )
    return _COMPUTE_DOMAIN_GROUPS[m]


def _voltage_for_freq(points: Sequence[VfPoint], target_f_ghz: float) -> float:
    """Voltage of the smallest table point with frequency >= target_f_ghz."""
    sorted_pts = sorted(points, key=lambda p: p.frequency_GHz)
    for p in sorted_pts:
        if p.frequency_GHz >= target_f_ghz - 1e-9:
            return p.voltage_V
    return sorted_pts[-1].voltage_V


def couple_compute_domains(
    plan: dict[str, ComponentDVFSConfig],
    mode: str | None,
) -> dict[str, ComponentDVFSConfig]:
    """
    Merge the SA/VU/SRAM entries of `plan` into shared V/f domains per `mode`.

    Mutates and returns `plan`. For each multi-member group:
      - shared frequency = max selected frequency among active members
        (members with frequency > 0); the fastest-needed member sets the rail.
      - shared voltage    = max over members of the voltage each needs to run at
        the shared frequency in its own V/f table.
    Single-member groups (and the default "dom5") leave `plan` unchanged.
    """
    for group in get_compute_domain_groups(mode):
        if len(group) < 2:
            continue
        freqs = [(plan[c].frequency_GHz or 0.0) for c in group]
        active = [f for f in freqs if f > 0.0]
        shared_f = max(active) if active else max(freqs)
        if shared_f <= 0.0:
            continue
        shared_v = max(_voltage_for_freq(_COMPUTE_POINTS[c], shared_f) for c in group)
        for c in group:
            plan[c] = plan[c].model_copy(
                update={"voltage_V": shared_v, "frequency_GHz": shared_f}
            )
    return plan
