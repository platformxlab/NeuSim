from unittest.mock import DEFAULT, patch

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.power_gating.PowerGatingConfig import PowerGatingConfig
from neusim.npusim.frontend import dvfs_optimizer, power_analysis_lib
from neusim.npusim.frontend.Operator import (
    ComponentDVFSConfig,
    DVFSConfig,
    DVFSPolicy,
    Operator,
)


def _config(frequency: float) -> ComponentDVFSConfig:
    return ComponentDVFSConfig(
        policy=DVFSPolicy.CUSTOM,
        voltage_V=0.6,
        frequency_GHz=frequency,
    )


def test_configure_dvfs_for_op_assigns_all_eight_domains():
    plan = {
        "sa": _config(1.01),
        "vu": _config(1.02),
        "sram": _config(1.03),
        "hbm_mc": _config(1.04),
        "hbm_die": _config(1.05),
        "hbm_io": _config(1.06),
        "ici_mc": _config(1.07),
        "ici_phy": _config(1.08),
    }
    op = Operator()
    with patch.object(dvfs_optimizer, "get_dvfs_config", return_value=plan):
        power_analysis_lib.configure_dvfs_for_op(
            op, ChipConfig(), DVFSConfig(policy=DVFSPolicy.CUSTOM)
        )

    assert op.dvfs_sa is plan["sa"]
    assert op.dvfs_vu is plan["vu"]
    assert op.dvfs_sram is plan["sram"]
    assert op.dvfs_hbm_mc is plan["hbm_mc"]
    assert op.dvfs_hbm_die is plan["hbm_die"]
    assert op.dvfs_hbm_io is plan["hbm_io"]
    assert op.dvfs_ici_mc is plan["ici_mc"]
    assert op.dvfs_ici_phy is plan["ici_phy"]


def test_analyze_operator_energy_can_ignore_regulator_loss():
    op = Operator()
    op.stats.execution_time_ns = 100
    model_functions = {
        "scale_dvfs_component_time": DEFAULT,
        "add_op_dvfs_exe_time_overhead": DEFAULT,
        "analyze_dynamic_energy": DEFAULT,
        "analyze_sa_static_energy": DEFAULT,
        "analyze_vu_static_energy": DEFAULT,
        "analyze_vmem_static_energy": DEFAULT,
        "analyze_ici_static_energy": DEFAULT,
        "analyze_hbm_static_energy": DEFAULT,
        "analyze_other_static_energy": DEFAULT,
        "apply_regulator_efficiency": DEFAULT,
    }
    with patch.multiple(power_analysis_lib, **model_functions) as mocks:
        power_analysis_lib.analyze_operator_energy(
            op,
            ChipConfig(),
            PowerGatingConfig(),
            set_dvfs_config_for_op=False,
            ignore_vr_power_loss=True,
        )
        mocks["apply_regulator_efficiency"].assert_not_called()

        power_analysis_lib.analyze_operator_energy(
            op,
            ChipConfig(),
            PowerGatingConfig(),
            set_dvfs_config_for_op=False,
            ignore_vr_power_loss=False,
        )
        mocks["apply_regulator_efficiency"].assert_called_once_with(op)


def test_power_facade_exports_request_level_optimizer_delegate():
    assert power_analysis_lib.configure_dvfs_for_ops is dvfs_optimizer.configure_dvfs_for_ops
