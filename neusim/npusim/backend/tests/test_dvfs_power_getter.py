import unittest

from neusim.npusim.frontend.Operator import DVFSPolicy, ComponentDVFSConfig
from neusim.npusim.backend.dvfs_power_getter import (
    get_power_from_dvfs,
    get_all_dvfs_configs_for_component,
    SA_POINTS,
    VU_POINTS,
    HBM_POINTS,
    ICI_POINTS,
    HBM_MC_POINTS,
    HBM_DIE_POINTS,
    HBM_IO_POINTS,
    ICI_MC_POINTS,
    ICI_PHY_POINTS,
    AGGREGATE_POWER_COMPONENTS,
    DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
    FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE,
)


class TestGetPowerFromDvfs(unittest.TestCase):
    def test_sa_none_policy_returns_max_perf(self):
        dvfs = ComponentDVFSConfig(policy=DVFSPolicy.NONE, voltage_V=0.7, frequency_GHz=1.7)
        dyn, static = get_power_from_dvfs("SA", dvfs)
        max_sa = max(SA_POINTS, key=lambda p: p.frequency_GHz)
        self.assertAlmostEqual(dyn, max_sa.dynamic_power_W)
        self.assertAlmostEqual(static, max_sa.static_power_W)

    def test_vu_none_policy_returns_max_perf(self):
        dvfs = ComponentDVFSConfig(policy=DVFSPolicy.NONE, voltage_V=0.7, frequency_GHz=1.7)
        dyn, static = get_power_from_dvfs("VU", dvfs)
        max_vu = max(VU_POINTS, key=lambda p: p.frequency_GHz)
        self.assertAlmostEqual(dyn, max_vu.dynamic_power_W)
        self.assertAlmostEqual(static, max_vu.static_power_W)

    def test_sa_ideal_at_known_point(self):
        dvfs = ComponentDVFSConfig(policy=DVFSPolicy.IDEAL, voltage_V=0.7, frequency_GHz=1.7)
        dyn, static = get_power_from_dvfs("SA", dvfs)
        self.assertGreater(dyn, 0)
        self.assertGreater(static, 0)

    def test_hbm_returns_positive(self):
        dvfs = ComponentDVFSConfig(policy=DVFSPolicy.IDEAL, voltage_V=0.7, frequency_GHz=1.7)
        dyn, static = get_power_from_dvfs("HBM", dvfs)
        self.assertGreater(dyn, 0)
        self.assertGreater(static, 0)

    def test_ici_returns_positive(self):
        dvfs = ComponentDVFSConfig(policy=DVFSPolicy.IDEAL, voltage_V=0.7, frequency_GHz=1.7)
        dyn, static = get_power_from_dvfs("ICI", dvfs)
        self.assertGreater(dyn, 0)
        self.assertGreater(static, 0)

    def test_unsupported_component_raises(self):
        dvfs = ComponentDVFSConfig(policy=DVFSPolicy.IDEAL, voltage_V=0.7, frequency_GHz=1.7)
        with self.assertRaises(ValueError):
            get_power_from_dvfs("UNSUPPORTED", dvfs)

    def test_zero_freq_returns_max_perf(self):
        dvfs = ComponentDVFSConfig(policy=DVFSPolicy.IDEAL, voltage_V=0.7, frequency_GHz=0.0)
        dyn, static = get_power_from_dvfs("SA", dvfs)
        max_sa = max(SA_POINTS, key=lambda p: p.frequency_GHz)
        self.assertAlmostEqual(dyn, max_sa.dynamic_power_W)


class TestGetAllDvfsConfigsForComponent(unittest.TestCase):
    def test_sa_configs_count(self):
        configs = get_all_dvfs_configs_for_component("sa", DVFSPolicy.IDEAL)
        self.assertEqual(len(configs), len(SA_POINTS))

    def test_hbm_configs_count(self):
        configs = get_all_dvfs_configs_for_component("hbm", DVFSPolicy.IDEAL)
        self.assertEqual(len(configs), len(HBM_POINTS))

    def test_legacy_search_names_stay_controller_only(self):
        self.assertIs(HBM_POINTS, HBM_MC_POINTS)
        self.assertIs(ICI_POINTS, ICI_MC_POINTS)
        self.assertEqual(
            get_all_dvfs_configs_for_component("HBM", DVFSPolicy.IDEAL),
            get_all_dvfs_configs_for_component("hbm_mc", DVFSPolicy.IDEAL),
        )
        self.assertEqual(
            get_all_dvfs_configs_for_component("ICI", DVFSPolicy.IDEAL),
            get_all_dvfs_configs_for_component("ici_mc", DVFSPolicy.IDEAL),
        )

    def test_unsupported_component_raises(self):
        with self.assertRaises(ValueError):
            get_all_dvfs_configs_for_component("UNKNOWN", DVFSPolicy.IDEAL)


class TestPaperCalibration(unittest.TestCase):
    def test_split_domain_table_endpoints(self):
        endpoints = {
            "hbm_mc": max(HBM_MC_POINTS, key=lambda p: p.bandwidth_GBs),
            "hbm_die": max(HBM_DIE_POINTS, key=lambda p: p.bandwidth_GBs),
            "hbm_io": max(HBM_IO_POINTS, key=lambda p: p.bandwidth_GBs),
            "ici_mc": max(ICI_MC_POINTS, key=lambda p: p.bandwidth_GBs),
            "ici_phy": max(ICI_PHY_POINTS, key=lambda p: p.bandwidth_GBs),
        }
        self.assertEqual(endpoints["hbm_mc"].bandwidth_GBs, 2755.0)
        self.assertEqual(endpoints["hbm_die"].static_power_W, 6.2)
        self.assertEqual(endpoints["hbm_io"].voltage_V, 0.9)
        self.assertEqual(endpoints["ici_mc"].bandwidth_GBs, 599.57)
        self.assertEqual(endpoints["ici_phy"].bandwidth_GBs, 599.57)

    def test_legacy_aggregate_names_sum_current_split_domains(self):
        config = ComponentDVFSConfig(
            policy=DVFSPolicy.IDEAL, voltage_V=0.55, frequency_GHz=1.0
        )
        for aggregate in ("hbm", "ici"):
            expected_dynamic = 0.0
            expected_static = 0.0
            for split_component in AGGREGATE_POWER_COMPONENTS[aggregate]:
                dynamic, static = get_power_from_dvfs(split_component, config)
                expected_dynamic += dynamic
                expected_static += static
            aggregate_dynamic, aggregate_static = get_power_from_dvfs(
                aggregate.upper(), config
            )
            self.assertAlmostEqual(aggregate_dynamic, expected_dynamic)
            self.assertAlmostEqual(aggregate_static, expected_static)

    def test_legacy_aggregate_peak_totals_leave_controllers_split(self):
        config = ComponentDVFSConfig(policy=DVFSPolicy.NONE)

        hbm_dynamic, hbm_static = get_power_from_dvfs("HBM", config)
        hbm_mc_dynamic, hbm_mc_static = get_power_from_dvfs("hbm_mc", config)
        hbm_split = [
            max(points, key=lambda point: point.bandwidth_GBs)
            for points in (HBM_MC_POINTS, HBM_DIE_POINTS, HBM_IO_POINTS)
        ]
        self.assertAlmostEqual(
            hbm_dynamic, sum(point.dynamic_power_W for point in hbm_split)
        )
        self.assertAlmostEqual(
            hbm_static, sum(point.static_power_W for point in hbm_split)
        )
        self.assertAlmostEqual(hbm_mc_dynamic, hbm_split[0].dynamic_power_W)
        self.assertAlmostEqual(hbm_mc_static, hbm_split[0].static_power_W)
        self.assertGreater(hbm_dynamic, hbm_mc_dynamic)
        self.assertGreater(hbm_static, hbm_mc_static)

        ici_dynamic, ici_static = get_power_from_dvfs("ICI", config)
        ici_mc_dynamic, ici_mc_static = get_power_from_dvfs("ici_mc", config)
        ici_split = [
            max(points, key=lambda point: point.bandwidth_GBs)
            for points in (ICI_MC_POINTS, ICI_PHY_POINTS)
        ]
        self.assertAlmostEqual(
            ici_dynamic, sum(point.dynamic_power_W for point in ici_split)
        )
        self.assertAlmostEqual(
            ici_static, sum(point.static_power_W for point in ici_split)
        )
        self.assertAlmostEqual(ici_mc_dynamic, ici_split[0].dynamic_power_W)
        self.assertAlmostEqual(ici_mc_static, ici_split[0].static_power_W)
        self.assertGreater(ici_dynamic, ici_mc_dynamic)
        self.assertGreater(ici_static, ici_mc_static)

    def test_regulator_calibration_snapshot(self):
        self.assertEqual(
            {point.scaling_time_ns for point in DVFS_VOLTAGE_REGULATOR_OVERHEAD_TABLE},
            {20},
        )
        fixed = {
            point.activity_factor: point.power_efficiency_percent
            for point in FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE
        }
        self.assertEqual(fixed[0.0], 89.0)
        self.assertEqual(fixed[1.0], 92.0)


if __name__ == "__main__":
    unittest.main()
