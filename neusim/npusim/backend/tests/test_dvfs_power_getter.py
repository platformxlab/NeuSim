import unittest

from neusim.npusim.frontend.Operator import DVFSPolicy, ComponentDVFSConfig
from neusim.npusim.backend.dvfs_power_getter import (
    get_power_from_dvfs,
    get_all_dvfs_configs_for_component,
    SA_POINTS,
    VU_POINTS,
    HBM_POINTS,
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

    def test_unsupported_component_raises(self):
        with self.assertRaises(ValueError):
            get_all_dvfs_configs_for_component("UNKNOWN", DVFSPolicy.IDEAL)


if __name__ == "__main__":
    unittest.main()
