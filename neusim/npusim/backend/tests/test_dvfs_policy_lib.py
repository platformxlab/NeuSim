import unittest

from neusim.npusim.frontend.Operator import DVFSPolicy, DVFSConfig, ComponentDVFSConfig
from neusim.npusim.backend.dvfs_policy_lib import (
    slowdown_freq,
    pick_v_from_freq,
    get_dvfs_policy_None,
    get_dvfs_config,
    SA_VF_TABLE,
)


class TestSlowdownFreq(unittest.TestCase):
    def test_zero_ratio_returns_base(self):
        self.assertEqual(slowdown_freq(0.0, 1.7), 1.7)

    def test_negative_ratio_returns_base(self):
        self.assertEqual(slowdown_freq(-1.0, 1.7), 1.7)

    def test_positive_ratio_slows_down(self):
        f = slowdown_freq(1.0, 1.7)
        self.assertAlmostEqual(f, 0.85)

    def test_large_ratio_clamps_to_min(self):
        f = slowdown_freq(1000.0, 1.7, min_freq_GHz=0.05)
        self.assertAlmostEqual(f, 0.05)


class TestPickVFromFreq(unittest.TestCase):
    def test_zero_freq_returns_zero(self):
        self.assertEqual(pick_v_from_freq(0.0, SA_VF_TABLE), 0.0)

    def test_below_min_returns_first_voltage(self):
        v = pick_v_from_freq(0.1, SA_VF_TABLE)
        self.assertEqual(v, 0.45)

    def test_above_max_returns_last_voltage(self):
        v = pick_v_from_freq(2.0, SA_VF_TABLE)
        self.assertEqual(v, 0.70)

    def test_mid_range_returns_correct_band(self):
        v = pick_v_from_freq(1.0, SA_VF_TABLE)
        self.assertEqual(v, 0.55)


class TestGetDvfsPolicyNone(unittest.TestCase):
    def test_returns_all_components(self):
        plan = get_dvfs_policy_None()
        self.assertIn("sa", plan)
        self.assertIn("vu", plan)
        self.assertIn("sram", plan)
        self.assertIn("hbm", plan)
        self.assertIn("ici", plan)

    def test_all_none_policy(self):
        plan = get_dvfs_policy_None()
        for comp_cfg in plan.values():
            self.assertEqual(comp_cfg.policy, DVFSPolicy.NONE)
            self.assertAlmostEqual(comp_cfg.voltage_V, 0.7)
            self.assertAlmostEqual(comp_cfg.frequency_GHz, 1.7)


class TestGetDvfsConfig(unittest.TestCase):
    def test_none_policy_dispatches(self):
        from unittest.mock import MagicMock
        op = MagicMock()
        from neusim.configs.chips.ChipConfig import ChipConfig
        config = ChipConfig()
        dvfs_cfg = DVFSConfig(policy=DVFSPolicy.NONE)
        plan = get_dvfs_config(op, config, dvfs_cfg)
        self.assertIn("sa", plan)
        self.assertEqual(plan["sa"].policy, DVFSPolicy.NONE)

    def test_unsupported_policy_raises(self):
        from unittest.mock import MagicMock
        op = MagicMock()
        from neusim.configs.chips.ChipConfig import ChipConfig
        config = ChipConfig()
        dvfs_cfg = DVFSConfig()
        dvfs_cfg.policy = "UNSUPPORTED"
        with self.assertRaises((ValueError, AttributeError)):
            get_dvfs_config(op, config, dvfs_cfg)


if __name__ == "__main__":
    unittest.main()
