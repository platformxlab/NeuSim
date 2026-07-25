import unittest

from neusim.npusim.frontend.Operator import Operator, DVFSPolicy, DVFSConfig, ComponentDVFSConfig
from neusim.npusim.backend.dvfs_power_getter import (
    SA_POINTS,
    SRAM_POINTS,
    VU_POINTS,
)
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


class TestIntegratedPolicies(unittest.TestCase):
    @staticmethod
    def _memory_bound_op() -> Operator:
        op = Operator(name="memory_bound")
        op.stats.execution_time_ns = 100
        op.stats.sa_time_ns = 20
        op.stats.vu_time_ns = 30
        op.stats.vmem_time_ns = 40
        op.stats.memory_time_ns = 100
        op.stats.ici_time_ns = 0
        op.stats.bounded_by = "Memory"
        return op

    def test_direct_customall_couples_full_hbm_but_custom_does_not(self):
        from neusim.configs.chips.ChipConfig import ChipConfig

        op = self._memory_bound_op()
        chip = ChipConfig(freq_GHz=1.7)
        custom = get_dvfs_config(
            op, chip, DVFSConfig(policy=DVFSPolicy.CUSTOM)
        )
        custom_all = get_dvfs_config(
            op, chip, DVFSConfig(policy=DVFSPolicy.CUSTOM_ALL)
        )

        self.assertEqual(custom["hbm_die"].policy, DVFSPolicy.NONE)
        self.assertEqual(custom["hbm_io"].policy, DVFSPolicy.NONE)
        self.assertEqual(custom_all["hbm_die"].policy, DVFSPolicy.CUSTOM)
        self.assertEqual(custom_all["hbm_io"].policy, DVFSPolicy.CUSTOM)
        self.assertEqual(
            custom_all["hbm_die"].frequency_GHz,
            custom_all["hbm_mc"].frequency_GHz,
        )
        self.assertEqual(
            custom_all["hbm_io"].frequency_GHz,
            custom_all["hbm_mc"].frequency_GHz,
        )

    def test_dom3_couples_compute_voltage_and_frequency(self):
        from neusim.configs.chips.ChipConfig import ChipConfig

        plan = get_dvfs_config(
            self._memory_bound_op(),
            ChipConfig(freq_GHz=1.7),
            DVFSConfig(
                policy=DVFSPolicy.CUSTOM_ALL,
                custom_compute_domain_mode="dom3",
            ),
        )
        compute_vf = {
            (plan[name].voltage_V, plan[name].frequency_GHz)
            for name in ("sa", "vu", "sram")
        }
        self.assertEqual(len(compute_vf), 1)

    def test_dvfsc_couples_compute_and_leaves_memory_at_peak(self):
        from neusim.configs.chips.ChipConfig import ChipConfig

        plan = get_dvfs_config(
            self._memory_bound_op(),
            ChipConfig(freq_GHz=1.7),
            DVFSConfig(policy=DVFSPolicy.DVFS_C),
        )
        compute_vf = {
            (plan[name].voltage_V, plan[name].frequency_GHz)
            for name in ("sa", "vu", "sram")
        }
        self.assertEqual(len(compute_vf), 1)
        shared_frequency = plan["sa"].frequency_GHz
        self.assertIsNotNone(shared_frequency)

        def required_voltage(points):
            for point in sorted(points, key=lambda item: item.frequency_GHz):
                if point.frequency_GHz >= shared_frequency - 1e-9:
                    return point.voltage_V
            return points[-1].voltage_V

        expected_voltage = max(
            required_voltage(points)
            for points in (SA_POINTS, VU_POINTS, SRAM_POINTS)
        )
        self.assertEqual(plan["sa"].voltage_V, expected_voltage)
        self.assertEqual(plan["hbm_mc"].policy, DVFSPolicy.NONE)
        self.assertEqual(plan["ici_mc"].policy, DVFSPolicy.NONE)
        self.assertIs(plan["hbm"], plan["hbm_mc"])
        self.assertIs(plan["ici"], plan["ici_mc"])


if __name__ == "__main__":
    unittest.main()
