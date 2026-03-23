import unittest
from unittest.mock import MagicMock

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.frontend.Operator import DVFSPolicy, ComponentDVFSConfig
from neusim.npusim.backend.power_model import (
    compute_peak_sa_flops_per_sec_from_chip_config,
    compute_peak_vu_flops_per_sec_from_chip_config,
    compute_peak_sa_flops_per_sec_from_dvfs_config,
    cycle_to_ns,
    ns_to_cycle,
    scale_dvfs_component_time,
    apply_regulator_efficiency,
)


class TestComputePeakFlops(unittest.TestCase):
    def test_sa_flops_positive(self):
        config = ChipConfig()
        flops = compute_peak_sa_flops_per_sec_from_chip_config(config)
        self.assertGreater(flops, 0)

    def test_vu_flops_positive(self):
        config = ChipConfig()
        flops = compute_peak_vu_flops_per_sec_from_chip_config(config)
        self.assertGreater(flops, 0)

    def test_sa_flops_from_dvfs_with_zero_freq_falls_back(self):
        config = ChipConfig()
        dvfs = ComponentDVFSConfig(policy=DVFSPolicy.NONE, voltage_V=0.7, frequency_GHz=0.0)
        flops = compute_peak_sa_flops_per_sec_from_dvfs_config(config, dvfs)
        expected = compute_peak_sa_flops_per_sec_from_chip_config(config)
        self.assertAlmostEqual(flops, expected)


class TestCycleNsConversion(unittest.TestCase):
    def test_roundtrip(self):
        ns = cycle_to_ns(10, 1.7)
        cycles = ns_to_cycle(ns, 1.7)
        self.assertAlmostEqual(cycles, 10.0)

    def test_cycle_to_ns_known_value(self):
        ns = cycle_to_ns(17, 1.7)
        self.assertAlmostEqual(ns, 10.0)


class TestScaleDvfsComponentTime(unittest.TestCase):
    def test_half_freq_doubles_time(self):
        op = MagicMock()
        op.stats.sa_time_ns = 100
        op.stats.vu_time_ns = 100
        op.stats.vmem_time_ns = 100
        op.stats.ici_time_ns = 100
        op.stats.memory_time_ns = 100

        config = ChipConfig()
        # Use exactly half the base frequency
        half_freq = ComponentDVFSConfig(
            policy=DVFSPolicy.IDEAL, voltage_V=0.5, frequency_GHz=config.freq_GHz / 2
        )
        op.dvfs_sa = half_freq
        op.dvfs_vu = half_freq
        op.dvfs_sram = half_freq
        op.dvfs_ici = half_freq
        op.dvfs_hbm = half_freq

        scale_dvfs_component_time(op, config)
        self.assertEqual(op.stats.sa_time_ns, 200)


class TestApplyRegulatorEfficiency(unittest.TestCase):
    def test_scales_energies(self):
        op = MagicMock()
        op.stats.dynamic_energy_sa_J = 1.0
        op.stats.static_energy_sa_J = 1.0
        op.stats.dynamic_energy_vu_J = 1.0
        op.stats.static_energy_vu_J = 1.0
        op.stats.dynamic_energy_sram_J = 1.0
        op.stats.static_energy_sram_J = 1.0
        op.stats.dynamic_energy_hbm_J = 1.0
        op.stats.static_energy_hbm_J = 1.0
        op.stats.dynamic_energy_ici_J = 1.0
        op.stats.static_energy_ici_J = 1.0

        eff_cfg = ComponentDVFSConfig(
            policy=DVFSPolicy.NONE,
            voltage_V=0.7,
            frequency_GHz=1.7,
            voltage_conversion_power_efficiency_percent=50.0,
        )
        op.dvfs_sa = eff_cfg
        op.dvfs_vu = eff_cfg
        op.dvfs_sram = eff_cfg
        op.dvfs_hbm = eff_cfg
        op.dvfs_ici = eff_cfg

        apply_regulator_efficiency(op)
        # 100/50 = 2x scaling
        self.assertAlmostEqual(op.stats.dynamic_energy_sa_J, 2.0)
        self.assertAlmostEqual(op.stats.static_energy_sa_J, 2.0)
        self.assertAlmostEqual(op.stats.dynamic_energy_hbm_J, 2.0)


if __name__ == "__main__":
    unittest.main()
