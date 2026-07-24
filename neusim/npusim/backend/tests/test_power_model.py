import unittest
from unittest.mock import MagicMock

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.power_gating.PowerGatingConfig import get_power_gating_config
from neusim.npusim.backend.power_model import (
    analyze_dynamic_energy,
    analyze_hbm_static_energy,
    analyze_ici_static_energy,
    apply_regulator_efficiency,
    compute_peak_sa_flops_per_sec_from_chip_config,
    compute_peak_sa_flops_per_sec_from_dvfs_config,
    compute_peak_vu_flops_per_sec_from_chip_config,
    cycle_to_ns,
    ns_to_cycle,
    scale_dvfs_component_time,
)
from neusim.npusim.frontend.Operator import (
    ComponentDVFSConfig,
    DVFSPolicy,
    Operator,
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


class TestSplitPowerAccounting(unittest.TestCase):
    @staticmethod
    def _peak_config(efficiency: float = 100.0) -> ComponentDVFSConfig:
        return ComponentDVFSConfig(
            policy=DVFSPolicy.NONE,
            voltage_V=0.7,
            frequency_GHz=1.7,
            voltage_regulator_scaling_time_ns=0,
            voltage_conversion_power_efficiency_percent=efficiency,
        )

    def _operator(self) -> Operator:
        op = Operator()
        op.stats.execution_time_ns = 1_000
        op.stats.memory_time_ns = 400
        op.stats.ici_time_ns = 250
        op.stats.memory_traffic_bytes = 1024
        for field in (
            "dvfs_sa",
            "dvfs_vu",
            "dvfs_sram",
            "dvfs_hbm_mc",
            "dvfs_hbm_die",
            "dvfs_hbm_io",
            "dvfs_ici_mc",
            "dvfs_ici_phy",
        ):
            setattr(op, field, self._peak_config())
        return op

    def test_split_energies_sum_to_compatibility_aggregates(self):
        op = self._operator()
        config = ChipConfig(enable_dvfs=True)
        no_pg = get_power_gating_config("NoPG")

        analyze_dynamic_energy(op, config)
        analyze_hbm_static_energy(op, config, no_pg)
        analyze_ici_static_energy(op, config, no_pg)

        self.assertGreater(op.stats.dynamic_energy_hbm_die_J, 0.0)
        self.assertGreater(op.stats.dynamic_energy_ici_phy_J, 0.0)
        self.assertEqual(
            op.stats.dynamic_energy_hbm_J,
            op.stats.dynamic_energy_hbm_mc_J
            + op.stats.dynamic_energy_hbm_die_J
            + op.stats.dynamic_energy_hbm_io_J,
        )
        self.assertEqual(
            op.stats.static_energy_hbm_J,
            op.stats.static_energy_hbm_mc_J
            + op.stats.static_energy_hbm_die_J
            + op.stats.static_energy_hbm_io_J,
        )
        self.assertEqual(
            op.stats.dynamic_energy_ici_J,
            op.stats.dynamic_energy_ici_mc_J
            + op.stats.dynamic_energy_ici_phy_J,
        )
        self.assertEqual(
            op.stats.static_energy_ici_J,
            op.stats.static_energy_ici_mc_J
            + op.stats.static_energy_ici_phy_J,
        )

    def test_disabled_dvfs_preserves_legacy_aggregate_totals(self):
        op = self._operator()
        config = ChipConfig(enable_dvfs=False)
        no_pg = get_power_gating_config("NoPG")

        analyze_dynamic_energy(op, config)
        analyze_hbm_static_energy(op, config, no_pg)
        analyze_ici_static_energy(op, config, no_pg)

        self.assertAlmostEqual(
            op.stats.dynamic_energy_hbm_J,
            config.dynamic_power_hbm_W * op.stats.memory_time_ns / 1e9,
        )
        self.assertAlmostEqual(
            op.stats.dynamic_energy_ici_J,
            config.dynamic_power_ici_W * op.stats.ici_time_ns / 1e9,
        )
        self.assertAlmostEqual(
            op.stats.static_energy_hbm_J,
            config.static_power_hbm_W * op.stats.execution_time_ns / 1e9,
        )
        self.assertAlmostEqual(
            op.stats.static_energy_ici_J,
            config.static_power_ici_W * op.stats.execution_time_ns / 1e9,
        )

    def test_regulator_scales_every_split_field_once(self):
        op = self._operator()
        for field in (
            "dynamic_energy_sa_J",
            "static_energy_sa_J",
            "dynamic_energy_vu_J",
            "static_energy_vu_J",
            "dynamic_energy_sram_J",
            "static_energy_sram_J",
            "dynamic_energy_hbm_mc_J",
            "static_energy_hbm_mc_J",
            "dynamic_energy_hbm_die_J",
            "static_energy_hbm_die_J",
            "dynamic_energy_hbm_io_J",
            "static_energy_hbm_io_J",
            "dynamic_energy_ici_mc_J",
            "static_energy_ici_mc_J",
            "dynamic_energy_ici_phy_J",
            "static_energy_ici_phy_J",
        ):
            setattr(op.stats, field, 1.0)
        for field in (
            "dvfs_sa",
            "dvfs_vu",
            "dvfs_sram",
            "dvfs_hbm_mc",
            "dvfs_hbm_die",
            "dvfs_hbm_io",
            "dvfs_ici_mc",
            "dvfs_ici_phy",
        ):
            setattr(op, field, self._peak_config(efficiency=50.0))

        apply_regulator_efficiency(op)

        for field in (
            "dynamic_energy_sa_J",
            "static_energy_sa_J",
            "dynamic_energy_vu_J",
            "static_energy_vu_J",
            "dynamic_energy_sram_J",
            "static_energy_sram_J",
            "dynamic_energy_hbm_mc_J",
            "static_energy_hbm_mc_J",
            "dynamic_energy_hbm_die_J",
            "static_energy_hbm_die_J",
            "dynamic_energy_hbm_io_J",
            "static_energy_hbm_io_J",
            "dynamic_energy_ici_mc_J",
            "static_energy_ici_mc_J",
            "dynamic_energy_ici_phy_J",
            "static_energy_ici_phy_J",
        ):
            self.assertEqual(getattr(op.stats, field), 2.0, field)

    def test_full_pg_with_zero_hbm_traffic_does_not_divide_by_zero(self):
        op = self._operator()
        op.stats.memory_time_ns = 0
        op.stats.memory_traffic_bytes = 0
        config = ChipConfig(enable_dvfs=False)
        full_pg = get_power_gating_config("Full")

        analyze_hbm_static_energy(op, config, full_pg)

        expected_mc = (
            config.static_power_hbm_mc_W
            * full_pg.hbm_power_level_factors[-1]
            * op.stats.execution_time_ns
            / 1e9
        )
        self.assertAlmostEqual(op.stats.static_energy_hbm_mc_J, expected_mc)
        self.assertAlmostEqual(
            op.stats.static_energy_hbm_die_J,
            config.static_power_hbm_die_W * op.stats.execution_time_ns / 1e9,
        )


if __name__ == "__main__":
    unittest.main()
