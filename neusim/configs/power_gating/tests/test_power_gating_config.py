import unittest
from math import ceil

from neusim.configs.power_gating.PowerGatingConfig import (
    PowerGatingConfig,
    get_power_gating_config,
)


class TestPowerGatingConfig(unittest.TestCase):
    def test_default_construction(self):
        pg = PowerGatingConfig()
        self.assertEqual(pg.name, "PowerGatingConfig")
        self.assertFalse(pg.SA_PG_enabled)
        self.assertFalse(pg.VU_PG_enabled)
        self.assertFalse(pg.vmem_PG_enabled)
        self.assertFalse(pg.ici_PG_enabled)
        self.assertFalse(pg.hbm_PG_enabled)
        self.assertFalse(pg.other_PG_enabled)

    def test_NoPG(self):
        pg = get_power_gating_config("NoPG")
        self.assertEqual(pg.name, "NoPG")
        self.assertFalse(pg.SA_PG_enabled)

    def test_disabled_alias(self):
        pg = get_power_gating_config("disabled")
        self.assertEqual(pg.name, "NoPG")

    def test_ideal_inst_component(self):
        pg = get_power_gating_config("ideal_inst_component")
        self.assertTrue(pg.SA_PG_enabled)
        self.assertEqual(
            pg.SA_temporal_granularity,
            PowerGatingConfig.TemporalGranularity.INSTRUCTION,
        )
        self.assertEqual(
            pg.SA_spatial_granularity,
            PowerGatingConfig.SASpatialGranularity.COMPONENT,
        )
        self.assertTrue(pg.VU_PG_enabled)
        self.assertTrue(pg.vmem_PG_enabled)
        self.assertTrue(pg.ici_PG_enabled)

    def test_ideal_op_component(self):
        pg = get_power_gating_config("ideal_op_component")
        self.assertEqual(
            pg.SA_temporal_granularity,
            PowerGatingConfig.TemporalGranularity.OPERATOR,
        )

    def test_Ideal(self):
        pg = get_power_gating_config("Ideal")
        self.assertEqual(pg.name, "Ideal")
        self.assertTrue(pg.SA_PG_enabled)
        self.assertEqual(
            pg.SA_spatial_granularity,
            PowerGatingConfig.SASpatialGranularity.PE,
        )
        self.assertEqual(pg.sa_pe_pg_delay_cycles, 0)
        self.assertTrue(pg.hbm_PG_enabled)
        self.assertEqual(pg.sa_power_level_factors, [1.0, 0.0])

    def test_ideal_inst_PE_ALU_alias(self):
        pg = get_power_gating_config("ideal_inst_PE_ALU")
        self.assertEqual(pg.name, "Ideal")

    def test_Base(self):
        pg = get_power_gating_config("Base")
        self.assertEqual(pg.name, "Base")
        self.assertEqual(pg.sa_power_level_factors, [1.0, 0.05])
        self.assertEqual(pg.sa_pe_pg_delay_cycles, 1)
        self.assertEqual(
            pg.SA_spatial_granularity,
            PowerGatingConfig.SASpatialGranularity.COMPONENT,
        )

    def test_HW(self):
        pg = get_power_gating_config("HW")
        self.assertEqual(pg.name, "HW")
        self.assertEqual(
            pg.SA_spatial_granularity,
            PowerGatingConfig.SASpatialGranularity.PE,
        )

    def test_Full(self):
        pg = get_power_gating_config("Full")
        self.assertEqual(pg.name, "Full")
        self.assertEqual(
            pg.VU_PG_policy,
            PowerGatingConfig.PowerGatingPolicy.SW,
        )
        self.assertEqual(
            pg.vmem_PG_policy,
            PowerGatingConfig.PowerGatingPolicy.SW,
        )

    def test_vary_Vth(self):
        pg = get_power_gating_config("Full_vary_Vth_0.1_0.2")
        self.assertEqual(pg.name, "Full_vary_Vth_0.1_0.2")
        self.assertAlmostEqual(pg.sa_power_level_factors[-1], 0.1)
        self.assertAlmostEqual(pg.vmem_power_level_factors[-1], 0.2)
        self.assertAlmostEqual(pg.vu_power_level_factors[-1], 0.1)

    def test_vary_PG_delay(self):
        base = get_power_gating_config("Base")
        pg = get_power_gating_config("Base_vary_PG_delay_2.0")
        self.assertEqual(pg.name, "Base_vary_PG_delay_2.0")
        self.assertEqual(pg.sa_pg_delay_cycles, ceil(base.sa_pg_delay_cycles * 2.0))
        self.assertEqual(pg.vu_pg_delay_cycles, ceil(base.vu_pg_delay_cycles * 2.0))

    def test_invalid_name_raises(self):
        with self.assertRaises(ValueError):
            get_power_gating_config("nonexistent_config")


if __name__ == "__main__":
    unittest.main()
