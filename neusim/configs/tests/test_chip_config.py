import json
import unittest
from pathlib import Path

from neusim.configs.chips.ChipConfig import ChipConfig


class TestChipConfig(unittest.TestCase):
    def test_chip_config_properties(self):
        # Create a default ChipConfig
        config = ChipConfig()

        # Test vmem_bw_GBps
        # Default: num_vu_ports=6, freq_GHz=1.75
        # Expected: 6 * 1.75 * 8 * 128 * 4 = 43008.0
        self.assertAlmostEqual(config.vmem_bw_GBps, 43008.0)

        # Test static_power_hbm_W
        # Default split calibration: MC=10.264041296, die=6.2, I/O=0.
        expected_static_hbm = 10.264041296 + 6.2
        self.assertAlmostEqual(config.static_power_hbm_W, expected_static_hbm)

        # Test peak_SA_tflops_per_sec
        # Default: num_sa=8, sa_dim=128, freq_GHz=1.75
        # Expected: 8 * (128^2) * 2 * 1.75 * 1e9 / 1e12 = 0.458752
        expected_sa_tflops = 8 * (128 ** 2) * 2 * 1.75 / 1000
        self.assertAlmostEqual(config.peak_SA_tflops_per_sec, expected_sa_tflops)

        # Test peak_VU_tflops_per_sec
        # Default: num_vu=6, freq_GHz=1.75
        # Expected: 6 * (8 * 128) * 1.75 * 1e9 / 1e12 = 0.010752
        expected_vu_tflops = 6 * (8 * 128) * 1.75 / 1000
        self.assertAlmostEqual(config.peak_VU_tflops_per_sec, expected_vu_tflops)

        # Test peak_tflops_per_sec
        self.assertAlmostEqual(config.peak_tflops_per_sec, expected_sa_tflops + expected_vu_tflops)

        # Test static_power_sa_W
        # Default: static_power_W_per_sa=1.35868996, num_sa=8
        expected_static_sa = 1.35868996 * 8
        self.assertAlmostEqual(config.static_power_sa_W, expected_static_sa)

        # Test static_power_vu_W
        # Default: static_power_W_per_vu=0.475076728, num_vu=6
        expected_static_vu = 0.475076728 * 6
        self.assertAlmostEqual(config.static_power_vu_W, expected_static_vu)

        # Test static_power_vmem_W_per_MB
        # Default: static_power_vmem_W=24.21353615, vmem_size_MB=128
        expected_static_vmem_per_mb = 24.21353615 / 128
        self.assertAlmostEqual(config.static_power_vmem_W_per_MB, expected_static_vmem_per_mb)

        # Test static_power_W
        # Sum of components
        expected_static_total = (
            expected_static_sa +
            expected_static_vu +
            config.static_power_vmem_W +
            config.static_power_ici_W +
            expected_static_hbm +
            config.static_power_other_W
        )
        self.assertAlmostEqual(config.static_power_W, expected_static_total)

        # Test idle_power_W
        self.assertAlmostEqual(config.idle_power_W, expected_static_total)

        # Test dynamic_power_sa_W
        # Default: dynamic_power_W_per_SA=28.19413333, num_sa=8
        expected_dynamic_sa = 28.19413333 * 8
        self.assertAlmostEqual(config.dynamic_power_sa_W, expected_dynamic_sa)

        # Test dynamic_power_vu_W
        # Default: dynamic_power_W_per_VU=2.65216, num_vu=6
        expected_dynamic_vu = 2.65216 * 6
        self.assertAlmostEqual(config.dynamic_power_vu_W, expected_dynamic_vu)

        # Test dynamic_power_hbm_W
        # Default: hbm_bw_GBps=2765, dynamic_power_hbm_W_per_GBps=0.01261538462
        expected_dynamic_hbm = 2765 * 0.01261538462
        self.assertAlmostEqual(config.dynamic_power_hbm_W, expected_dynamic_hbm)

        # Test dynamic_power_ici_W
        # Default: ici_bw_GBps=200, dynamic_power_ici_W_per_GBps=0.01767315271
        expected_dynamic_ici = 200 * 0.01767315271
        self.assertAlmostEqual(config.dynamic_power_ici_W, expected_dynamic_ici)

        # Test dynamic_power_peak_W
        # Sum of components
        expected_dynamic_total = (
            expected_dynamic_sa +
            expected_dynamic_vu +
            config.dynamic_power_vmem_W +
            expected_dynamic_ici +
            expected_dynamic_hbm +
            config.dynamic_power_other_W
        )
        self.assertAlmostEqual(config.dynamic_power_peak_W, expected_dynamic_total)

        # Test total_power_peak_W
        self.assertAlmostEqual(config.total_power_peak_W, expected_static_total + expected_dynamic_total)


    def test_legacy_hbm_phy_migration(self):
        config = ChipConfig(
            static_power_hbm_mc_W=10.0,
            static_power_hbm_phy_W=15.0,
        )
        self.assertEqual(config.static_power_hbm_die_W, 15.0)
        self.assertEqual(config.static_power_hbm_io_W, 0.0)
        self.assertEqual(config.static_power_hbm_phy_W, 15.0)
        self.assertEqual(config.static_power_hbm_W, 25.0)

        # Explicit split values take precedence over a stale legacy aggregate.
        explicit = ChipConfig(
            static_power_hbm_phy_W=99.0,
            static_power_hbm_die_W=6.2,
            static_power_hbm_io_W=1.0,
        )
        self.assertEqual(explicit.static_power_hbm_phy_W, 7.2)

    def test_checked_in_v5p_uses_explicit_corrected_calibration(self):
        config_path = Path(__file__).parents[3] / "configs/chips/tpuv5p.json"
        config = ChipConfig(**json.loads(config_path.read_text()))
        self.assertEqual(config.static_power_hbm_die_W, 6.2)
        self.assertEqual(config.static_power_hbm_io_W, 0.0)
        self.assertEqual(config.static_power_other_W, 44.82811018)
