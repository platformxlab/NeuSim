import unittest
from neusim.configs.systems.SystemConfig import SystemConfig

class TestSystemConfig(unittest.TestCase):
    def test_system_config_instantiation(self):
        # Create a default SystemConfig
        config = SystemConfig()
        
        # Test default values
        self.assertEqual(config.PUE, 1.1)
        self.assertEqual(config.carbon_intensity_kgCO2_per_kWh, 0.5)

        # Test custom values
        config_custom = SystemConfig(PUE=1.2, carbon_intensity_kgCO2_per_kWh=0.6)
        self.assertEqual(config_custom.PUE, 1.2)
        self.assertEqual(config_custom.carbon_intensity_kgCO2_per_kWh, 0.6)
