import unittest
from neusim.configs.models.DiTConfig import DiTConfig

class TestDiTConfig(unittest.TestCase):
    def test_dit_config_instantiation(self):
        # Create DiTConfig
        config = DiTConfig(
            image_width=256,
            num_channels=3,
            patch_size=16,
            num_diffusion_steps=1000
        )
        
        self.assertEqual(config.model_type, "dit")
        self.assertEqual(config.image_width, 256)
        self.assertEqual(config.num_channels, 3)
        self.assertEqual(config.input_seqlen, 0)
