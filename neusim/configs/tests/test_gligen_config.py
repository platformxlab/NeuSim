import unittest
from neusim.configs.models.GLIGENConfig import GLIGENConfig

class TestGLIGENConfig(unittest.TestCase):
    def test_gligen_config_instantiation(self):
        # Create GLIGENConfig
        config = GLIGENConfig()
        
        self.assertEqual(config.model_type, "gligen")
        self.assertEqual(config.num_diffusion_steps, 1)
        
        # Test nested configs default values
        self.assertEqual(config.fourier_embedder_config.num_freqs, 64)
        self.assertEqual(config.text_embedder_config.d_model, 512)
        self.assertEqual(config.image_embedder_config.d_model, 1024)
        self.assertEqual(config.spatial_condition_embedder_config.stem.in_channels, 3)
        self.assertEqual(config.grounding_input_config.text.input_seqlen, 512)
        self.assertEqual(config.unet_config.model_channels, 320)
