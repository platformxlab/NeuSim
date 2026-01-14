import unittest
from neusim.configs.models.ModelConfig import ModelConfig

class TestModelConfig(unittest.TestCase):
    def test_model_config_hash(self):
        # Create two identical ModelConfigs
        config1 = ModelConfig(
            model_type="llm",
            model_name="test_model",
            name="test_chip",
            global_batch_size=8,
            num_chips=4
        )
        config2 = ModelConfig(
            model_type="llm",
            model_name="test_model",
            name="test_chip",
            global_batch_size=8,
            num_chips=4
        )
        
        # Check if their hash is the same
        self.assertEqual(hash(config1), hash(config2))

        # Create a different ModelConfig
        config3 = ModelConfig(
            model_type="llm",
            model_name="test_model_diff",
            name="test_chip",
            global_batch_size=8,
            num_chips=4
        )
        
        # Check if their hash is different (likely)
        self.assertNotEqual(hash(config1), hash(config3))
