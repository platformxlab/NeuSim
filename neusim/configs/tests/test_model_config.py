import unittest

from neusim.configs.models.ModelConfig import ModelConfig


class TestModelConfig(unittest.TestCase):
    def test_fleetsim_parallelism_identity(self):
        config = ModelConfig(
            model_type="llm",
            name="5p",
            microbatch_size_ici=2,
            microbatch_size_dcn=8,
            data_parallelism_degree=4,
            tensor_parallelism_degree=2,
            pipeline_parallelism_degree=3,
            data_parallel_degree_dcn=5,
            tensor_parallel_degree_dcn=6,
            pipeline_parallel_degree_dcn=7,
        )

        self.assertEqual(
            config.get_chip_version_and_parallelism_degree_tuple(),
            ("5p", 2, 8, 4, 2, 3, 5, 6, 7),
        )

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
