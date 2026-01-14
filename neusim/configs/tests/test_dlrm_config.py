import unittest
from neusim.configs.models.DLRMConfig import DLRMConfig, MLPLayerConfig

class TestDLRMConfig(unittest.TestCase):
    def test_dlrm_config_instantiation(self):
        # Create DLRMConfig
        bottom_mlp = [MLPLayerConfig(in_features=10, out_features=20)]
        top_mlp = [MLPLayerConfig(in_features=20, out_features=1)]
        
        config = DLRMConfig(
            model_type="dlrm",
            embedding_dim=64,
            num_indices_per_lookup=[100, 200],
            embedding_table_sizes=[1000, 2000],
            num_dense_features=10,
            bottom_mlp_config=bottom_mlp,
            top_mlp_config=top_mlp
        )
        
        self.assertEqual(config.model_type, "dlrm")
        self.assertEqual(config.embedding_dim, 64)
        self.assertEqual(len(config.bottom_mlp_config), 1)
        self.assertEqual(config.bottom_mlp_config[0].in_features, 10)
