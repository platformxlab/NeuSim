import unittest
from neusim.configs.models.LLMConfig import LLMConfig, MoELLMConfig, DeepSeekConfig

class TestLLMConfig(unittest.TestCase):
    def test_llm_config(self):
        # Test default init
        config = LLMConfig()
        # Default: num_heads=64, num_kv_heads should default to num_heads if not provided
        self.assertEqual(config.num_heads, 64)
        self.assertEqual(config.num_kv_heads, 64)

        # Test valid init with num_kv_heads
        config_mqa = LLMConfig(num_kv_heads=1)
        self.assertEqual(config_mqa.num_kv_heads, 1)

        # Test hash
        config1 = LLMConfig()
        config2 = LLMConfig()
        self.assertEqual(hash(config1), hash(config2))

class TestMoELLMConfig(unittest.TestCase):
    def test_moe_llm_config(self):
        # Test default init
        config = MoELLMConfig()
        # Default: d_ff=11008, moe_d_ff should default to d_ff
        self.assertEqual(config.d_ff, 11008)
        self.assertEqual(config.moe_d_ff, 11008)
        
        # Test custom moe_d_ff
        config_custom = MoELLMConfig(moe_d_ff=2048)
        self.assertEqual(config_custom.moe_d_ff, 2048)

        # Test expert_tensor_parallelism_degree
        # dp=1, tp=1, ep=1 -> 1*1 // 1 = 1
        self.assertEqual(config.expert_tensor_parallelism_degree, 1)
        
        # dp=2, tp=2, ep=2 -> 2*2 // 2 = 2
        config_parallel = MoELLMConfig(
            data_parallelism_degree=2,
            tensor_parallelism_degree=2,
            expert_parallelism_degree=2
        )
        self.assertEqual(config_parallel.expert_tensor_parallelism_degree, 2)

        # Test num_expert_tensor_parallel_axes
        # ndp=1, ntp=1, nep=1 -> 1+1-1 = 1
        self.assertEqual(config.num_expert_tensor_parallel_axes, 1)
        
        # ndp=2, ntp=2, nep=2 -> 2+2-2 = 2
        config_axes = MoELLMConfig(
            num_data_parallel_axes=2,
            num_tensor_parallel_axes=2,
            num_expert_parallel_axes=2
        )
        self.assertEqual(config_axes.num_expert_tensor_parallel_axes, 2)

        # Test num_experts_per_token
        # shared=1, routed=8 -> 9
        self.assertEqual(config.num_experts_per_token, 9)

        # Test hash
        config1 = MoELLMConfig()
        config2 = MoELLMConfig()
        self.assertEqual(hash(config1), hash(config2))

    def test_moe_load_imbalance_defaults(self):
        # Defaults: E=256, K=8 -> E/K = 32
        config = MoELLMConfig()
        self.assertEqual(config.expert_load_imbalance_factor, -1.0)
        self.assertFalse(config.all_to_all_load_imbalance_aware)
        self.assertEqual(config.num_worst_case_experts, 1)
        # -1.0 sentinel auto-resolves to E/K (worst case)
        self.assertEqual(config.effective_expert_load_imbalance_factor, 256 / 8)

    def test_effective_expert_load_imbalance_factor_explicit(self):
        config = MoELLMConfig(expert_load_imbalance_factor=2.0)
        self.assertEqual(config.effective_expert_load_imbalance_factor, 2.0)

    def test_get_effective_expert_tokens_worst_case(self):
        # f = -1 (auto = E/K = 32): T*K/E*f = T -> all tokens land on the hot expert
        config = MoELLMConfig()
        self.assertEqual(config.get_effective_expert_tokens(256), 256)
        self.assertEqual(config.get_effective_expert_tokens(1), 1)

    def test_get_effective_expert_tokens_balanced(self):
        # f = 1.0 (balanced): each expert gets T*K/E tokens. T=256,K=8,E=256 -> 8
        config = MoELLMConfig(expert_load_imbalance_factor=1.0)
        self.assertEqual(config.get_effective_expert_tokens(256), 8)
        # at least 1 token even when the balanced share rounds below 1
        self.assertEqual(config.get_effective_expert_tokens(1), 1)

    def test_get_effective_expert_tokens_ceil(self):
        # Fractional share is rounded up: T=100,K=8,E=256,f=2.0 -> ceil(6.25) = 7
        config = MoELLMConfig(expert_load_imbalance_factor=2.0)
        self.assertEqual(config.get_effective_expert_tokens(100), 7)

    def test_expert_load_imbalance_factor_validation(self):
        # below 1.0 is invalid
        with self.assertRaises(ValueError):
            MoELLMConfig(expert_load_imbalance_factor=0.5)
        # above E/K (=32) is invalid
        with self.assertRaises(ValueError):
            MoELLMConfig(expert_load_imbalance_factor=64.0)
        # boundary values are valid
        MoELLMConfig(expert_load_imbalance_factor=1.0)
        MoELLMConfig(expert_load_imbalance_factor=32.0)

    def test_num_worst_case_experts_validation(self):
        # must be in [1, K=8]
        with self.assertRaises(ValueError):
            MoELLMConfig(num_worst_case_experts=0)
        with self.assertRaises(ValueError):
            MoELLMConfig(num_worst_case_experts=9)
        # boundaries valid
        MoELLMConfig(num_worst_case_experts=1)
        MoELLMConfig(num_worst_case_experts=8)

class TestDeepSeekConfig(unittest.TestCase):
    def test_deepseek_config(self):
        config = DeepSeekConfig(
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=64,
            qk_nope_head_dim=64,
            v_head_dim=128
        )
        
        # Test qk_head_dim
        # 64 + 64 = 128
        self.assertEqual(config.qk_head_dim, 128)
        
        # Test hash
        config2 = DeepSeekConfig(
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=64,
            qk_nope_head_dim=64,
            v_head_dim=128
        )
        self.assertEqual(hash(config), hash(config2))
