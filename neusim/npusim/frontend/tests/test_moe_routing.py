'''
Tests for MoE expert-routing load-imbalance modeling in llm_ops_lib:
- _all_to_all_receiver_skew (dispatch/combine incast skew under load imbalance)
- create_all_to_all_op receiver_skew scaling
- create_ffn_deepseek_moe worst-case-device expert computation
'''
import types
import unittest

from neusim.configs.models.LLMConfig import MoELLMConfig, DeepSeekConfig
from neusim.npusim.frontend.Operator import Tensor
from neusim.npusim.frontend import llm_ops_lib


def _deepseek_config(**overrides) -> DeepSeekConfig:
    kwargs = dict(
        d_model=512,
        moe_d_ff=256,
        num_routed_experts=256,
        num_activated_routed_experts_per_token=8,
        num_shared_experts=1,
        num_limited_groups=4,
        global_batch_size=1,
        # required MLA fields (no defaults)
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        v_head_dim=128,
    )
    kwargs.update(overrides)
    return DeepSeekConfig(**kwargs)


class TestAllToAllReceiverSkew(unittest.TestCase):
    def test_skew_is_one_without_expert_parallelism(self):
        # EP <= 1 -> always balanced, even with the flag on.
        config = MoELLMConfig(all_to_all_load_imbalance_aware=True)
        self.assertEqual(
            llm_ops_lib._all_to_all_receiver_skew(config, total_tokens=256, expert_parallelism_degree=1),
            1.0,
        )

    def test_skew_is_one_when_flag_disabled(self):
        # Default flag is False -> balanced model regardless of EP.
        config = MoELLMConfig()
        self.assertFalse(config.all_to_all_load_imbalance_aware)
        self.assertEqual(
            llm_ops_lib._all_to_all_receiver_skew(config, total_tokens=256, expert_parallelism_degree=8),
            1.0,
        )

    def test_balanced_load_gives_unit_skew(self):
        # f = 1.0 (perfectly balanced): hottest group == average group -> skew 1.0.
        config = MoELLMConfig(
            all_to_all_load_imbalance_aware=True,
            expert_load_imbalance_factor=1.0,
        )
        skew = llm_ops_lib._all_to_all_receiver_skew(config, total_tokens=256, expert_parallelism_degree=8)
        self.assertAlmostEqual(skew, 1.0)

    def test_balanced_load_unit_skew_at_small_token_counts(self):
        # Regression for the floor-of-1 inflation: a balanced load (f=1.0) must give
        # skew 1.0 at ANY token count, including decode (T=1), where the earlier
        # floored model wrongly reported 32.0 / 4.0 / 2.0 for T=1 / 8 / 16.
        config = MoELLMConfig(
            all_to_all_load_imbalance_aware=True,
            expert_load_imbalance_factor=1.0,
        )
        for T in (1, 2, 8, 16):
            skew = llm_ops_lib._all_to_all_receiver_skew(config, total_tokens=T, expert_parallelism_degree=8)
            self.assertAlmostEqual(skew, 1.0, msg=f"balanced skew should be 1.0 at T={T}, got {skew}")

    def test_worst_case_load_gives_skew_above_one(self):
        # Default f = -1 -> E/K worst case. Real-unit anchor (T=256, E=256, K=8, EP=8, W=1):
        #   experts_per_group = 32, total_routings = 2048, eff_real = 256,
        #   rem_real = (2048-256)/255 = 1792/255,
        #   hot_group = 256 + 31*1792/255 = 120832/255, avg_group = 256
        #   -> skew = 120832/255/256 = 120832/65280 ~= 1.851.
        config = MoELLMConfig(all_to_all_load_imbalance_aware=True)
        skew = llm_ops_lib._all_to_all_receiver_skew(config, total_tokens=256, expert_parallelism_degree=8)
        self.assertGreater(skew, 1.0)
        self.assertAlmostEqual(skew, 120832 / 65280)

    def test_skew_uses_ceil_experts_per_group_when_ep_indivisible(self):
        # When EP does not divide E, the hottest group holds ceil(E/EP) experts.
        # E=10, K=2, EP=3, f=-1 (->E/K=5), W=1: experts_per_group=ceil(10/3)=4,
        #   eff_real=T*K/2, rem_real=(T*K/2)/9,
        #   hot_group = eff_real + 3*rem_real = (1/2 + 1/6)*T*K = (2/3)*T*K,
        #   avg_group = T*K/3 -> skew = 2.0. (Floored E//EP=3 would give 11/6 ~= 1.833.)
        config = MoELLMConfig(
            all_to_all_load_imbalance_aware=True,
            num_routed_experts=10,
            num_activated_routed_experts_per_token=2,
        )
        skew = llm_ops_lib._all_to_all_receiver_skew(config, total_tokens=64, expert_parallelism_degree=3)
        self.assertAlmostEqual(skew, 2.0)

    def test_skew_never_below_one(self):
        config = MoELLMConfig(all_to_all_load_imbalance_aware=True)
        for ep in (2, 4, 8, 16):
            skew = llm_ops_lib._all_to_all_receiver_skew(config, total_tokens=128, expert_parallelism_degree=ep)
            self.assertGreaterEqual(skew, 1.0)

    def test_intermediate_factor_multiple_worst_case_experts(self):
        # f=4.0, W=4, T=256, E=256, K=8, EP=8 (real units): experts_per_group=32,
        #   total_routings=2048, eff_real=32, rem_real=(2048-4*32)/252=1920/252,
        #   hot_group = 4*32 + (32-4)*1920/252 = 7168/21, avg_group=256
        #   -> skew = 7168/21/256 = 4/3.
        config = MoELLMConfig(
            all_to_all_load_imbalance_aware=True,
            expert_load_imbalance_factor=4.0,
            num_worst_case_experts=4,
        )
        skew = llm_ops_lib._all_to_all_receiver_skew(config, total_tokens=256, expert_parallelism_degree=8)
        self.assertAlmostEqual(skew, 4 / 3)


class TestExpertsOnWorstCaseDevice(unittest.TestCase):
    def test_divisible(self):
        self.assertEqual(llm_ops_lib._num_experts_on_worst_case_device(256, 8), 32)
        self.assertEqual(llm_ops_lib._num_experts_on_worst_case_device(256, 1), 256)

    def test_indivisible_rounds_up(self):
        # ceil(10/3) = 4, not floor 3 (which would drop the remainder experts).
        self.assertEqual(llm_ops_lib._num_experts_on_worst_case_device(10, 3), 4)
        self.assertEqual(llm_ops_lib._num_experts_on_worst_case_device(256, 7), 37)

    def test_more_groups_than_experts_gives_at_least_one(self):
        # EP > E must model 1 expert on the busiest device, never 0 (which would
        # silently zero out MoE compute).
        self.assertEqual(llm_ops_lib._num_experts_on_worst_case_device(8, 16), 1)
        self.assertEqual(llm_ops_lib._num_experts_on_worst_case_device(256, 512), 1)


class TestAllToAllReceiverSkewScaling(unittest.TestCase):
    def _make_alltoall(self, receiver_skew):
        # bandwidth-bound regime so the skew factor dominates the latency term.
        config = types.SimpleNamespace(ici_latency_ns=100)
        tensor = Tensor.from_shape("a2a_in", [1000, 1000], dtype="DT_FLOAT16")
        return llm_ops_lib.create_all_to_all_op(
            input=tensor,
            config=config,
            bisection_bw=1.0,
            num_parallelism=2,
            receiver_skew=receiver_skew,
            dtype="DT_FLOAT16",
        )

    def test_receiver_skew_scales_bandwidth_term(self):
        base = self._make_alltoall(1.0)
        skewed = self._make_alltoall(2.0)
        self.assertEqual(skewed.stats.ici_time_ns, 2 * base.stats.ici_time_ns)

    def test_default_skew_is_one(self):
        config = types.SimpleNamespace(ici_latency_ns=100)
        tensor = Tensor.from_shape("a2a_in", [1000, 1000], dtype="DT_FLOAT16")
        explicit = self._make_alltoall(1.0)
        default = llm_ops_lib.create_all_to_all_op(
            input=tensor,
            config=config,
            bisection_bw=1.0,
            num_parallelism=2,
            dtype="DT_FLOAT16",
        )
        self.assertEqual(default.stats.ici_time_ns, explicit.stats.ici_time_ns)


class TestDeepSeekMoEWorstCaseExperts(unittest.TestCase):
    def test_worst_case_expert_structure_no_ep(self):
        # EP = 1: E_per_device = 256, W = 1 (default) -> one hot + remaining experts.
        config = _deepseek_config()
        ops = llm_ops_lib.create_ffn_deepseek_moe(
            batch_size=1,
            seqlen=64,
            config=config,
            count=1,
            tensor_parallelism_axes=[1],
            expert_parallelism_axes=[1],
        )
        descriptions = [op.description for op in ops]
        # The new worst-case model emits "most_loaded" and "remaining" expert blocks...
        self.assertTrue(any("FFN_routed_expert_most_loaded-" in d for d in descriptions))
        self.assertTrue(any("FFN_routed_expert_remaining-" in d for d in descriptions))
        # ...and no longer the old per-expert loop markers.
        self.assertFalse(any("FFN_routed_expert0-" in d for d in descriptions))

    def test_worst_case_expert_counts(self):
        # count scaling: hot experts run count*W_dev; remaining run count*(E_per_device - W_dev).
        config = _deepseek_config(num_worst_case_experts=1)
        count = 3
        ops = llm_ops_lib.create_ffn_deepseek_moe(
            batch_size=1,
            seqlen=64,
            config=config,
            count=count,
            tensor_parallelism_axes=[1],
            expert_parallelism_axes=[1],
        )
        e_per_device = config.num_routed_experts  # EP = 1
        w_dev = 1
        hot_counts = [op.stats.count for op in ops if "FFN_routed_expert_most_loaded-" in op.description]
        rem_counts = [op.stats.count for op in ops if "FFN_routed_expert_remaining-" in op.description]
        self.assertTrue(hot_counts)
        self.assertTrue(rem_counts)
        self.assertTrue(all(c == count * w_dev for c in hot_counts))
        self.assertTrue(all(c == count * (e_per_device - w_dev) for c in rem_counts))

    def test_worst_case_expert_seqlens(self):
        # Default f=-1 (worst case): hot expert gets effective_tokens = T = batch*seqlen tokens;
        # remaining experts split the rest: ceil((T*K - W*eff)/(E-W)).
        from math import ceil
        config = _deepseek_config()  # E=256, K=8, W=1
        batch_size, seqlen = 1, 64
        T = batch_size * seqlen
        ops = llm_ops_lib.create_ffn_deepseek_moe(
            batch_size=batch_size,
            seqlen=seqlen,
            config=config,
            count=1,
            tensor_parallelism_axes=[1],
            expert_parallelism_axes=[1],
        )
        # The first matmul of each FFN block carries input shape [batch, seqlen, d_model].
        hot = next(o for o in ops if "FFN_routed_expert_most_loaded-" in o.description and "FFgate" in o.description)
        rem = next(o for o in ops if "FFN_routed_expert_remaining-" in o.description and "FFgate" in o.description)
        expected_hot = config.get_effective_expert_tokens(T)  # = T = 64
        expected_rem = max(1, ceil((T * 8 - 1 * expected_hot) / (256 - 1)))  # = 2
        self.assertEqual(hot.input_tensors[0].shape[1], expected_hot)
        self.assertEqual(rem.input_tensors[0].shape[1], expected_rem)

    def test_decode_remaining_experts_deinflated(self):
        # Decode (T=1) worst case: real remaining tokens (7) << remaining experts (255),
        # so only ~the routed number of experts activate (each on 1 token), NOT all 255.
        # T=1,K=8,E=256,W=1: eff_real=1, remaining_tokens=7, rem_real=7/255<1,
        #   device_remaining_tokens = 255*(7/255) = 7 -> 7 active remaining experts.
        config = _deepseek_config()
        count = 1
        ops = llm_ops_lib.create_ffn_deepseek_moe(
            batch_size=1,
            seqlen=1,
            config=config,
            count=count,
            is_decode=True,
            tensor_parallelism_axes=[1],
            expert_parallelism_axes=[1],
        )
        rem = [o for o in ops if "FFN_routed_expert_remaining-" in o.description and "FFgate" in o.description]
        self.assertTrue(rem)
        # de-inflated: 7 active experts (was 255), each on a single token.
        self.assertTrue(all(o.stats.count == count * 7 for o in rem))
        self.assertTrue(all(o.input_tensors[0].shape[1] == 1 for o in rem))

    def test_prefill_remaining_experts_unchanged(self):
        # Large-T regime (rem_real >= 1) must be IDENTICAL to the source model:
        # all E_per_device - W_dev remaining experts active. T=64 -> rem_real=1.756.
        from math import ceil
        config = _deepseek_config()
        count = 1
        ops = llm_ops_lib.create_ffn_deepseek_moe(
            batch_size=1,
            seqlen=64,
            config=config,
            count=count,
            tensor_parallelism_axes=[1],
            expert_parallelism_axes=[1],
        )
        rem = [o for o in ops if "FFN_routed_expert_remaining-" in o.description and "FFgate" in o.description]
        self.assertTrue(rem)
        self.assertTrue(all(o.stats.count == count * (256 - 1) for o in rem))  # all 255 active
        self.assertTrue(all(o.input_tensors[0].shape[1] == ceil((64 * 8 - 64) / 255) for o in rem))  # seqlen 2

    def test_worst_case_structure_with_expert_parallelism(self):
        # EP>1 exercises E_per_device = E//EP and the dispatch/combine all-to-all.
        # Needs a merged model+chip config so get_bisection_bw_per_chip_GBps works.
        import json
        from pathlib import Path
        from neusim.configs.models.LLMConfig import DeepSeekConfig

        repo_root = Path(__file__).resolve().parents[4]
        with open(repo_root / "configs/models/deepseekv3-671b.json") as f:
            cfg = json.load(f)
        with open(repo_root / "configs/chips/tpuv6p.json") as f:
            cfg.update(json.load(f))
        with open(repo_root / "configs/systems/system_config.json") as f:
            cfg.update(json.load(f))
        ep = 8
        cfg.update({
            "expert_parallelism_degree": ep,
            "num_expert_parallel_axes": 1,
            "tensor_parallelism_degree": 8,
            "num_tensor_parallel_axes": 1,
            "num_chips": 8,
            "global_batch_size": 8,
        })
        config = DeepSeekConfig(**cfg)
        count = 2
        ops = llm_ops_lib.create_ffn_deepseek_moe(
            batch_size=1,
            seqlen=64,
            config=config,
            count=count,
            tensor_parallelism_axes=[8],
            expert_parallelism_axes=[ep],
        )
        # dispatch + combine all-to-all present
        a2a = [o for o in ops if o.opcode == "AllToAll"]
        self.assertEqual(len(a2a), 2)
        # worst-case counts at EP=8: E_per_device = 256//8 = 32, W_dev = 1
        e_per_device = config.num_routed_experts // ep
        w_dev = 1
        hot_counts = [o.stats.count for o in ops if "FFN_routed_expert_most_loaded-" in o.description]
        rem_counts = [o.stats.count for o in ops if "FFN_routed_expert_remaining-" in o.description]
        self.assertTrue(hot_counts and rem_counts)
        self.assertTrue(all(c == count * w_dev for c in hot_counts))
        self.assertTrue(all(c == count * (e_per_device - w_dev) for c in rem_counts))


if __name__ == "__main__":
    unittest.main()
