from math import ceil

from pydantic import TypeAdapter, model_validator

import neusim.configs.models.ModelConfig as ModelConfig


class LLMConfig(ModelConfig.ModelConfig):
    input_seqlen: int = 4096
    output_seqlen: int = 512
    d_model: int = 4096
    num_heads: int = 64
    num_kv_heads: int = -1
    '''
    -1 or the same as num_heads: MHA, will be treated as num_kv_heads == num_heads.
    1: MQA.
    '''
    d_head: int = 64
    d_ff: int = 11008
    num_layers: int = 80
    ffn_type: str = "llama"

    decode_width: int = 1
    use_flash_attention: bool = True

    model_type: str = "llm"

    enable_swap_kv_cache: bool = False
    '''
    Whether to enable swapping the KV cache when it exceeds the HBM capacity.
    When enabled, a new KVSwap operator will be added to the final ops results.
    If the model weights are already OOM, this option will have no effects.
    This is for LLM inference only.
    '''
    max_swap_kv_cache_times_prefill: int = 2
    '''
    The number of times allowed at maximum for swapping during prefill.
    We typically swap once per multiple layers/operators to amortize the overhead
    and overlap with compute.
    This controls how much overlap we can have between swapping and computation in our performance model.

    TODO: in the future, we may want to integrate more sophisticated swapping algorithm
    and modelling approach.
    '''
    max_swap_kv_cache_times_decode: int = 16
    '''
    The number of times allowed at maximum for swapping during decode.
    '''

    def __init__(self, **kwargs):
        if "num_kv_heads" not in kwargs:
            # If num_kv_heads is not provided, default to num_heads.
            kwargs["num_kv_heads"] = kwargs.get("num_heads", 64)
        super().__init__(**kwargs)

    def __hash__(self) -> int:
        return super().__hash__()


class MoELLMConfig(LLMConfig):
    '''
    MoE LLM model configuration.
    '''

    router_type: str = "topk"
    num_shared_experts: int = 1
    '''Number of common experts that are shared across all tokens.'''
    num_routed_experts: int = 256
    '''Total number of routed experts.'''
    num_activated_routed_experts_per_token: int = 8
    '''Total number of activated routed experts per token, excluding shared experts.'''
    num_limited_groups: int = 4
    '''Max number of expert groups to route to.'''
    moe_d_ff: int = -1
    '''Dimension of the MoE feed-forward network. Defaults to d_ff.'''

    expert_load_imbalance_factor: float = -1.0
    '''
    Expert load imbalance factor for MoE expert computation.
    Controls how many tokens the most-loaded expert receives relative to the
    perfectly balanced case.

    Variables:
        T = batch_size * seqlen (total tokens per device dispatched together)
        E = num_routed_experts (e.g., 256 for DeepSeek V3)
        K = num_activated_routed_experts_per_token (top-k, e.g., 8)
        f = expert_load_imbalance_factor

    In the balanced case, each expert receives T * K / E tokens.
    The most-loaded expert receives T * K / E * f tokens.

    - f = 1.0: perfectly balanced (each expert gets T*K/E tokens)
    - f = E/K: worst case (all tokens routed to one expert, equivalent to T tokens)
    - f = -1.0 (default sentinel): auto-resolves to E/K (worst case) for backward compatibility

    The effective number of tokens for the most-loaded expert is:
        effective_tokens = max(1, ceil(T * K / E * f))
    '''

    all_to_all_load_imbalance_aware: bool = False
    '''
    If True, scale the MoE dispatch/combine all-to-all latency by a receiver
    "incast" skew factor derived from expert_load_imbalance_factor (the hottest
    EP group receives/sends disproportionate traffic under skew). Default False
    preserves the balanced (skew = 1) all-to-all model. The skew is computed by
    _all_to_all_receiver_skew() in llm_ops_lib.py and ranges in [1, E/K].
    '''

    num_worst_case_experts: int = 1
    '''
    Number of equally-most-loaded ("worst-case") experts. Each of these W experts
    receives the most-loaded token count (effective_tokens, set by
    expert_load_imbalance_factor); the remaining E-W experts split what's left.

    Models how concentrated the routing is across the K experts each token picks:
      - W = 1 (default): one hottest expert; the other E-1 share the rest. With many
        experts per device this single hot expert is diluted by the average ones.
      - W = K (= num_activated_routed_experts_per_token): every token selects the SAME
        set of K experts -> those K experts each receive all T tokens (the absolute
        worst case), and the other E-K experts receive nothing.

    Range [1, K]. Increasing W concentrates the load onto fewer experts, so a device
    holding the hot set spends proportionally more time on worst-case experts.

    NOTE: W only sets HOW MANY experts are hot; the per-hot-expert load is governed
    solely by expert_load_imbalance_factor (f) via get_effective_expert_tokens. The
    "W = K -> each hot expert receives all T tokens" absolute-worst-case above holds
    only when f is also at its maximum E/K (the default f = -1.0 sentinel resolves to
    exactly that). With a smaller f, W = K still gives only T*K/E*f tokens per hot
    expert, not T -- raise f together with W to model the absolute worst case.
    '''

    expert_parallelism_degree: int = 1
    num_expert_parallel_axes: int = 1
    expert_parallel_degree_dcn: int = 1

    @property
    def expert_tensor_parallelism_degree(self) -> int:
        '''
        Returns the expert tensor parallelism degree.
        This is computed as dp*tp // ep.
        '''
        return self.data_parallelism_degree * self.tensor_parallelism_degree // self.expert_parallelism_degree

    @property
    def num_expert_tensor_parallel_axes(self) -> int:
        '''
        Returns the number of expert tensor parallel axes.
        This is computed as dp + tp - ep.
        '''
        return (self.num_data_parallel_axes + self.num_tensor_parallel_axes) - self.num_expert_parallel_axes

    @property
    def num_experts_per_token(self) -> int:
        '''
        Returns the total number of experts per token, including shared experts and routed experts.
        '''
        return self.num_shared_experts + self.num_activated_routed_experts_per_token

    @property
    def effective_expert_load_imbalance_factor(self) -> float:
        '''
        Returns the effective expert load imbalance factor.
        Resolves the -1.0 sentinel to E/K (worst case).
        '''
        if self.expert_load_imbalance_factor == -1.0:
            return self.num_routed_experts / self.num_activated_routed_experts_per_token
        return self.expert_load_imbalance_factor

    def get_effective_expert_tokens(self, total_tokens: int) -> int:
        '''
        Computes the effective number of tokens for the most-loaded expert.
        total_tokens = batch_size * seqlen (total tokens dispatched together).
        Returns max(1, ceil(total_tokens * K / E * f)).
        '''
        f = self.effective_expert_load_imbalance_factor
        K = self.num_activated_routed_experts_per_token
        E = self.num_routed_experts
        return max(1, ceil(total_tokens * K / E * f))

    @model_validator(mode='after')
    def _validate_expert_load_imbalance_factor(self) -> 'MoELLMConfig':
        # Guard the E/K divisions below (and in get_effective_expert_tokens /
        # effective_expert_load_imbalance_factor) so a degenerate config fails with a
        # clear message instead of a bare ZeroDivisionError.
        if self.num_routed_experts <= 0 or self.num_activated_routed_experts_per_token <= 0:
            raise ValueError(
                "num_routed_experts and num_activated_routed_experts_per_token must both "
                f"be >= 1. Got num_routed_experts={self.num_routed_experts}, "
                f"num_activated_routed_experts_per_token={self.num_activated_routed_experts_per_token}."
            )
        f = self.expert_load_imbalance_factor
        max_f = self.num_routed_experts / self.num_activated_routed_experts_per_token
        if f != -1.0 and not (1.0 <= f <= max_f):
            raise ValueError(
                f"expert_load_imbalance_factor must be -1.0 (auto) or in [1.0, {max_f}] "
                f"(E/K = {self.num_routed_experts}/{self.num_activated_routed_experts_per_token}). "
                f"Got {f}."
            )
        K = self.num_activated_routed_experts_per_token
        if not (1 <= self.num_worst_case_experts <= K):
            raise ValueError(
                f"num_worst_case_experts must be in [1, K={K}] "
                f"(K = num_activated_routed_experts_per_token). "
                f"Got {self.num_worst_case_experts}."
            )
        return self

    def __init__(self, **kwargs):
        if "moe_d_ff" not in kwargs:
            # If moe_d_ff is not provided, default to d_ff.
            kwargs["moe_d_ff"] = kwargs.get("d_ff", self.model_fields["d_ff"].default)
        super().__init__(**kwargs)

    def __hash__(self) -> int:
        '''
        Just hash some critical fields of the config.
        This is probably not the most efficient way, so use it with caution.
        '''
        return hash(
            (
                self.name,  # chip name
                self.model_name,  # model name
                self.num_chips,
                self.data_parallelism_degree,
                self.tensor_parallelism_degree,
                self.pipeline_parallelism_degree,
                self.expert_parallelism_degree,
                self.num_data_parallel_axes,
                self.num_tensor_parallel_axes,
                self.num_pipeline_parallel_axes,
                self.num_expert_parallel_axes,
                self.data_parallel_degree_dcn,
                self.tensor_parallel_degree_dcn,
                self.pipeline_parallel_degree_dcn,
                self.expert_parallel_degree_dcn,
                self.microbatch_size_dcn,
                self.microbatch_size_ici,
                self.global_batch_size,
                # Load-imbalance fields change the generated op graph, so configs that
                # differ only in these must not collide on the config hash.
                self.expert_load_imbalance_factor,
                self.all_to_all_load_imbalance_aware,
                self.num_worst_case_experts,
            )
        )


class DeepSeekConfig(MoELLMConfig):
    '''
    DeepSeek model configuration.
    '''

    num_dense_layers: int = 1
    '''Number of dense layers in the model. Will be the first layer(s) in the model.'''

    # MLA configs
    kv_lora_rank: int
    q_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int

    @property
    def qk_head_dim(self) -> int:
        '''
        Returns the total dimension of the query-key head.
        '''
        return self.qk_rope_head_dim + self.qk_nope_head_dim

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __hash__(self) -> int:
        return super().__hash__()
