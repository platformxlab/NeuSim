from pydantic import TypeAdapter

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

    def __init__(self, **kwargs):
        if "moe_d_ff" not in kwargs:
            # If moe_d_ff is not provided, default to d_ff.
            kwargs["moe_d_ff"] = kwargs["d_ff"]
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
