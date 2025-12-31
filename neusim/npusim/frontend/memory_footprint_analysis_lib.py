from copy import deepcopy
import json
from math import ceil
from typing import Any

from neusim.configs.models.ModelConfig import ModelConfig
import neusim.npusim.frontend.query_results_helper_lib as results_lib

from neusim.configs.models.DiTConfig import DiTConfig
from neusim.configs.models.DLRMConfig import DLRMConfig
from neusim.configs.models.GLIGENConfig import GLIGENConfig
from neusim.configs.models.LLMConfig import DeepSeekConfig, LLMConfig


BYTES_FP32 = 4
BYTES_FP16 = 2


def get_llm_training_mem_requirement(
    config: str | dict[str, Any] | LLMConfig,
    weight_bytes_per_element: int = BYTES_FP16,
    activation_bytes_per_element: int = BYTES_FP16,
    optimizer_bytes_per_element: int = BYTES_FP32,
) -> int:
    '''
    Calculate the memory requirement for training a LLM model with DP/TP/PP.
    @config: path to the config file or the config dict.
    @weight_bytes_per_element: bytes per element for weights. Defaults to FP16.
    @activation_bytes_per_element: bytes per element for activations. Defaults to FP16.
    @optimizer_bytes_per_element: bytes per element for optimizer states. Defaults to FP32.

    @return: memory requirement in bytes per chip.
    '''
    if isinstance(config, str):
        with open(config, "r") as config_file:
            config = dict(json.load(config_file))
    if isinstance(config, dict):
        config = LLMConfig.model_validate(config)

    dp: int = config.data_parallelism_degree
    tp: int = config.tensor_parallelism_degree
    pp: int = config.pipeline_parallelism_degree

    global_batch_size = config.global_batch_size
    batch_size = ceil(global_batch_size / dp)

    num_heads: int = config.num_heads
    d_head: int = config.d_head
    d_model: int = config.d_model
    d_ff: int = config.d_ff
    num_layers: int = config.num_layers

    num_heads = ceil(num_heads / tp)
    d_ff = ceil(d_ff / tp)
    num_layers_per_chip = ceil(num_layers / pp)
    input_seqlen: int = config.input_seqlen

    # attn
    w_attn = 0
    a_attn = 0

    # input
    a_attn += batch_size * input_seqlen * ceil(d_model / tp)
    # layer norm
    w_attn += 2 * batch_size * input_seqlen * d_model
    a_attn += 2 * batch_size * input_seqlen * d_model
    # Wq, Wk, Wv -> einsum
    w_attn += 3 * d_model * num_heads * d_head
    a_attn += max(3 * d_model * num_heads * d_head, 3 * batch_size * input_seqlen * num_heads * d_head)
    # Q*K
    w_attn += 0
    a_attn += batch_size * input_seqlen * input_seqlen * num_heads
    # softmax
    w_attn += 0
    a_attn += batch_size * input_seqlen * input_seqlen * num_heads
    # (Q*K)*V
    w_attn += 0
    a_attn += batch_size * input_seqlen * num_heads * d_head
    # output einsum
    w_attn += num_heads * d_head * d_model
    a_attn += max(num_heads * d_head * d_model, batch_size * input_seqlen * d_model)
    # output layernorm
    w_attn += 2 * batch_size * input_seqlen * d_model
    a_attn += 2 * batch_size * input_seqlen * d_model

    # ffn
    w_ff = 0
    a_ff = 0
    # up + gate
    w_ff += 2 * (d_model * d_ff)
    a_ff += max(2 * (d_model * d_ff), 2 * batch_size * input_seqlen * d_ff)
    # elementwise mul
    w_ff += 0
    a_ff += batch_size * input_seqlen * d_ff
    # down
    w_ff += d_ff * d_model
    a_ff += max(d_ff * d_model, batch_size * input_seqlen * d_model)

    w_ff *= num_layers_per_chip
    a_ff *= num_layers_per_chip

    w = w_attn + w_ff
    a = a_attn + a_ff  # activations and grads

    # https://arxiv.org/pdf/2108.05818
    ## master param + momentum + variance in fp32
    opt = 3 * w

    mem = w * weight_bytes_per_element + a * activation_bytes_per_element + opt * optimizer_bytes_per_element
    return mem


def get_llm_inference_mem_requirement(
    config: str | dict[str, Any] | LLMConfig,
    prefill_or_decode: str = "decode",
    weight_bytes_per_element: int = BYTES_FP16,
    activation_bytes_per_element: int = BYTES_FP16,
) -> int:
    '''
    Calculate the memory requirement for serving a LLM model with DP/TP/PP. \\
    @config: path to the config file or the config dict. \\
    @prefill_or_decode: "prefill" or "decode". This is used to determine the KV cache size.
        If "prefill", only the input sequence length is considered.
        If "decode", both input and output sequence lengths are considered. \\
    @weight_bytes_per_element: bytes per element for weights. Defaults to FP16. \\
    @activation_bytes_per_element: bytes per element for activations. Defaults to FP16. \\

    @return: memory requirement in bytes per chip.
    '''
    if isinstance(config, str):
        with open(config, "r") as config_file:
            config = dict(json.load(config_file))
    if isinstance(config, dict):
        config = LLMConfig.model_validate(config)

    # dp: int = config.data_parallelism_degree * config.data_parallel_degree_dcn
    tp: int = config.tensor_parallelism_degree * config.tensor_parallel_degree_dcn
    pp: int = config.pipeline_parallelism_degree * config.pipeline_parallel_degree_dcn

    # global_batch_size = config.global_batch_size
    # batch_size = ceil(global_batch_size / dp)
    # per dp-ici replica batch size
    batch_size = ceil(config.microbatch_size_ici / config.data_parallelism_degree)

    num_heads: int = config.num_heads
    num_kv_heads: int = config.num_kv_heads
    if num_kv_heads == -1:
        num_kv_heads = num_heads
    d_head: int = config.d_head
    d_model: int = config.d_model
    d_ff: int = config.d_ff
    num_layers: int = config.num_layers

    num_heads = ceil(num_heads / tp)
    num_kv_heads = ceil(num_kv_heads / tp)
    d_ff = ceil(d_ff / tp)
    num_layers_per_chip = ceil(num_layers / pp)
    input_seqlen: int = config.input_seqlen
    output_seqlen: int = config.output_seqlen
    seqlen = input_seqlen + output_seqlen if prefill_or_decode == "decode" else input_seqlen

    ### Attention Layer ###

    # For GQA, if TP <= # of KV heads, then the KV cache is shared.
    # Otherwise, the KV cache needs to be replicated across TP chips.
    if tp <= num_kv_heads:
        w_attn_kv = d_model * num_kv_heads * d_head * 2
    else:
        w_attn_kv = d_model * num_heads * d_head * 2
    w_attn_q = d_model * num_heads * d_head
    w_attn_qkv = w_attn_kv + w_attn_q
    w_attn_output = num_heads * d_head * d_model
    w_attn = w_attn_qkv + w_attn_output  # attention weights

    a_attn_q = batch_size * seqlen * num_heads * d_head
    if tp <= num_kv_heads:
        a_attn_kv = batch_size * seqlen * num_kv_heads * d_head * 2
    else:
        a_attn_kv = batch_size * seqlen * num_heads * d_head * 2
    a_attn_qkv = a_attn_kv + a_attn_q  # KV cache size + Q activation size
    a_attn = a_attn_qkv  # KV cache needs separate storage, Q activation can be used in place

    ### FFN Layer ###

    if config.ffn_type == "default":
        w_ff = 2 * d_model * d_ff
        a_ff = batch_size * seqlen * max(d_ff, d_model)
    elif config.ffn_type == "llama":
        w_ff = 3 * d_model * d_ff
        a_ff = 2 * batch_size * seqlen * max(d_ff, d_model)
    else:
        raise ValueError(f"Unknown ffn_type: {config.ffn_type}")

    # a_ff = 2 * batch_size * seqlen * max(d_ff, d_model)

    total_weights = (w_attn + w_ff) * num_layers_per_chip
    total_act = (a_attn + a_ff) * num_layers_per_chip

    mem = total_weights * weight_bytes_per_element + total_act * activation_bytes_per_element
    return mem


def get_llm_inference_weight_mem_requirement(
    _config: str | dict[str, Any] | LLMConfig,
    weight_bytes_per_element: int = BYTES_FP16,
) -> int:
    '''
    Calculate memory capacity requirements for LLM inference for model weights.
    '''
    if isinstance(_config, str):
        with open(_config, "r") as config_file:
            _config = dict(json.load(config_file))
    if isinstance(_config, dict):
        config = LLMConfig.model_validate(_config)
    else:
        config = deepcopy(_config)

    config.input_seqlen = 0
    config.output_seqlen = 0

    if isinstance(config, DeepSeekConfig):
        mem_bytes = get_deepseek_inference_mem_requirement(
            config, "prefill"
        )
    else:
        mem_bytes = get_llm_inference_mem_requirement(
            config, "prefill", weight_bytes_per_element, 2
        )
    return mem_bytes


def get_llm_inference_kv_cache_mem_requirement(
    config: str | dict[str, Any] | LLMConfig,
    prefill_or_decode: str = "decode",
) -> int:
    '''
    Calculate the KV cache size in bytes.
    '''
    if isinstance(config, str):
        with open(config, "r") as config_file:
            config = dict(json.load(config_file))
    if isinstance(config, dict):
        config = LLMConfig.model_validate(config)

    seqlen = config.input_seqlen if prefill_or_decode == "prefill" else (config.input_seqlen + config.output_seqlen)
    tp = (
        config.tensor_parallelism_degree
        * config.tensor_parallel_degree_dcn
    )
    bs = ceil(config.microbatch_size_ici / config.data_parallelism_degree) # This is microbatch size per PP stage in each DP_DCN replica

    # total capacity
    if isinstance(config, DeepSeekConfig):
        kv_cache_elem_bytes = 1  # FP8
        pe_cache_elem_bytes = 4  # FP32

        num_layers = config.num_layers  # This is num layers per pipeline stage
        kv_cache_bytes_per_token = (
            config.kv_lora_rank * kv_cache_elem_bytes
            + config.qk_rope_head_dim * pe_cache_elem_bytes
        ) * num_layers
        kv_cache_bytes = ceil(
            kv_cache_bytes_per_token * bs * seqlen / tp
        )  # sequence parallelism over TP axes
        # print(f"KV+PE cache: {kv_cache_bytes} bytes")
    else:
        num_kv_heads = config.num_kv_heads
        d_head = config.d_head
        num_heads = ceil(config.num_heads / tp)
        if tp <= num_kv_heads:
            kv_cache_bytes = bs * seqlen * num_kv_heads * d_head * 2
        else:
            kv_cache_bytes = bs * seqlen * num_heads * d_head * 2
    return kv_cache_bytes


def get_deepseek_inference_mem_requirement(config: DeepSeekConfig, prefill_or_decode: str = "decode") -> int:
    """
    Compute the memory footprint of the LLM ops in bytes.
    For DeepSeek models, assume the FFN weights and activations are FP8.
    Other weights are BF16.
    Refer to the implementation in https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
    """
    assert isinstance(
        config, DeepSeekConfig
    ), f"Invalid config type: {type(config)}. Expected DeepSeekConfig."

    if prefill_or_decode == "prefill":
        # only count input sequence length for prefill
        seqlen = config.input_seqlen
    elif prefill_or_decode == "decode":
        # count both input+output sequence length for decode
        seqlen = config.input_seqlen + config.output_seqlen
    else:
        raise ValueError(
            f"Invalid prefill_or_decode value: {prefill_or_decode}. Expected 'prefill' or 'decode'."
        )

    tp = (
        config.tensor_parallelism_degree
        * config.tensor_parallel_degree_dcn
    )
    ep = (
        config.expert_parallelism_degree
        * config.expert_parallel_degree_dcn
    )
    etp = config.expert_tensor_parallelism_degree
    bs = ceil(config.microbatch_size_ici / config.data_parallelism_degree)  # This is microbatch size per PP stage in each DP_DCN replica
    num_layers = config.num_layers  # This is num layers per pipeline stage

    attn_weight_elem_bytes = 2  # BF16
    ffn_weight_elem_bytes = 1  # FP8
    ffn_act_elem_bytes = 2  # BF16 intermediate and output
    kv_cache_elem_bytes = 1  # FP8
    pe_cache_elem_bytes = 4  # FP32
    total_mem_footprint_bytes = 0

    ### MLA attention layer weights
    wq_a = config.d_model * config.q_lora_rank
    wq_b = (
        config.q_lora_rank
        * config.num_heads
        * config.qk_head_dim
        / tp
    )
    wkv_a = (
        config.d_model
        * config.kv_lora_rank
        * config.qk_rope_head_dim
    )
    wkv_b = (
        config.kv_lora_rank
        * config.num_heads
        * (config.qk_rope_head_dim + config.v_head_dim)
        / tp
    )
    wo = config.num_heads * config.v_head_dim * config.d_model / tp
    MLA_num_params = (wq_a + wq_b + wkv_a + wkv_b + wo) * num_layers
    MLA_weight_bytes = MLA_num_params * attn_weight_elem_bytes
    total_mem_footprint_bytes += MLA_weight_bytes
    # print(f"MLA attention layer weights: {MLA_weight_bytes} bytes")

    ### MLA attention layer activations (excluding KV cache)
    # q_nope (FP8) and q_pe (FP32 intermediate + FP8 result)
    q_act_bytes = (
        bs * seqlen * config.num_heads * (config.qk_rope_head_dim * (4 + 1) + config.qk_nope_head_dim * 1)
        / tp
    )
    total_mem_footprint_bytes += q_act_bytes
    # print(f"MLA attention layer activations: {q_act_bytes} bytes")

    ### MoE FFN weights
    # TODO: we currently just assume all layers are MoE layers.
    # Actually for DeepSeek models, the first few layers are dense layers.
    # But they have larger d_ff than MoE experts, and should not impact
    # our estimation of the memory footprint too much.
    ffn_num_params = (
        config.d_model
        * config.moe_d_ff
        * 3
        / etp  # each expert has 3 matrices
        * (config.num_shared_experts + config.num_routed_experts)
        / ep  # total # of experts
        * num_layers
    )
    moe_gate_params = (
        config.num_routed_experts * config.d_model * num_layers
    )
    ffn_weight_bytes = (ffn_num_params + moe_gate_params) * ffn_weight_elem_bytes
    total_mem_footprint_bytes += ffn_weight_bytes
    # print(f"MoE FFN layer weights: {ffn_weight_bytes} bytes")

    ### MoE FFN activations
    # Only needs to account for the intermediate matrices in activated experts.
    # The input and output matrices are already considered in MLA and KV cache.
    # Specifically, need to allocate spaces for self.w1(x) and self.w3(x).
    # Then w1(x) can be point-wise multiplied into w3(x) in place, and the result
    # can be multipled with w2.
    # We need to reserve space for the worst case token distribution:
    #   All tokens are routed to the same expert group.
    ffn_expert_act_bytes = (
        bs * seqlen * config.moe_d_ff * 2 / etp
    ) * ffn_act_elem_bytes
    ffn_act_bytes = ffn_expert_act_bytes * num_layers
    total_mem_footprint_bytes += ffn_act_bytes
    # print(f"MoE FFN layer activations: {ffn_act_bytes} bytes")

    ### KV+PE cache
    kv_cache_bytes_per_token = (
        config.kv_lora_rank * kv_cache_elem_bytes
        + config.qk_rope_head_dim * pe_cache_elem_bytes
    ) * num_layers
    kv_cache_bytes = (
        kv_cache_bytes_per_token * bs * seqlen / tp
    )  # sequence parallelism over TP axes
    total_mem_footprint_bytes += kv_cache_bytes
    # print(f"KV+PE cache: {kv_cache_bytes} bytes")

    return round(total_mem_footprint_bytes)


def get_dlrm_inference_mem_requirement(
    config: str | dict[str, Any] | DLRMConfig,
    weight_bytes_per_element: int = BYTES_FP32,
    activation_bytes_per_element: int = BYTES_FP32,
) -> int:
    if isinstance(config, str):
        with open(config, "r") as config_file:
            config = dict(json.load(config_file))
    if isinstance(config, dict):
        config = DLRMConfig.model_validate(config)
    memory_capacity_per_chip = config.hbm_size_GB
    # TODO: for now, assume at least 8 chips for DLRM
    return 8 * memory_capacity_per_chip * 1024**3 - 1024  # offset a little bit for safety


def get_dit_inference_mem_requirement(
    config: str | dict[str, Any] | DiTConfig,
    weight_bytes_per_element: int = BYTES_FP16,
    activation_bytes_per_element: int = BYTES_FP16,
) -> int:
    if isinstance(config, str):
        with open(config, "r") as config_file:
            config = dict(json.load(config_file))
    if isinstance(config, dict):
        config = DiTConfig.model_validate(config)

    def get_llm_config_from_dit_config(config: DiTConfig) -> LLMConfig:
        '''
        Convert DiT config to LLM config.
        '''
        llm_config = LLMConfig(**config.__dict__)
        llm_config.model_type = "llm"
        llm_config.input_seqlen = ceil(
            (config.image_width * config.image_width)
            / (config.patch_size * config.patch_size)
        )
        llm_config.output_seqlen = 1
        return llm_config

    llm_config = get_llm_config_from_dit_config(config)
    return get_llm_inference_mem_requirement(
        llm_config,
        prefill_or_decode="prefill",
        weight_bytes_per_element=weight_bytes_per_element,
        activation_bytes_per_element=activation_bytes_per_element,
    )


def get_gligen_inference_mem_requirement(
    config: str | dict[str, Any] | GLIGENConfig,
    weight_bytes_per_element: int = BYTES_FP32,
    activation_bytes_per_element: int = BYTES_FP32,
) -> int:
    if isinstance(config, str):
        with open(config, "r") as config_file:
            config = dict(json.load(config_file))
    if isinstance(config, dict):
        config = GLIGENConfig.model_validate(config)
    memory_capacity_per_chip = config.hbm_size_GB
    # TODO: for now, assume at least 1 chip for GLIGEN
    return memory_capacity_per_chip * 1024**3 - 1024  # offset a little bit for safety


def get_mem_requirement(
    config: str | dict[str, Any] | ModelConfig,
    model: str,
    workload: str,
    weight_bytes_per_element: int = 4,
    activation_bytes_per_element: int = 4,
) -> int:
    if results_lib.is_model_llm(model):
        assert isinstance(config, (str, dict, LLMConfig)), \
            f"Expected config to be a path, dict, or LLMConfig instance, got {type(config)}"
        if workload == "training":
            return get_llm_training_mem_requirement(
                config,
                weight_bytes_per_element=weight_bytes_per_element,
                activation_bytes_per_element=activation_bytes_per_element,
            )
        elif workload == "inference":
            return get_llm_inference_mem_requirement(
                config,
                weight_bytes_per_element=weight_bytes_per_element,
                activation_bytes_per_element=activation_bytes_per_element,
            )
        else:
            raise ValueError(f"Unknown workload: {workload}")
    elif results_lib.is_model_dlrm(model):
        assert isinstance(config, (str, dict, DLRMConfig)), \
            f"Expected config to be a path, dict, or DLRMConfig instance, got {type(config)}"
        return get_dlrm_inference_mem_requirement(
            config,
            weight_bytes_per_element=weight_bytes_per_element,
            activation_bytes_per_element=activation_bytes_per_element,
        )
    elif results_lib.is_model_sd(model):
        if "dit" in model.lower():
            assert isinstance(config, (str, dict, DiTConfig)), \
                f"Expected config to be a path, dict, or DiTConfig instance, got {type(config)}"
            return get_dit_inference_mem_requirement(
                config,
                weight_bytes_per_element=weight_bytes_per_element,
                activation_bytes_per_element=activation_bytes_per_element,
            )
        elif "gligen" in model.lower():
            assert isinstance(config, (str, dict, GLIGENConfig)), \
                f"Expected config to be a path, dict, or GLIGENConfig instance, got {type(config)}"
            return get_gligen_inference_mem_requirement(
                config,
                weight_bytes_per_element=weight_bytes_per_element,
                activation_bytes_per_element=activation_bytes_per_element,
            )
        else:
            raise ValueError(f"Unknown model: {model}")
    else:
        raise ValueError(f"Unknown model: {model}")
