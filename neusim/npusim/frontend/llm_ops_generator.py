from abc import abstractmethod
from copy import deepcopy
import csv
from functools import lru_cache
import os
from typing import Any
from math import ceil
from absl import flags, logging

import neusim.npusim.frontend.memory_footprint_analysis_lib as mem_footprint_lib
import neusim.npusim.frontend.Operator as Operator
import neusim.npusim.frontend.power_analysis_lib as power_lib
from neusim.configs.models.LLMConfig import (
    DeepSeekConfig,
    LLMConfig,
)

from neusim.npusim.frontend import llm_ops_lib as ops_lib
from neusim.npusim.frontend import op_analysis_lib as analysis_lib
from neusim.npusim.frontend.util import split_parallelism_degree


@lru_cache(maxsize=None)
def get_all_factors(n: int) -> list[int]:
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors


class LLMOpsGeneratorBase:
    def __init__(self, config: dict[str, Any] | LLMConfig):
        if isinstance(config, dict):
            self.config: LLMConfig = LLMConfig.model_validate(config)
        else:
            self.config = config
        # assert self.config.use_flash_attention == True
        assert self.config.model_type == "llm", f"Invalid config: {self.config}"
        self.global_batch_size: int = self.config.global_batch_size
        """Global batch size."""
        self.microbatch_size_dcn: int = self.config.microbatch_size_dcn
        """Microbatch size for PP_DCN."""
        self.microbatch_size_ici: int = self.config.microbatch_size_ici
        """Microbatch size for PP_ICI."""
        self.input_seqlen: int = self.config.input_seqlen
        """Input (prefill) sequence length."""
        self.output_seqlen: int = self.config.output_seqlen
        """Output (decode) sequence length."""

        self.decode_width: int = self.config.decode_width
        """
        Number of tokens decoded per iteration. Default to 1.
        Useful for speculative decoding, beam search, etc.
        """
        self.d_model: int = self.config.d_model
        """d_model dimension."""
        self.num_heads: int = self.config.num_heads
        """Number of attention heads."""
        if self.config.num_kv_heads == -1:
            # Default to num_heads if not specified (MHA).
            self.num_kv_heads = self.num_heads
        else:
            # GQA if num_kv_heads != 1; MQA if num_kv_heads == 1.
            self.num_kv_heads: int = self.config.num_kv_heads
            """
            Number of key and value heads.
            Used for GQA and MQA.
            """
        self.d_head: int = self.config.d_head
        """Size of each attention head."""
        self.d_ff: int = self.config.d_ff
        """FFN intermediate size."""
        self.ffn_type: str = self.config.ffn_type
        """
        FFN layer architecture.
        "default": FFN_out = FFo(act(FFi(input))).
        "llama": LLaMA 2 has 3 matrices in FFN: gate, up, and down.
        It is computed as FFN_out = down(act(gate(input)) * up(input)).
        "*" is element-wise multiplication.
        """
        # assert self.ffn_type in ["default", "llama"]
        self.use_flash_attention: bool = self.config.use_flash_attention
        self.num_layers: int = self.config.num_layers
        """Number of transformer blocks (layers)."""
        self.output_file_path: str = self.config.output_file_path
        """Output file path."""

        ### NPU chip configs
        self.num_sa: int = self.config.num_sa
        self.num_vu: int = self.config.num_vu
        self.hbm_bw_GBps: float = self.config.hbm_bw_GBps
        self.vmem_size_MB: int = self.config.vmem_size_MB
        self.freq_GHz: float = self.config.freq_GHz
        self.ici_bw_GBps: float = self.config.ici_bw_GBps

        ### Multi-chip parallelism configuration.
        self.num_chips: int = self.config.num_chips
        """Number of TPU/NPU chips."""
        self.data_parallelism: int = self.config.data_parallelism_degree
        """Data parallelism degree over ICI."""
        self.tensor_parallelism: int = self.config.tensor_parallelism_degree
        """Tensor parallelism degree over ICI."""
        self.pipeline_parallelism: int = self.config.pipeline_parallelism_degree
        """Pipeline parallelism degree over ICI."""
        assert self.num_chips >= (
            self.data_parallelism * self.tensor_parallelism * self.pipeline_parallelism
        ), "Parallelism configuration is incompatible with the number of NPU chips!"

        self.num_data_parallel_axes: int = self.config.num_data_parallel_axes
        """Number of physical axes for data parallelism."""
        self.num_tensor_parallel_axes: int = self.config.num_tensor_parallel_axes
        """Number of physical axes for tensor parallelism."""
        self.num_pipeline_parallel_axes: int = self.config.num_pipeline_parallel_axes
        """Number of physical axes for pipeline parallelism."""
        self.dp_ici_bw_GBps: float = self.ici_bw_GBps
        """ICI bisection BW for a single data parallelism axis."""
        self.tp_ici_bw_GBps: float = self.ici_bw_GBps
        """ICI bisection BW for a single tensor parallelism axis."""
        self.pp_ici_bw_GBps: float = self.ici_bw_GBps
        """ICI bisection BW for a single pipeline parallelism axis."""

        # init default parallelism degree to axes mappings
        self.data_parallelism_axes = [1] * self.num_data_parallel_axes
        """Dim size of each ICI data parallel axis."""
        self.tensor_parallelism_axes = [1] * self.num_tensor_parallel_axes
        """Dim size of each ICI tensor parallel axis."""
        self.pipeline_parallelism_axes = [1] * self.num_pipeline_parallel_axes
        """Dim size of each ICI pipeline parallel axis. This is always 1 for now."""

        # automatically split DP and TP degrees based on num axes
        if self.num_data_parallel_axes > 0:
            self.data_parallelism_axes = split_parallelism_degree(
                self.data_parallelism, self.num_data_parallel_axes
            )
        if self.num_tensor_parallel_axes > 0:
            self.tensor_parallelism_axes = split_parallelism_degree(
                self.tensor_parallelism, self.num_tensor_parallel_axes
            )

        self.data_parallelism_dcn: int = self.config.data_parallel_degree_dcn
        """Data parallelism degree over DCN."""
        self.tensor_parallelism_dcn: int = self.config.tensor_parallel_degree_dcn
        """Tensor parallelism degree over DCN."""
        assert (
            self.tensor_parallelism_dcn == 1
        ), "Tensor parallelism over DCN is not supported yet."
        self.pipeline_parallelism_dcn: int = self.config.pipeline_parallel_degree_dcn
        """Pipeline parallelism degree over DCN."""
        self.dcn_bw_GBps: float = self.config.dcn_bw_GBps
        """DCN BW."""

        self.batch_size: int = ceil(self.microbatch_size_ici / self.data_parallelism)
        """Local batch size per DP_ICI replica."""
        self.num_layers: int = ceil(
            self.num_layers / self.pipeline_parallelism / self.pipeline_parallelism_dcn
        )

        ### Energy and Carbon configs
        self.TDP_W = self.config.TDP_W
        self.idle_power_W = self.config.idle_power_W
        self.min_power_W = self.config.min_power_W
        self.avg_power_W = self.config.avg_power_W
        self.max_power_W = self.config.max_power_W

        self.embodied_carbon = self.config.embodied_carbon_kgCO2

    # def set_new_parallelism_cfg(
    #     self,
    #     config: dict[str, Any],
    #     t_parallel: int,
    #     d_parallel: int,
    #     p_parallel: int,
    # ):
    #     self.tensor_parallelism = t_parallel
    #     self.data_parallelism = d_parallel
    #     self.pipeline_parallelism = p_parallel
    #     self.batch_size = ceil(config["microbatch_size_ici"] / d_parallel)
    #     self.num_layers = ceil(config["num_layers"] / p_parallel)
    #     raise NotImplementedError("This function is not implemented yet.")

    @abstractmethod
    def generate(
        self, fusion_id_start: int = 2, dump_to_file: bool = True, **kwargs
    ) -> list[Operator.Operator] | tuple[list[Operator.Operator], ...]:
        """
        Generate the LLM ops and return them as a list of Operator objects.
        """
        raise NotImplementedError("This function is not implemented yet.")

    @abstractmethod
    def compute_memory_footprint_bytes(self) -> int:
        """
        Compute the memory footprint of the LLM ops in bytes.
        """
        raise NotImplementedError("This function is not implemented yet.")


class LLMOpsGenerator(LLMOpsGeneratorBase):
    """LLM ops generator for inference."""

    def __init__(self, config: dict[str, Any] | LLMConfig):
        super().__init__(config)

    def generate_prefill_ops(self, fusion_id_start: int = 2) -> list[Operator.Operator]:
        ops: list[Operator.Operator] = []
        fusion_id = fusion_id_start

        d_model_parallel = ceil(self.d_model / self.tensor_parallelism)

        # KVSwap: add ops to swap KV cache
        if self.config.enable_swap_kv_cache:
            # compute memory requirements
            weight_size_bytes = mem_footprint_lib.get_llm_inference_weight_mem_requirement(
                self.config
            )
            kv_cache_size_bytes = mem_footprint_lib.get_llm_inference_kv_cache_mem_requirement(
                self.config, "prefill"
            )
            total_mem_footprint_bytes = self.compute_memory_footprint_bytes("prefill")

            chip_mem_capacity_bytes = self.config.hbm_size_GB * 1024 * 1024 * 1024
            if total_mem_footprint_bytes > chip_mem_capacity_bytes:
                # swap in/out entire kvcache and weights
                swap_size_bytes = ceil(
                    (total_mem_footprint_bytes - chip_mem_capacity_bytes)
                    / min(self.config.max_swap_kv_cache_times_prefill, self.num_layers)  # swap per multiple layers to overlap with compute
                )
                ops.append(
                    ops_lib.create_kvswap_op(
                        input_shape=[swap_size_bytes],
                        name=f"PCIeMemSwapIn",
                        description="PCIeMemSwapIn",
                        config=self.config,
                        count=1,
                        fusion_id_start=fusion_id,
                        transfer_type="Input",
                    )
                )
                fusion_id += 1
                ops.append(
                    ops_lib.create_kvswap_op(
                        input_shape=[swap_size_bytes],
                        name=f"PCIeMemSwapOut",
                        description="PCIeMemSwapOut",
                        config=self.config,
                        count=1,
                        fusion_id_start=fusion_id,
                        transfer_type="Output",
                    )
                )
                fusion_id += 1

        # PP DCN input
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"receiveInput_DCNTTransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="PrefillReceiveInputFromPipelineDCN",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP ICI input
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"receiveInput_ICITransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="PrefillReceiveInputFromPipelineICI",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # ops.append(
        #     ops_lib.create_input_op(
        #         input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
        #         name=f"receive_from_dcn(X) HBMTransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
        #         description="Receive_input",
        #         count=self.num_layers,
        #         fusion_id=fusion_id,
        #     )
        # )

        # fusion_id += 1
        ops.append(
            ops_lib.create_unary_op(
                input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                op_name="LayerNorm",
                name="X_norm = LayerNorm(X)",
                description="Fwd-Attention-encoder-Input_layernorm",
                count=self.num_layers,
                fusion_id=fusion_id,
            )
        )
        fusion_id += 1
        ops += ops_lib.create_multi_head_attention(
            batch_size=self.batch_size,
            input_seqlen=self.input_seqlen,
            output_seqlen=self.output_seqlen,
            decode_width=self.decode_width,
            num_heads=self.num_heads,
            d_model=self.d_model,
            d_head=self.d_head,
            config=self.config,
            num_layers=self.num_layers,
            fusion_id_start=fusion_id,
            is_decode=False,
            use_flash_attention=self.use_flash_attention,
            tensor_parallelism_axes=self.tensor_parallelism_axes,
            ici_bw_GBps=self.tp_ici_bw_GBps,
        )
        fusion_id = ops[-1].fusion_id + 1
        ops += ops_lib.create_ffn(
            batch_size=self.batch_size,
            input_seqlen=self.input_seqlen,
            output_seqlen=self.output_seqlen,
            decode_width=self.decode_width,
            d_model=self.d_model,
            d_ff=self.d_ff,
            config=self.config,
            num_layers=self.num_layers,
            ffn_type=self.ffn_type,
            fusion_id_start=fusion_id,
            tensor_parallelism_axes=self.tensor_parallelism_axes,
            is_decode=False,
            ici_bw_GBps=self.tp_ici_bw_GBps,
        )

        # PP ICI output
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"SendOutput_ICITransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="PrefillSendOutputToPipelineICI",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP DCN output
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"SendOutput_DCNTTransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="PrefillSendOutputToPipelineDCN",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        return ops

    def generate_decode_ops(self, fusion_id_start: int = 2) -> list[Operator.Operator]:
        ops: list[Operator.Operator] = []
        fusion_id = fusion_id_start
        count = self.num_layers * self.output_seqlen
        d_model_parallel = ceil(self.d_model / self.tensor_parallelism)

        # KVSwap: add ops to swap KV cache
        if self.config.enable_swap_kv_cache:
            # compute memory requirements
            weight_size_bytes = mem_footprint_lib.get_llm_inference_weight_mem_requirement(
                self.config
            )
            kv_cache_size_bytes = mem_footprint_lib.get_llm_inference_kv_cache_mem_requirement(
                self.config, "prefill"
            )
            total_mem_footprint_bytes = self.compute_memory_footprint_bytes("decode")

            chip_mem_capacity_bytes = self.config.hbm_size_GB * 1024 * 1024 * 1024
            if total_mem_footprint_bytes > chip_mem_capacity_bytes:
                # swap in/out entire kvcache and weights
                swap_size_bytes = ceil(
                    (total_mem_footprint_bytes - chip_mem_capacity_bytes)
                    / min(self.config.max_swap_kv_cache_times_decode, self.num_layers)  # swap per multiple layers to overlap with compute
                )
                ops.append(
                    ops_lib.create_kvswap_op(
                        input_shape=[swap_size_bytes],
                        name=f"PCIeMemSwapIn",
                        description="PCIeMemSwapIn",
                        config=self.config,
                        count=self.output_seqlen,
                        fusion_id_start=fusion_id,
                        transfer_type="Input",
                    )
                )
                fusion_id += 1
                ops.append(
                    ops_lib.create_kvswap_op(
                        input_shape=[swap_size_bytes],
                        name=f"PCIeMemSwapOut",
                        description="PCIeMemSwapOut",
                        config=self.config,
                        count=self.output_seqlen,
                        fusion_id_start=fusion_id,
                        transfer_type="Output",
                    )
                )
                fusion_id += 1

        # PP DCN input
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.decode_width, d_model_parallel],
                    name=f"receiveInput_DCNTTransfer[{self.batch_size * self.decode_width * d_model_parallel}]",
                    description="DecodeReceiveInputFromPipelineDCN",
                    config=self.config,
                    count=self.output_seqlen,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP ICI input
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.decode_width, d_model_parallel],
                    name=f"receiveInput_ICITransfer[{self.batch_size * self.decode_width * d_model_parallel}]",
                    description="DecodeReceiveInputFromPipelineICI",
                    config=self.config,
                    count=self.output_seqlen,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1
        # ops.append(
        #     ops_lib.create_input_op(
        #         input_shape=[self.batch_size, self.decode_width, d_model_parallel],
        #         name=f"receive_from_dcn(X) HBMTransfer[{self.batch_size * self.decode_width * d_model_parallel}]",
        #         description="Attention-serving-decode-Receive_input",
        #         count=count,
        #         fusion_id=fusion_id,
        #     )
        # )
        # fusion_id += 1
        ops.append(
            ops_lib.create_unary_op(
                input_shape=[self.batch_size, self.decode_width, d_model_parallel],
                op_name="LayerNorm",
                name="X_norm = LayerNorm(X)",
                description="Attention-serving-decode-Input_layernorm",
                count=count,
                fusion_id=fusion_id,
            )
        )
        fusion_id += 1
        ops += ops_lib.create_multi_head_attention(
            batch_size=self.batch_size,
            input_seqlen=self.input_seqlen,
            output_seqlen=self.output_seqlen,
            decode_width=self.decode_width,
            num_heads=self.num_heads,
            d_model=self.d_model,
            d_head=self.d_head,
            config=self.config,
            num_layers=self.num_layers,
            fusion_id_start=fusion_id,
            tensor_parallelism_axes=self.tensor_parallelism_axes,
            is_decode=True,
            ici_bw_GBps=self.tp_ici_bw_GBps,
        )
        fusion_id = ops[-1].fusion_id + 1
        ops += ops_lib.create_ffn(
            batch_size=self.batch_size,
            input_seqlen=self.input_seqlen,
            output_seqlen=self.output_seqlen,
            decode_width=self.decode_width,
            d_model=self.d_model,
            d_ff=self.d_ff,
            config=self.config,
            num_layers=self.num_layers,
            ffn_type=self.ffn_type,
            fusion_id_start=fusion_id,
            tensor_parallelism_axes=self.tensor_parallelism_axes,
            is_decode=True,
            ici_bw_GBps=self.tp_ici_bw_GBps,
        )

        # PP ICI output
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.decode_width, d_model_parallel],
                    name=f"sendOutput_ICITransfer[{self.batch_size * self.decode_width * d_model_parallel}]",
                    description="DecodeSendOutputToPipelineICI",
                    config=self.config,
                    count=self.output_seqlen,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP DCN output
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.decode_width, d_model_parallel],
                    name=f"sendOutput_DCNTTransfer[{self.batch_size * self.decode_width * d_model_parallel}]",
                    description="DecodeSendOutputToPipelineDCN",
                    config=self.config,
                    count=self.output_seqlen,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        return ops

    def generate(
        self,
        fusion_id_start: int = 2,
        dump_to_file: bool = True,
        separate_prefill_decode: bool = True,
        analyze_energy: bool = True,
        **kwargs,
    ) -> list[Operator.Operator] | tuple[list[Operator.Operator], ...]:
        prefill_ops = self.generate_prefill_ops(fusion_id_start=fusion_id_start)
        decode_ops = self.generate_decode_ops(
            fusion_id_start=prefill_ops[-1].fusion_id + 1
        )
        ops = prefill_ops + decode_ops

        ops = analysis_lib.fill_operators_execution_info(ops, self.config, analyze_energy=analyze_energy)
        # if analyze_energy:
        #     for op in ops:
        #         power_lib.analyze_operator_energy(
        #             op, self.config, pg_config="NoPG"
        #         )

        if dump_to_file:
            self.dump_to_file(separate_prefill_decode, ops, prefill_ops, decode_ops)

        if separate_prefill_decode:
            return ops, prefill_ops, decode_ops
        else:
            return ops

    def dump_to_file(self, separate_prefill_decode: bool, ops: list[Operator.Operator], prefill_ops: list[Operator.Operator], decode_ops: list[Operator.Operator]):
        logging.info(
            "Generating LLM ops and dumping to %s.",
            os.path.abspath(self.output_file_path),
        )
        if separate_prefill_decode:
            prefill_ops_dict = [Operator.to_csv_dict(op) for op in prefill_ops]
            decode_ops_dict = [Operator.to_csv_dict(op) for op in decode_ops]
            with open(
                self.output_file_path.replace(".csv", "_prefill.csv"), "w"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=prefill_ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(prefill_ops_dict)
            with open(
                self.output_file_path.replace(".csv", "_decode.csv"), "w"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=decode_ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(decode_ops_dict)
        ops_dict = [Operator.to_csv_dict(op) for op in ops]
        with open(self.output_file_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
            writer.writeheader()
            writer.writerows(ops_dict)

    def compute_memory_footprint_bytes(self, prefill_or_decode: str = "decode") -> int:
        """
        Compute the memory footprint of the LLM ops in bytes.
        """
        return mem_footprint_lib.get_llm_inference_mem_requirement(
            self.config, prefill_or_decode=prefill_or_decode
        )


LLMOpsGeneratorInference = LLMOpsGenerator
"""Just an alias of class LLMOpsGenerator."""


# ********************
# ***** TRAINING *****
# ********************
class LLMOpsGeneratorTraining(LLMOpsGeneratorBase):
    """LLM ops generator for training."""

    def __init__(self, config: dict[str, Any] | LLMConfig):
        super().__init__(config)

    def generate_prefill_ops_fwd(
        self, fusion_id_start: int = 2
    ) -> list[Operator.Operator]:
        ops: list[Operator.Operator] = []
        fusion_id = fusion_id_start

        d_model_parallel = ceil(self.d_model / self.tensor_parallelism)

        # PP DCN input
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"receiveInput_DCNTTransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="Fwd-ReceiveInputFromPipelineDCN",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP ICI input
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"receiveInput_ICITransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="Fwd-ReceiveInputFromPipelineICI",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        fusion_id += 1
        ops.append(
            ops_lib.create_unary_op(
                input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                op_name="LayerNorm",
                name="X_norm = LayerNorm(X)",
                description="Fwd-Attention-encoder-Input_layernorm",
                count=self.num_layers,
                fusion_id=fusion_id,
            )
        )
        fusion_id += 1
        ops += ops_lib.create_multi_head_attention(
            batch_size=self.batch_size,
            input_seqlen=self.input_seqlen,
            output_seqlen=self.output_seqlen,
            decode_width=self.decode_width,
            num_heads=self.num_heads,
            d_model=self.d_model,
            d_head=self.d_head,
            config=self.config,
            num_layers=self.num_layers,
            fusion_id_start=fusion_id,
            is_decode=False,
            use_flash_attention=self.use_flash_attention,
            tensor_parallelism_axes=self.tensor_parallelism_axes,
            ici_bw_GBps=self.tp_ici_bw_GBps,
        )
        fusion_id = ops[-1].fusion_id + 1
        ops += ops_lib.create_ffn(
            batch_size=self.batch_size,
            input_seqlen=self.input_seqlen,
            output_seqlen=self.output_seqlen,
            decode_width=self.decode_width,
            d_model=self.d_model,
            d_ff=self.d_ff,
            config=self.config,
            num_layers=self.num_layers,
            ffn_type=self.ffn_type,
            fusion_id_start=fusion_id,
            tensor_parallelism_axes=self.tensor_parallelism_axes,
            is_decode=False,
            ici_bw_GBps=self.tp_ici_bw_GBps,
        )

        # PP ICI output
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"SendOutput_ICITransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="SendOutputToPipelineICI",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP DCN output
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"SendOutput_DCNTTransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="SendOutputToPipelineDCN",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        return ops

    def generate_prefill_ops_bwd(
        self, fusion_id_start: int = 2
    ) -> list[Operator.Operator]:
        ops: list[Operator.Operator] = []
        fusion_id = fusion_id_start

        d_model_parallel = ceil(self.d_model / self.tensor_parallelism)

        # PP DCN input
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"receiveInput_DCNTTransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="Bwd-ReceiveInputFromPipelineDCN",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP ICI input
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"receiveInput_ICITransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="Bwd-ReceiveInputFromPipelineICI",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # receive HBM input
        # ops.append(
        #     ops_lib.create_input_op(
        #         input_shape=[self.batch_size, self.input_seqlen, d_model_parallel], # TODO since we load more during bwd should the size change
        #         name=f"receive_from_dcn(X) HBMTransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
        #         description="Receive_input",
        #         count=self.num_layers,
        #         fusion_id=fusion_id,
        #     )
        # )
        fusion_id += 1

        # reverse the "body" of the operation
        # ffn - DONE
        ops += ops_lib.create_ffn_bwd(
            batch_size=self.batch_size,
            input_seqlen=self.input_seqlen,
            output_seqlen=self.output_seqlen,
            decode_width=self.decode_width,
            d_model=self.d_model,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            ffn_type=self.ffn_type,
            config=self.config,
            fusion_id_start=fusion_id,
            tensor_parallelism_axes=self.tensor_parallelism_axes,
            is_decode=False,
            ici_bw_GBps=self.tp_ici_bw_GBps,
        )
        fusion_id = ops[-1].fusion_id + 1

        # mha - DONE
        ops += ops_lib.create_multi_head_attention_bwd(
            batch_size=self.batch_size,
            input_seqlen=self.input_seqlen,
            output_seqlen=self.output_seqlen,
            decode_width=self.decode_width,
            num_heads=self.num_heads,
            d_model=self.d_model,
            d_head=self.d_head,
            num_layers=self.num_layers,
            config=self.config,
            fusion_id_start=fusion_id,
            is_decode=False,
            use_flash_attention=self.use_flash_attention,
            tensor_parallelism_axes=self.tensor_parallelism_axes,
            ici_bw_GBps=self.tp_ici_bw_GBps,
        )
        fusion_id = ops[-1].fusion_id + 1

        # attention layernorm - DONE
        ops.extend(
            ops_lib.create_layernorm_op_bwd(
                input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                op_name="LayerNorm",
                name="X_norm = LayerNorm(X)",
                description="Bwd-Attention-encoder-Input_layernorm",
                count=self.num_layers,
                fusion_id=fusion_id,
            )
        )
        fusion_id += 1

        # PP ICI output
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"SendOutput_ICITransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="Bwd-SendOutputToPipelineICI",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP DCN output
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"SendOutput_DCNTTransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="Bwd-SendOutputToPipelineDCN",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        param_count = sum([op.stats.weight_size_bytes * op.stats.count for op in ops])
        ops += ops_lib.create_weight_update(
            config=self.config,
            param_count=param_count // 2,  # assumes 2-Byte (FP16) weights
            ici_data_parallel_axes=self.data_parallelism_axes,
            fusion_id_start=fusion_id,
        )
        # fusion_id = ops[-1]["Fusion index"] + 1

        return ops

    def generate(
        self, fusion_id_start: int = 2, dump_to_file: bool = True, analyze_energy: bool = True, **kwargs,
    ) -> list[Operator.Operator]:
        prefill_ops_fwd = self.generate_prefill_ops_fwd(fusion_id_start=fusion_id_start)
        prefill_ops_bwd = self.generate_prefill_ops_bwd(
            fusion_id_start=prefill_ops_fwd[-1].fusion_id + 1
        )
        ops = prefill_ops_fwd + prefill_ops_bwd

        ops = analysis_lib.fill_operators_execution_info(ops, self.config, analyze_energy=analyze_energy)

        # if analyze_energy:
        #     for op in ops:
        #         power_lib.analyze_operator_energy(
        #             op, self.config, pg_config="NoPG"
        #         )

        if dump_to_file:
            logging.info(
                "Generating LLM ops and dumping to %s.",
                os.path.abspath(self.output_file_path),
            )
            ops_dict = [Operator.to_csv_dict(op) for op in ops]
            with open(self.output_file_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(ops_dict)

        return ops

    def compute_memory_footprint_bytes(self) -> int:
        """
        Compute the memory footprint of the LLM ops in bytes.
        """
        return mem_footprint_lib.get_llm_training_mem_requirement(self.config)


class DeepSeekOpsGenerator(LLMOpsGeneratorBase):
    """DeepSeek model ops generator for inference."""

    def __init__(self, config: dict[str, Any] | DeepSeekConfig):
        super().__init__(config)
        if isinstance(config, dict):
            self.config = DeepSeekConfig.model_validate(config)
        if not isinstance(self.config, DeepSeekConfig):
            raise TypeError(
                "config must be a dict or an instance of DeepSeekConfig, "
                f"got self.config={type(self.config)}. input config={type(config)}."
            )
        self.num_dense_layers: int = self.config.num_dense_layers
        self.expert_tensor_parallelism_degree: int = (
            self.config.expert_tensor_parallelism_degree
        )
        """Expert tensor parallelism degree."""
        self.expert_parallelism_degree: int = self.config.expert_parallelism_degree
        """Expert parallelism degree."""
        self.num_expert_parallelism_axes: int = self.config.num_expert_parallel_axes
        """Number of physical axes for expert parallelism."""
        self.num_expert_tensor_parallelism_axes: int = (
            self.config.num_expert_tensor_parallel_axes
        )
        """Number of physical axes for expert tensor parallelism."""
        self.expert_tensor_parallelism_axes: list[int] = [
            1
        ] * self.num_expert_tensor_parallelism_axes
        """Dim size of each ICI expert tensor parallelism axis."""
        self.expert_parallelism_axes: list[int] = [1] * self.num_expert_parallelism_axes
        """Dim size of each ICI expert parallelism axis."""

        if self.num_expert_tensor_parallelism_axes > 0:
            self.expert_tensor_parallelism_axes = split_parallelism_degree(
                self.expert_tensor_parallelism_degree,
                self.num_expert_tensor_parallelism_axes,
            )
        if self.num_expert_parallelism_axes > 0:
            self.expert_parallelism_axes = split_parallelism_degree(
                self.expert_parallelism_degree, self.num_expert_parallelism_axes
            )

    def generate_prefill_ops(self, fusion_id_start: int = 2) -> list[Operator.Operator]:
        ops: list[Operator.Operator] = []
        fusion_id = fusion_id_start

        d_model_parallel = ceil(self.d_model / self.tensor_parallelism)

        # KVSwap: add ops to swap KV cache
        if self.config.enable_swap_kv_cache:
            # compute memory requirements
            weight_size_bytes = mem_footprint_lib.get_llm_inference_weight_mem_requirement(
                self.config
            )
            kv_cache_size_bytes = mem_footprint_lib.get_llm_inference_kv_cache_mem_requirement(
                self.config, "prefill"
            )
            total_mem_footprint_bytes = self.compute_memory_footprint_bytes("prefill")

            chip_mem_capacity_bytes = self.config.hbm_size_GB * 1024 * 1024 * 1024
            if total_mem_footprint_bytes > chip_mem_capacity_bytes:
                # swap in/out entire kvcache and weights
                swap_size_bytes = ceil(
                    (total_mem_footprint_bytes - chip_mem_capacity_bytes)
                    / min(self.config.max_swap_kv_cache_times_prefill, self.num_layers)  # swap per multiple layers to overlap with compute
                )
                ops.append(
                    ops_lib.create_kvswap_op(
                        input_shape=[swap_size_bytes],
                        name=f"PCIeMemSwapIn",
                        description="PCIeMemSwapIn",
                        config=self.config,
                        count=1,
                        fusion_id_start=fusion_id,
                        transfer_type="Input",
                    )
                )
                fusion_id += 1
                ops.append(
                    ops_lib.create_kvswap_op(
                        input_shape=[swap_size_bytes],
                        name=f"PCIeMemSwapOut",
                        description="PCIeMemSwapOut",
                        config=self.config,
                        count=1,
                        fusion_id_start=fusion_id,
                        transfer_type="Output",
                    )
                )
                fusion_id += 1

        # PP DCN input
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"receiveInput_DCNTTransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="PrefillReceiveInputFromPipelineDCN",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP ICI input
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"receiveInput_ICITransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="PrefillReceiveInputFromPipelineICI",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        ops.append(
            ops_lib.create_unary_op(
                input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                op_name="RMSNorm",
                name="X_norm = RMSNorm(X)",
                description="Fwd-Attention-encoder-Input_rmsnorm",
                count=self.num_layers,
                fusion_id=fusion_id,
            )
        )
        fusion_id += 1

        ops += ops_lib.create_multi_head_latent_attention(
            batch_size=self.batch_size,
            config=self.config,
            fusion_id_start=fusion_id,
            is_decode=False,
            description_prefix="Fwd-Attention-encoder-",
            use_flash_attention=self.use_flash_attention,
            tensor_parallelism_axes=self.tensor_parallelism_axes,
        )
        fusion_id = ops[-1].fusion_id + 1

        ops += ops_lib.create_ffn(
            batch_size=self.batch_size,
            input_seqlen=self.input_seqlen,
            output_seqlen=self.output_seqlen,
            decode_width=self.decode_width,
            d_model=self.d_model,
            d_ff=self.d_ff,
            config=self.config,
            # TODO: currently assuming all layers are MoE layers.
            # May want to separately model dense layers in DeepSeek models
            # (aggregated activated moe_inter_dim is similar to the dense layers).
            num_layers=self.num_layers,
            ffn_type="deepseek_moe",
            fusion_id_start=fusion_id,
            tensor_parallelism_axes=self.expert_tensor_parallelism_axes,
            is_decode=False,
            expert_parallelism_axes=self.expert_parallelism_axes,
        )
        fusion_id = ops[-1].fusion_id + 1

        # PP ICI output
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"SendOutput_ICITransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="PrefillSendOutputToPipelineICI",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP DCN output
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.input_seqlen, d_model_parallel],
                    name=f"SendOutput_DCNTTransfer[{self.batch_size * self.input_seqlen * d_model_parallel}]",
                    description="PrefillSendOutputToPipelineDCN",
                    config=self.config,
                    count=1,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        return ops

    def generate_decode_ops(self, fusion_id_start: int = 2) -> list[Operator.Operator]:
        ops: list[Operator.Operator] = []
        fusion_id = fusion_id_start
        count = self.num_layers * self.output_seqlen
        d_model_parallel = ceil(self.d_model / self.tensor_parallelism)

        # KVSwap: add ops to swap KV cache
        if self.config.enable_swap_kv_cache:
            # compute memory requirements
            weight_size_bytes = mem_footprint_lib.get_llm_inference_weight_mem_requirement(
                self.config
            )
            kv_cache_size_bytes = mem_footprint_lib.get_llm_inference_kv_cache_mem_requirement(
                self.config, "prefill"
            )
            total_mem_footprint_bytes = self.compute_memory_footprint_bytes("decode")

            chip_mem_capacity_bytes = self.config.hbm_size_GB * 1024 * 1024 * 1024
            if total_mem_footprint_bytes > chip_mem_capacity_bytes:
                # swap in/out entire kvcache and weights
                swap_size_bytes = ceil(
                    (total_mem_footprint_bytes - chip_mem_capacity_bytes)
                    / min(self.config.max_swap_kv_cache_times_decode, self.num_layers)  # swap per multiple layers to overlap with compute
                )
                ops.append(
                    ops_lib.create_kvswap_op(
                        input_shape=[swap_size_bytes],
                        name=f"PCIeMemSwapIn",
                        description="PCIeMemSwapIn",
                        config=self.config,
                        count=self.output_seqlen,
                        fusion_id_start=fusion_id,
                        transfer_type="Input",
                    )
                )
                fusion_id += 1
                ops.append(
                    ops_lib.create_kvswap_op(
                        input_shape=[swap_size_bytes],
                        name=f"PCIeMemSwapOut",
                        description="PCIeMemSwapOut",
                        config=self.config,
                        count=self.output_seqlen,
                        fusion_id_start=fusion_id,
                        transfer_type="Output",
                    )
                )
                fusion_id += 1

        # PP DCN input
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.decode_width, d_model_parallel],
                    name=f"receiveInput_DCNTTransfer[{self.batch_size * self.decode_width * d_model_parallel}]",
                    description="DecodeReceiveInputFromPipelineDCN",
                    config=self.config,
                    count=self.output_seqlen,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP ICI input
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_input_transfer_op(
                    input_shape=[self.batch_size, self.decode_width, d_model_parallel],
                    name=f"receiveInput_ICITransfer[{self.batch_size * self.decode_width * d_model_parallel}]",
                    description="DecodeReceiveInputFromPipelineICI",
                    config=self.config,
                    count=self.output_seqlen,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        ops.append(
            ops_lib.create_unary_op(
                input_shape=[self.batch_size, self.decode_width, d_model_parallel],
                op_name="RMSNorm",
                name="X_norm = RMSNorm(X)",
                description="Attention-serving-decode-Input_rmsnorm",
                count=count,
                fusion_id=fusion_id,
            )
        )
        fusion_id += 1
        ops += ops_lib.create_multi_head_latent_attention(
            batch_size=self.batch_size,
            config=self.config,
            fusion_id_start=fusion_id,
            is_decode=True,
            description_prefix="Attention-serving-decode-",
            use_flash_attention=self.use_flash_attention,
            tensor_parallelism_axes=self.tensor_parallelism_axes,
        )
        fusion_id = ops[-1].fusion_id + 1

        ops += ops_lib.create_ffn(
            batch_size=self.batch_size,
            input_seqlen=self.input_seqlen,
            output_seqlen=self.output_seqlen,
            decode_width=self.decode_width,
            d_model=self.d_model,
            d_ff=self.d_ff,
            config=self.config,
            # TODO: currently assuming all layers are MoE layers.
            # May want to separately model dense layers in DeepSeek models
            # (aggregated activated moe_inter_dim is similar to the dense layers).
            num_layers=self.num_layers,
            ffn_type="deepseek_moe",
            fusion_id_start=fusion_id,
            tensor_parallelism_axes=self.expert_tensor_parallelism_axes,
            expert_parallelism_axes=self.expert_parallelism_axes,
            is_decode=True,
        )

        # PP ICI output
        if self.pipeline_parallelism > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.decode_width, d_model_parallel],
                    name=f"sendOutput_ICITransfer[{self.batch_size * self.decode_width * d_model_parallel}]",
                    description="DecodeSendOutputToPipelineICI",
                    config=self.config,
                    count=self.output_seqlen,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        # PP DCN output
        if self.pipeline_parallelism_dcn > 1:
            ops.append(
                ops_lib.create_output_transfer_op(
                    input_shape=[self.batch_size, self.decode_width, d_model_parallel],
                    name=f"sendOutput_DCNTTransfer[{self.batch_size * self.decode_width * d_model_parallel}]",
                    description="DecodeSendOutputToPipelineDCN",
                    config=self.config,
                    count=self.output_seqlen,
                    fusion_id_start=fusion_id,
                )
            )
            fusion_id += 1

        return ops

    def generate(
        self,
        fusion_id_start: int = 2,
        dump_to_file: bool = True,
        separate_prefill_decode: bool = True,
        analyze_energy: bool = True,
        **kwargs,
    ) -> list[Operator.Operator] | tuple[list[Operator.Operator], ...]:
        prefill_ops = self.generate_prefill_ops(fusion_id_start=fusion_id_start)
        decode_ops = self.generate_decode_ops(
            fusion_id_start=prefill_ops[-1].fusion_id + 1
        )
        ops = prefill_ops + decode_ops

        ops = analysis_lib.fill_operators_execution_info(ops, self.config, analyze_energy=analyze_energy)

        # if analyze_energy:
        #     for op in ops:
        #         power_lib.analyze_operator_energy(
        #             op, self.config, pg_config="NoPG"
        #         )

        if dump_to_file:
            self.dump_to_file(separate_prefill_decode, ops, prefill_ops, decode_ops)

        if separate_prefill_decode:
            return ops, prefill_ops, decode_ops
        else:
            return ops

    def dump_to_file(self, separate_prefill_decode: bool, ops: list[Operator.Operator], prefill_ops: list[Operator.Operator], decode_ops: list[Operator.Operator]):
        logging.info(
            "Generating LLM ops and dumping to %s.",
            os.path.abspath(self.output_file_path),
        )
        if separate_prefill_decode:
            prefill_ops_dict = [Operator.to_csv_dict(op) for op in prefill_ops]
            decode_ops_dict = [Operator.to_csv_dict(op) for op in decode_ops]
            with open(
                self.output_file_path.replace(".csv", "_prefill.csv"), "w"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=prefill_ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(prefill_ops_dict)
            with open(
                self.output_file_path.replace(".csv", "_decode.csv"), "w"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=decode_ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(decode_ops_dict)
        ops_dict = [Operator.to_csv_dict(op) for op in ops]
        with open(self.output_file_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
            writer.writeheader()
            writer.writerows(ops_dict)

    def compute_memory_footprint_bytes(self, prefill_or_decode: str = "decode") -> int:
        """
        Compute the memory footprint of the LLM ops in bytes.
        For DeepSeek models, assume the FFN weights and activations are FP8.
        Other weights are BF16.
        Refer to the implementation in https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
        """
        assert isinstance(
            self.config, DeepSeekConfig
        ), f"Invalid config type: {type(self.config)}. Expected DeepSeekConfig."

        if prefill_or_decode == "prefill":
            # only count input sequence length for prefill
            seqlen = self.input_seqlen
        elif prefill_or_decode == "decode":
            # count both input+output sequence length for decode
            seqlen = self.input_seqlen + self.output_seqlen
        else:
            raise ValueError(
                f"Invalid prefill_or_decode value: {prefill_or_decode}. Expected 'prefill' or 'decode'."
            )

        tp = (
            self.config.tensor_parallelism_degree
            * self.config.tensor_parallel_degree_dcn
        )
        ep = (
            self.config.expert_parallelism_degree
            * self.config.expert_parallel_degree_dcn
        )
        etp = self.config.expert_tensor_parallelism_degree
        bs = self.batch_size  # This is microbatch size per PP stage in each DP replica
        num_layers = self.num_layers  # This is num layers per pipeline stage

        attn_weight_elem_bytes = 2  # BF16
        ffn_weight_elem_bytes = 1  # FP8
        ffn_act_elem_bytes = 2  # BF16 intermediate and output
        kv_cache_elem_bytes = 1  # FP8
        pe_cache_elem_bytes = 4  # FP32
        total_mem_footprint_bytes = 0

        ### MLA attention layer weights
        wq_a = self.config.d_model * self.config.q_lora_rank
        wq_b = (
            self.config.q_lora_rank
            * self.config.num_heads
            * self.config.qk_head_dim
            / tp
        )
        wkv_a = (
            self.config.d_model
            * self.config.kv_lora_rank
            * self.config.qk_rope_head_dim
        )
        wkv_b = (
            self.config.kv_lora_rank
            * self.config.num_heads
            * (self.config.qk_rope_head_dim + self.config.v_head_dim)
            / tp
        )
        wo = self.config.num_heads * self.config.v_head_dim * self.d_model / tp
        MLA_num_params = (wq_a + wq_b + wkv_a + wkv_b + wo) * num_layers
        MLA_weight_bytes = MLA_num_params * attn_weight_elem_bytes
        total_mem_footprint_bytes += MLA_weight_bytes
        # print(f"MLA attention layer weights: {MLA_weight_bytes} bytes")

        ### MLA attention layer activations (excluding KV cache)
        # q_nope (FP8) and q_pe (FP32 intermediate + FP8 result)
        q_act_bytes = (
            bs * seqlen * self.config.num_heads * (self.config.qk_rope_head_dim * (4 + 1) + self.config.qk_nope_head_dim * 1)
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
            self.config.d_model
            * self.config.moe_d_ff
            * 3
            / etp  # each expert has 3 matrices
            * (self.config.num_shared_experts + self.config.num_routed_experts)
            / ep  # total # of experts
            * num_layers
        )
        moe_gate_params = (
            self.config.num_routed_experts * self.config.d_model * num_layers
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
            bs * seqlen * self.config.moe_d_ff * 2 / etp
        ) * ffn_act_elem_bytes
        ffn_act_bytes = ffn_expert_act_bytes * num_layers
        total_mem_footprint_bytes += ffn_act_bytes
        # print(f"MoE FFN layer activations: {ffn_act_bytes} bytes")

        ### KV+PE cache
        kv_cache_bytes_per_token = (
            self.config.kv_lora_rank * kv_cache_elem_bytes
            + self.config.qk_rope_head_dim * pe_cache_elem_bytes
        ) * num_layers
        kv_cache_bytes = (
            kv_cache_bytes_per_token * bs * seqlen / tp
        )  # sequence parallelism over TP axes
        total_mem_footprint_bytes += kv_cache_bytes
        # print(f"KV+PE cache: {kv_cache_bytes} bytes")

        return round(total_mem_footprint_bytes)
