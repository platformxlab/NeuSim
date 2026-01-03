from copy import deepcopy
import csv
from functools import lru_cache
import os
from typing import Any
from math import ceil, sqrt
from absl import logging

import neusim.npusim.frontend.memory_footprint_analysis_lib as mem_footprint_lib
from neusim.npusim.frontend import op_analysis_lib as analysis_lib
import neusim.npusim.frontend.Operator as Operator
from neusim.configs.models.DiTConfig import DiTConfig
from neusim.configs.models.LLMConfig import LLMConfig
from neusim.npusim.frontend.llm_ops_generator import LLMOpsGenerator


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


class DiTOpsGenerator:
    '''
    Diffusion Transformer (DiT) inference. Just a wrapper of LLMOpsGenerator prefill pass since they are both transformer-based models.
    Original DiT paper: https://arxiv.org/pdf/2212.09748
    '''
    def __init__(self, config: dict[str, Any] | DiTConfig):
        if isinstance(config, dict):
            self.config: DiTConfig = DiTConfig.model_validate(config)
        else:
            self.config = config
        assert self.config.model_type == "dit", f"Invalid config: {self.config}"

        assert self.config.data_parallel_degree_dcn == 1, "DCN data parallelism is not supported yet."
        assert self.config.tensor_parallel_degree_dcn == 1, "DCN model parallelism is not supported yet."
        assert self.config.pipeline_parallel_degree_dcn == 1, "DCN pipeline parallelism is not supported yet."
        # assert self.config.data_parallelism_degree == 1, "data parallelism is not supported yet."
        # assert self.config.tensor_parallelism_degree == 1, "model parallelism is not supported yet."
        # assert self.config.pipeline_parallelism_degree == 1, "pipeline parallelism is not supported yet."

        # init underlying LLMOpsGenerator instance
        llm_config: LLMConfig = get_llm_config_from_dit_config(self.config)
        self.llm_ops_generator = LLMOpsGenerator(llm_config)


    def generate(self, fusion_id_start: int = 2, dump_to_file: bool = True, analyze_energy: bool = True) -> list[Operator.Operator]:
        '''
        Generate ops for DiT model.
        '''
        ops: list[Operator.Operator] = self.llm_ops_generator.generate_prefill_ops(fusion_id_start)
        for op in ops:
            op.stats.count *= self.config.num_diffusion_steps

        ops = analysis_lib.fill_operators_execution_info(
            ops, self.config, analyze_energy=analyze_energy
        )
        if dump_to_file:
            logging.info(
                "Generating DiT ops and dumping to %s.", os.path.abspath(self.config.output_file_path)
            )
            ops_dict = [Operator.to_csv_dict(op) for op in ops]
            with open(self.config.output_file_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(ops_dict)

        return ops


    def compute_memory_footprint_bytes(self) -> int:
        '''
        Compute the memory footprint in bytes.
        '''
        return mem_footprint_lib.get_dit_inference_mem_requirement(self.config)
