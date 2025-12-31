import csv
import json
import os
from typing import Any, Sequence
from math import ceil, sqrt
from absl import flags, logging

import numpy as np

import neusim.npusim.frontend.memory_footprint_analysis_lib as mem_footprint_lib
from neusim.npusim.frontend.Operator import Operator, Tensor
from neusim.configs.models.DLRMConfig import DLRMConfig

from neusim.npusim.frontend import llm_ops_lib as ops_lib
from neusim.npusim.frontend import op_analysis_lib as analysis_lib
from neusim.npusim.frontend.util import get_bisection_bw_per_chip_GBps


def pad_to_multiple_of(x: int, n: int) -> int:
    """
    Pad @x to the nearest multiple of @n.
    """
    return ceil(x / n) * n


class EmbeddingBag:
    """
    Config and parallelism info for an embedding table.
    """

    def __init__(
        self,
        embed_dim: int,
        num_entries: int,
        global_batch_size: int,
        num_indices_per_lookup: int = 1,
        row_sharding_degree: int = 1,
        column_sharding_degree: int = 1,
        data_parallel_degree: int = 1,
        chip_assignment: Sequence[int] | None = None,
    ):
        self.embed_dim: int = embed_dim
        """Embedding dimension size."""
        self.num_entries: int = num_entries
        """Number of entries in the embedding table."""
        self.num_indices_per_lookup: int = num_indices_per_lookup
        """
        Number of indices to lookup for each request.
        All retrieved entries are summed up to produce the final output.
        """

        self.row_sharding_degree: int = row_sharding_degree
        """
        Number of sub-tables to split the embedding table into.
        Each chip holds a subset of the entries (a sub-table).
        """
        self.column_sharding_degree: int = column_sharding_degree
        """
        TODO: for now, only support sharding the num_enrties (row) dimension.
        Maybe consider sharding the embed_dim (column) dimension in the future.
        """
        assert self.column_sharding_degree == 1, "Column sharding is not supported yet."
        self.data_parallel_degree: int = data_parallel_degree
        """
        Number of chips to replicate the embedding table.
        Each chip holds a full copy of the table.
        """
        self.chip_assignment: list[int] = list(chip_assignment or [])
        """
        Indices of assigned chips. Each chip holds a full copy of the table.
        """

        self.batch_size = ceil(global_batch_size / self.data_parallel_degree)
        """
        Local batch size on each chip for indexing this table.
        """

    def __str__(self) -> str:
        return f"""
            EmbeddingBag(
                embed_dim={self.embed_dim},
                num_entries={self.num_entries},
                num_indices_per_lookup={self.num_indices_per_lookup},
                row_sharding_degree={self.row_sharding_degree},
                column_sharding_degree={self.column_sharding_degree},
                data_parallel_degree={self.data_parallel_degree},
                chip_assignment={self.chip_assignment},
                batch_size={self.batch_size},
            )
        """

    def __repr__(self) -> str:
        return self.__str__()

    def generate_ops(self, fusion_id_start: int = 2) -> list[Operator]:
        """
        Generate embedding_ops for this embedding table.
        """
        ops: list[Operator] = []
        fusion_id = fusion_id_start

        ops += ops_lib.create_embedding_bag(
            batch_size=self.batch_size,
            num_indices=[self.num_indices_per_lookup],
            table_sizes=[ceil(self.num_entries / self.row_sharding_degree)],
            embedding_dim=self.embed_dim,
            fusion_id=fusion_id,
            name=f"embedding_bag_{fusion_id}",
            description=f"EmbeddingBag{fusion_id}",
        )

        return ops


class MultiLayerPerceptron:
    """
    Config and parallelism info for a multi-layer perceptron (MLP).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        global_batch_size: int,
        bias: bool = True,
        activation: str = "relu",
        data_parallel_degree: int = 1,
        name: str = "",
        description: str = "",
    ):
        self.in_features: int = in_features
        """Input feature size."""
        self.out_features: int = out_features
        """Output feature size."""
        self.bias: bool = bias
        """Whether to include a bias term."""
        self.activation: str = activation
        """Activation function name."""
        self.data_parallel_degree: int = data_parallel_degree
        """
        Number of chips to replicate the MLP.
        Each chip holds a full copy of the MLP.
        """
        self.batch_size: int = ceil(global_batch_size / self.data_parallel_degree)
        self.name_prefix: str = name
        self.description_prefix: str = description

    def generate_ops(self, fusion_id_start: int = 2) -> list[Operator]:
        """
        Generate MLP ops for this layer.
        """
        ops: list[Operator] = []
        fusion_id = fusion_id_start

        ops += [
            ops_lib.create_einsum_op(
                input_a_shape=[self.batch_size, self.in_features],
                input_b_shape=[self.in_features, self.out_features],
                einsum_expr="BI;IO->BO",
                dtype="DT_BFLOAT16",
                fusion_id=fusion_id_start,
                description=self.description_prefix + "_einsum",
                name=self.name_prefix + "_einsum",
                count=1,
            )
        ]

        return ops


class DLRMOpsGenerator:
    def __init__(self, config: dict[str, Any] | DLRMConfig):
        if isinstance(config, dict):
            self.config: DLRMConfig = DLRMConfig.model_validate(config)
        else:
            self.config = config
        assert self.config.model_type == "dlrm", f"Invalid config: {self.config}"

        # TODO: support DCN parallelism
        assert (
            self.config.data_parallel_degree_dcn == 1
        ), "DCN data parallelism is not supported yet."
        assert (
            self.config.tensor_parallel_degree_dcn == 1
        ), "DCN model parallelism is not supported yet."
        assert (
            self.config.pipeline_parallel_degree_dcn == 1
        ), "DCN pipeline parallelism is not supported yet."

        # TODO: maybe support pipeline parallelism
        assert (
            self.config.pipeline_parallelism_degree == 1
        ), "Pipeline parallelism is not supported yet."

        bisection_config = get_bisection_bw_per_chip_GBps(self.config)
        self.bisection_bw_GBps: float = bisection_config[0]
        self.topology: list[int] = bisection_config[1]

        self.embed_tables: list[EmbeddingBag] = [
            EmbeddingBag(
                embed_dim=self.config.embedding_dim,
                num_entries=self.config.embedding_table_sizes[i],
                global_batch_size=self.config.global_batch_size,
                num_indices_per_lookup=self.config.num_indices_per_lookup[i],
            )
            for i in range(len(self.config.embedding_table_sizes))
        ]
        self.mlp_bottom: list[MultiLayerPerceptron] = [
            MultiLayerPerceptron(
                in_features=self.config.bottom_mlp_config[i].in_features,
                out_features=self.config.bottom_mlp_config[i].out_features,
                global_batch_size=self.config.global_batch_size,
                bias=self.config.bottom_mlp_config[i].bias,
                activation=self.config.bottom_mlp_config[i].activation,
                data_parallel_degree=self.config.data_parallelism_degree,
                name=f"bottom_mlp_{i}",
                description=f"BottomMLP_{i}",
            )
            for i in range(len(self.config.bottom_mlp_config))
        ]

        self.num_interactions: int = pad_to_multiple_of(
            self.config.embedding_dim
            + (len(self.embed_tables) + 1) * len(self.embed_tables) // 2,
            128,  # pad to VPU dimension (8*128)
        )
        """
        Input dim of the first Top MLP layer.
        Derived from concatenating the interactions of all sparse and dense features and the dense feature.
        """
        self.mlp_top: list[MultiLayerPerceptron] = [
            MultiLayerPerceptron(
                in_features=self.num_interactions,
                out_features=self.config.top_mlp_config[0].in_features,
                global_batch_size=self.config.global_batch_size,
                bias=True,
                activation="relu",
                data_parallel_degree=self.config.data_parallelism_degree,
                name="top_mlp_init",
                description="TopMLP_init",
            )
        ]
        self.mlp_top += [
            MultiLayerPerceptron(
                in_features=self.config.top_mlp_config[i].in_features,
                out_features=self.config.top_mlp_config[i].out_features,
                global_batch_size=self.config.global_batch_size,
                bias=self.config.top_mlp_config[i].bias,
                activation=self.config.top_mlp_config[i].activation,
                data_parallel_degree=self.config.data_parallelism_degree,
                name=f"top_mlp_{i}",
                description=f"TopMLP_{i}",
            )
            for i in range(len(self.config.top_mlp_config))
        ]

        self.split_embedding_tables()

    ### END __init__() ###

    def split_embedding_tables(self):
        """
        Split embedding tables into multiple chips for model parallelism.
        Use heuristics to balance table size on different devices.
        TODO: for now, just hardcoded which tables are replicated and which are sharded based on table sizes.
        """
        # 1. Tables >=6GB will be sharded by the row dimension (row sharding).
        ROW_SHARDING_THRESHOLD_GB = 6
        # 2. Tables <=2GB will be replicated across all chips (data parallelism).
        REPLICATE_THRESHOLD_GB = 2
        # 3. Tables between 2GB and 6GB will be assigned to different chips (table sharding).
        ## /// ##

        num_chips = self.config.num_chips
        table_parallel_tables: list[EmbeddingBag] = []
        for table in self.embed_tables:
            # assume 4 byte (FP32) elements
            table_size_GB = table.num_entries * table.embed_dim * 4 / 1024 / 1024 / 1024
            if table_size_GB >= ROW_SHARDING_THRESHOLD_GB:  # 1
                table.row_sharding_degree = self.config.tensor_parallelism_degree
                table.data_parallel_degree = 1
                table.chip_assignment = list(range(num_chips))
            elif table_size_GB <= REPLICATE_THRESHOLD_GB:  # 2
                table.row_sharding_degree = 1
                table.data_parallel_degree = self.config.data_parallelism_degree
                table.chip_assignment = list(range(num_chips))
            else:
                table.row_sharding_degree = 1
                table.data_parallel_degree = 1
                table_parallel_tables.append(table)

        # Assign intermediate-sized (2GB-6GB) tables to chips in a round-robin manner.
        # Ideally, this should be done based on table size and # of lookup entries.
        # For now, since this case only applies to DLRM-S with 2.38GB tables in our benchmark models,
        # we just assign tables to chips in a round-robin manner.
        num_chips = self.config.num_chips
        for i, table in enumerate(table_parallel_tables):
            table.chip_assignment = [i % num_chips]  # 3

    def get_sharded_tables_on_chip_id(self, chip_id: int) -> list[EmbeddingBag]:
        """
        Returns the list of sharded embedding tables on chip @chip_id.
        These sharded tables (table/row/column sharding) will incur all-to-all communication.
        Replicated tables (data parallelism), on the contrary, will not incur all-to-all communication.
        """
        return [
            table
            for table in self.embed_tables
            if (
                (chip_id in table.chip_assignment)
                and (
                    table.row_sharding_degree > 1
                    or table.column_sharding_degree > 1
                    or table.data_parallel_degree < self.config.data_parallelism_degree
                )
            )
        ]

    def generate_embedding_ops(
        self, chip_index: int, fusion_id_start: int = 2
    ) -> list[Operator]:
        """
        Generate an embedding_op for each table for chip @chip_index (integer from 1 to @self.config.num_chips).
        """
        vocab_sizes = self.config.embedding_table_sizes
        embed_dim = self.config.embedding_dim

        ops: list[Operator] = []
        fusion_id = fusion_id_start

        for table in self.embed_tables:
            if chip_index in table.chip_assignment:
                ops += table.generate_ops(fusion_id)
                fusion_id += ops[-1].fusion_id

        return ops

    def generate_top_mlp_ops(self, fusion_id_start: int = 2) -> list[Operator]:
        ops: list[Operator] = []
        fusion_id = fusion_id_start

        for mlp in self.mlp_top:
            ops += mlp.generate_ops(fusion_id)
            fusion_id += ops[-1].fusion_id

        return ops

    def generate_bottom_mlp_ops(self, fusion_id_start: int = 2) -> list[Operator]:
        ops: list[Operator] = []
        fusion_id = fusion_id_start

        for mlp in self.mlp_bottom:
            ops += mlp.generate_ops(fusion_id)
            fusion_id += ops[-1].fusion_id

        return ops

    def generate_interaction_ops(
        self, fusion_id_start: int = 2
    ) -> list[Operator]:
        """
        Generate interaction ops for DLRM.
        """
        ops: list[Operator] = []
        fusion_id = fusion_id_start

        batch_size = ceil(
            self.config.global_batch_size / self.config.data_parallelism_degree
        )
        bottom_output_shape = [
            batch_size,
            len(self.embed_tables) + 1,
            self.config.embedding_dim,
        ]
        bottom_output_shape_T = [
            bottom_output_shape[0],
            bottom_output_shape[2],
            bottom_output_shape[1],
        ]
        ops += [
            ops_lib.create_einsum_op(
                input_a_shape=bottom_output_shape,
                input_b_shape=bottom_output_shape_T,
                einsum_expr="BNE;BEN->BNN",
                dtype="DT_BFLOAT16",
                fusion_id=fusion_id,
                description="DotInteraction_einsum",
                name="DotInteraction_einsum",
                count=1,
            )
        ]

        return ops

    def generate_ops_for_chip_id(
        self, chip_id: int, fusion_id_start: int = 2, dump_to_file: bool = True, analyze_energy: bool = True
    ) -> list[Operator]:
        """
        Generate DLRM ops for chip @chip_id.
        """
        ops: list[Operator] = []
        fusion_id = fusion_id_start

        # assume the inputs are sharded with data parallelism at the beginning
        ops += self.generate_bottom_mlp_ops(fusion_id)
        fusion_id += ops[-1].fusion_id

        # compute total number of lookup-ed features in the batch on this chip
        # sharded_tables_on_chip = self.get_sharded_tables_on_chip_id(chip_id)
        # num_features_per_batch = sum([
        #     table.num_indices_per_lookup
        #     for table in sharded_tables_on_chip
        # ])

        # all to all indices for the embedding lookups
        ops += [
            ops_lib.create_all_to_all_op(
                # assume the worst-case scenario with max traffic
                input=Tensor.from_shape(
                    name="EmbeddingIndicesAllToAllInput",
                    shape=[
                        self.config.global_batch_size,
                        len(self.embed_tables),
                        max(self.config.num_indices_per_lookup),
                    ],
                ),
                config=self.config,
                num_parallelism=len(self.embed_tables),
                bisection_bw=self.bisection_bw_GBps,
                dtype="DT_INT32",
                fusion_id=fusion_id,
                name="EmbeddingIndicesAllToAll",
                description=f"EmbeddingIndicesAllToAll-{fusion_id}",
            )
        ]
        fusion_id += 1

        ops += self.generate_embedding_ops(chip_id, fusion_id)
        fusion_id += ops[-1].fusion_id

        # all to all sparse features
        ops += [
            ops_lib.create_all_to_all_op(
                # assume the worst-case scenario with max traffic
                input=Tensor.from_shape(
                    name="EmbeddingAllToAllInput",
                    shape=[
                        self.config.global_batch_size,
                        len(self.embed_tables),
                        max(self.config.num_indices_per_lookup),
                        self.config.embedding_dim,
                    ],
                ),
                config=self.config,
                num_parallelism=len(self.embed_tables),
                bisection_bw=self.bisection_bw_GBps,
                dtype="DT_FLOAT",  # fp32
                fusion_id=fusion_id,
                name="EmbeddingAllToAll",
                description=f"EmbeddingAllToAll-{fusion_id}",
            )
        ]
        fusion_id += 1

        ops += self.generate_interaction_ops(fusion_id)
        fusion_id += ops[-1].fusion_id

        ops += self.generate_top_mlp_ops(fusion_id)
        fusion_id += ops[-1].fusion_id

        ops = analysis_lib.fill_operators_execution_info(ops, self.config, analyze_energy=analyze_energy)

        if dump_to_file:
            output_path = os.path.abspath(self.config.output_file_path)
            logging.info(
                "Generating DLRM ops and dumping to %s (chip_id=%s)",
                output_path,
                chip_id,
            )
            ops_dict = [op.to_csv_dict() for op in ops]
            with open(output_path.replace(".csv", f"_chip{chip_id}.csv"), "w") as f:
                writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(ops_dict)

        return ops

    def generate(
        self, fusion_id_start: int = 2, dump_to_file: bool = True, analyze_energy: bool = True
    ) -> list[list[Operator]]:
        all_ops: list[list[Operator]] = []

        for chip_id in range(self.config.num_chips):
            all_ops.append(
                self.generate_ops_for_chip_id(chip_id, fusion_id_start, dump_to_file, analyze_energy)
            )

        # pick the chip that executes for the longest time as the representative performance
        all_exe_times = [
            (sum([op.stats.execution_time_ns for op in ops]), ops, i)
            for i, ops in enumerate(all_ops)
        ]
        max_exe_time = max(all_exe_times, key=lambda x: x[0])
        max_time, max_time_ops, max_time_chip_id = max_exe_time
        max_time_ops_dict = [
            op.to_csv_dict() for op in max_time_ops
        ]

        if dump_to_file:
            output_path = os.path.abspath(self.config.output_file_path)
            logging.info(
                "Generating DLRM ops and dumping to %s (straggler chip %s)",
                output_path,
                max_time_chip_id,
            )
            with open(output_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=max_time_ops_dict[0].keys())
                writer.writeheader()
                writer.writerows(max_time_ops_dict)

            # dump table sharding info and ICI bisection BW
            config_stats = {}
            config_stats["ICI_bisection_BW_per_chip_GBps"] = self.bisection_bw_GBps
            config_stats["topology"] = self.topology
            config_stats["table_sharding"] = [table.__dict__ for table in self.embed_tables]

            with open(output_path.replace(".csv", "_config_stats.json"), "w") as f:
                json.dump(config_stats, f, indent=4)

        return all_ops


    def compute_memory_footprint_bytes(self) -> int:
        '''
        Compute the memory footprint in bytes.
        '''
        return mem_footprint_lib.get_dlrm_inference_mem_requirement(self.config)
