"""
The backend implementation of NeuSim
Version: 0.1.0
"""


from functools import lru_cache
import re
from typing import Any, Dict, List, Sequence, Tuple

from absl import logging
import numpy as np

import neusim.npusim.frontend.Operator as Operator
import neusim.npusim.backend.util as util
import neusim.xla_hlo_parser.xla_hlo_structures as hlo_struct
from neusim.configs.chips.ChipConfig import ChipConfig


EINSUM_OP_TYPES = [
    "Einsum", "MatMul", "BatchMatMulV2", "Mul"
]


def parse_input_tensor_shapes(input_tensor_shape_str: str) -> Tuple[List[List[int]], List[str]]:
    '''Returns (shape, dtype)'''
    tensor_shape_regex_pattern = r",*DT_[A-Z|0-9]+:"
    shapes = re.split(tensor_shape_regex_pattern, input_tensor_shape_str)
    shapes = [x for x in shapes if len(x) > 0]
    shapes = [x.removeprefix("[").removesuffix("]") for x in shapes]
    shapes = [
        [int(y) for y in x.split(",")]
        for x in shapes
    ]

    dtype_regex_pattern = r"DT_[A-Z|0-9]+:"
    dtypes = re.findall(dtype_regex_pattern, input_tensor_shape_str)
    dtypes = [x.removesuffix(":") for x in dtypes]

    return shapes, dtypes


def parse_output_tensor_shapes(output_tensor_shape_str: str) -> Tuple[List[List[int]], List[str]]:
    '''Returns (shape, dtype)'''
    tensor_shape_regex_pattern = r",*DT_[A-Z|0-9]+:"
    shapes = re.split(tensor_shape_regex_pattern, output_tensor_shape_str)
    shapes = [x.removeprefix("[").removesuffix("]").removesuffix("],[") for x in shapes]
    shapes = [x for x in shapes if len(x) > 0]
    shapes = [
        [int(y) for y in x[1:-1].split(",")]  # y is like "(1, 2, 3)"
        for x in shapes
    ]

    dtype_regex_pattern = r"DT_[A-Z|0-9]+:"
    dtypes = re.findall(dtype_regex_pattern, output_tensor_shape_str)
    dtypes = [x.removesuffix(":") for x in dtypes]

    return shapes, dtypes


def parse_tensor_shapes_for_node_cost_conv(node_cost: Operator.Operator, hlo_module: hlo_struct.HLOModule) -> Tuple[hlo_struct.HLOInstruction, Operator.Operator]:
    # if node_cost.opcode not in ["convolution", "Conv2D", "Einsum", "MatMul", "BatchMatMulV2"]:
    if node_cost.opcode_type not in [Operator.OpcodeType.CONV2D, Operator.OpcodeType.EINSUM]:
        raise ValueError(f"Op not supported: {node_cost}")
    assert isinstance(node_cost, (Operator.EinsumOperator, Operator.Conv2DOperator)), \
        f"node_cost is not an Einsum or Conv2D operator: {node_cost}"

    input_shapes, input_dtypes = parse_input_tensor_shapes(node_cost.input_tensor_shape_str)
    output_shapes, output_dtypes = parse_output_tensor_shapes(node_cost.output_tensor_shape_str)
    assert len(output_shapes) == 1, f"Output tensor shapes not expected: {output_shapes}"

    op_name = node_cost.name.split("/")[-1]
    I = hlo_module.getInstructionByName(op_name)
    assert I is not None, f"Instruction not found: {op_name}"
    # assert "dim_labels" in I.metadata, f"dim_labels not found for {op_name}"
    if "dim_labels" not in I.metadata:
        assert len(I.input_axes[0]), "Input axes not found"
        node_cost.stats.dim_labels_str = (
            "".join([ax.name for ax in I.input_axes[0]]) + "," +
            "".join([ax.name for ax in I.input_axes[1]]) + "->" +
            "".join([ax.name for ax in I.output_axes])
        )
        return I, node_cost

    node_cost.stats.dim_labels_str = I.metadata["dim_labels"]

    # lhs axes
    for ax in I.input_axes[0]:
        ax.size = input_shapes[0][ax.index]
        ax.data_type = input_dtypes[0]
    # rhs axes
    for ax in I.input_axes[1]:
        ax.size = input_shapes[1][ax.index]
        ax.data_type = input_dtypes[1]
    # output axes
    for ax in I.output_axes:
        ax.size = output_shapes[0][ax.index]
        ax.data_type = output_dtypes[0]

    # # special hack for each model
    # if "dlrm" in hlo_module.name:
    #     I.metadata["op_type"] = "MatMul"
    # elif "bert" in hlo_module.name:
    #     if I.metadata["op_type"] == "unknown":
    #         I.metadata["op_type"] = "Einsum"
    # elif "transformer" in hlo_module.name:
    #     if I.metadata["op_type"] == "unknown":
    #         I.metadata["op_type"] = "Einsum"

    def try_check_einsum() -> bool:
        '''
        try to check the parsed axes as an Einsum operator.
        If failed, then treat this op as a Conv2d.
        '''
        if I.metadata["op_type"] == "Conv2D":
            return False

        # for MatMul: remove axes of size 1 that is used as placeholder in HLO
        input0_axes = [ax for ax in I.input_axes[0] if ax.size != 1]
        input1_axes = [ax for ax in I.input_axes[1] if ax.size != 1]
        output_axes = [ax for ax in I.output_axes if ax.size != 1]

        # For Einsum/MatMul, axes with the same name have the same size.
        all_axes = input0_axes + input1_axes + output_axes
        axes_dict = {}
        for ax in all_axes:
            if ax.name not in axes_dict:
                axes_dict[ax.name] = ax.size
            if not axes_dict[ax.name] == ax.size:
                return False
        return True

    if try_check_einsum():
        # if check is a success, this op is a matmul
        if I.metadata["op_type"] not in EINSUM_OP_TYPES:
            I.metadata["op_type"] = "Einsum"
        I.input_axes[0] = [ax for ax in I.input_axes[0] if ax.size != 1]
        I.input_axes[1] = [ax for ax in I.input_axes[1] if ax.size != 1]
        I.output_axes = [ax for ax in I.output_axes if ax.size != 1]
    else:
        # otherwise treat this op as a conv2d
        I.metadata["op_type"] = "Conv2D"

    # # for MatMul: remove axes of size 1 that is used as placeholder in HLO
    # if I.metadata["op_type"] in EINSUM_OP_TYPES + ["unknown"]:
    #     I.input_axes[0] = [ax for ax in I.input_axes[0] if ax.size != 1]
    #     I.input_axes[1] = [ax for ax in I.input_axes[1] if ax.size != 1]
    #     I.output_axes = [ax for ax in I.output_axes if ax.size != 1]

    # For Einsum/MatMul, axes with the same name have the same size;
    # for Convolution, spatial0/1 axes have different sizes in kernel (input1). ?
    # all_axes = I.input_axes[0] + I.input_axes[1] + I.output_axes
    # axes_dict = {}
    # for ax in all_axes:
    #     if ax.name not in axes_dict:
    #         axes_dict[ax.name] = ax.size
    #     else:
    #         if I.metadata["op_type"] in EINSUM_OP_TYPES:
    #             # For Einsum/MatMul, check axis sizes match
    #             assert axes_dict[ax.name] == ax.size, f"axes size not consistent: {ax.name}"
    #         else:
    #             # If not match, guess this is a Conv2D
    #             I.metadata["op_type"] = "Conv2D"

    return I, node_cost


def parse_tensor_shapes_for_node_cost_default(node_cost: Operator.Operator, hlo_module: hlo_struct.HLOModule) -> tuple[hlo_struct.HLOInstruction, Operator.Operator]:
    input_shapes, input_dtypes = parse_input_tensor_shapes(node_cost.input_tensor_shape_str)
    output_shapes, output_dtypes = parse_output_tensor_shapes(node_cost.output_tensor_shape_str)
    assert len(output_shapes) == 1, f"Output tensor shapes not expected: {output_shapes}"

    op_name = node_cost.name.split("/")[-1]
    I = hlo_module.getInstructionByName(op_name)
    assert I is not None, f"Instruction not found: {op_name}"

    if hasattr(I, "input_axes"):
        assert len(I.input_axes) == len(input_shapes), \
            f"Input axes shape does not match: {len(I.input_axes)} vs. {len(input_shapes)}"
    else:
        I.input_axes = [
            [
                hlo_struct.HLOAxis(f"{i}", i, shape[i], dtype)
                for i in range(len(shape))
            ]
            for shape, dtype in zip(input_shapes, input_dtypes)
        ]
        I.output_axes = [
            hlo_struct.HLOAxis(f"{i}", i, output_shapes[0][i], output_dtypes[0])
            for i in range(len(output_shapes[0]))
        ]

    for i, input_tensor in enumerate(I.input_axes):
        for ax in input_tensor:
            ax.size = input_shapes[i][ax.index]
            ax.data_type = input_dtypes[i]

    return I, node_cost


def parse_tensor_shapes_for_node_cost(node_cost: Operator.Operator, hlo_module: hlo_struct.HLOModule) -> tuple[hlo_struct.HLOInstruction, Operator.Operator]:
    if node_cost.opcode_type in [Operator.OpcodeType.CONV2D, Operator.OpcodeType.EINSUM]:
        return parse_tensor_shapes_for_node_cost_conv(node_cost, hlo_module)
    else:
        return parse_tensor_shapes_for_node_cost_default(node_cost, hlo_module)


# def parse_tensor_shapes_for_op(op: Operator.Operator, hlo_module: hlo_struct.HLOModule) -> tuple[hlo_struct.HLOInstruction, Operator.Operator]:
#     if op.opcode in ["convolution", "Conv2D", "Einsum", "MatMul", "BatchMatMulV2"]:
#         I, op_dict = parse_tensor_shapes_for_node_cost_conv(Operator.to_csv_dict(op), hlo_module)
#     else:
#         I, op_dict = parse_tensor_shapes_for_node_cost_default(Operator.to_csv_dict(op), hlo_module)

#     return I, Operator.from_csv_dict(op_dict)


def separate_axes_by_type_for_matmul(
    lhs_axes: Sequence[hlo_struct.HLOAxis],
    rhs_axes: Sequence[hlo_struct.HLOAxis],
    output_axes: Sequence[hlo_struct.HLOAxis],
) -> Tuple[List[hlo_struct.HLOAxis], List[hlo_struct.HLOAxis], List[hlo_struct.HLOAxis], List[hlo_struct.HLOAxis]]:
    '''
    Given LHS, RHS, and output axes for a MatMul operator, separate the axes into batch, reduction, and lhs/rhs non-reduction axes.
    The returned reduction axes respect the axes order in lhs_axes.

    For a BatchMatMul [b, m, k] * [b, k, n] -> [b, m, n], the axes are separated as follows:
        batch axes: b, set of axes that are common to all input/output tensors
        reduction axes: k, set of axes that are present in both input tensors but not the output tensor
        non-reduction axes: m, n, set of axes that are present in the output tensor but not both input tensors
    '''
    # 1. Find m, n, k for the matmul
    lhs_axes_names = {ax.name for ax in lhs_axes}
    rhs_axes_names = {ax.name for ax in rhs_axes}
    output_axes_names = {ax.name for ax in output_axes}

    # 1.1 find the reduction axis size (k)
    #   reduction axes are the axes that are present in both input tensors but not the output tensor
    reduction_axes_names = lhs_axes_names.intersection(rhs_axes_names)
    reduction_axes_names = reduction_axes_names.difference(output_axes_names)
    reduction_axes = [ax for ax in lhs_axes if ax.name in reduction_axes_names]
    # axis size (k)

    # 1.2 find the batch axix size (b)
    #   batch axes are the axes that are present in all input/output tensors
    batch_axes_names = lhs_axes_names.intersection(rhs_axes_names).intersection(output_axes_names)
    batch_axes = [ax for ax in lhs_axes if ax.name in batch_axes_names]

    # 1.3 find the non-reduction axis sizes (m, n)
    #   the remaining axes are the non-reduction axes
    lhs_non_reduct_axes_names = lhs_axes_names.difference(reduction_axes_names).difference(batch_axes_names)
    rhs_non_reduct_axes_names = rhs_axes_names.difference(reduction_axes_names).difference(batch_axes_names)
    lhs_non_reduct_axes = [ax for ax in lhs_axes if ax.name in lhs_non_reduct_axes_names]
    rhs_non_reduct_axes = [ax for ax in rhs_axes if ax.name in rhs_non_reduct_axes_names]

    return batch_axes, reduction_axes, lhs_non_reduct_axes, rhs_non_reduct_axes


def compute_bytes_accessed_from_tile_shape_matmul(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    tile_shapes: Sequence[Sequence[int]],
) -> int:
    '''
    Compute HBM bytes accessed for MatMul given input/output tensor and tile shapes.
    @param node_cost: the operator object containing the node cost information
    @param tile_shapes: a list of list of integers representing the tile shapes for lhs, rhs, and output
    @return: the number of bytes accessed from/to HBM
    '''
    assert len(tile_shapes) == 3, f"tile_shape not expected: {tile_shapes}"
    lhs_tile_shape, rhs_tile_shape, output_tile_shape = tile_shapes
    lhs_shape = [ax.size for ax in I.input_axes[0]]
    rhs_shape = [ax.size for ax in I.input_axes[1]]
    output_shape = [ax.size for ax in I.output_axes]

    ### For BatchMatMul [b, m, k] * [b, k, n] -> [b, m, n] with tile size [B, M, K, N] along each dimension,
    ### the number of output elements accessed from/to HBM is:
    ###     bmn;
    ### the number of input elements accessed from/to HBM is:
    ###     for each output tile: B(Mk + kN) = Bk(M+N);
    ###         lhs: BMk;
    ###         rhs: BNk;
    ###     for all output tiles: b * (mn)/MN * k(M+N);
    ###         lhs: b * (mn)/MN * Mk = bkmn/N;
    ###         rhs: b * (mn)/MN * Nk = bkmn/M;

    # 1. Find m, n, k for the matmul
    lhs_axes = I.input_axes[0]
    rhs_axes = I.input_axes[1]
    output_axes = I.output_axes
    batch_axes, reduction_axes, lhs_non_reduct_axes, rhs_non_reduct_axes = separate_axes_by_type_for_matmul(lhs_axes, rhs_axes, output_axes)

    # axis size (k)
    reduction_axes_agg_size = int(np.prod([ax.size for ax in reduction_axes]))

    # axis size (b)
    if len(batch_axes) == 0:
        # batch axes are not always required
        batch_axes_agg_size = 1
    else:
        batch_axes_agg_size = int(np.prod([ax.size for ax in batch_axes]))

    # axis size (m)
    lhs_non_reduct_axes_agg_size = int(np.prod([ax.size for ax in lhs_non_reduct_axes]))
    # axis size (n)
    rhs_non_reduct_axes_agg_size = int(np.prod([ax.size for ax in rhs_non_reduct_axes]))

    # 2 find the non-reduction axis tile sizes (M, N)
    lhs_tile_sizes = [lhs_tile_shape[ax.index] for ax in lhs_non_reduct_axes]
    rhs_tile_sizes = [rhs_tile_shape[ax.index] for ax in rhs_non_reduct_axes]
    # tile size (M)
    lhs_tile_agg_size = int(np.prod(lhs_tile_sizes))
    # tile size (N)
    rhs_tile_agg_size = int(np.prod(rhs_tile_sizes))

    # compute number of tiles
    output_agg_size = int(np.prod(output_shape))
    output_tile_agg_size = int(np.prod(output_tile_shape))
    num_tiles = int(np.ceil(output_agg_size / output_tile_agg_size))
    assert isinstance(node_cost, Operator.EinsumOperator), \
        f"node_cost must be an EinsumOperator, got {type(node_cost)}"
    node_cost.stats.num_tiles = num_tiles

    # 3. Compute the number of bytes accessed from/to HBM
    lhs_dtype_bytes = util.get_size_bytes_from_dtype(I.input_axes[0][0].data_type)
    rhs_dtype_bytes = util.get_size_bytes_from_dtype(I.input_axes[1][0].data_type)
    output_dtype_bytes = util.get_size_bytes_from_dtype(I.output_axes[0].data_type)
    bmn = batch_axes_agg_size * lhs_non_reduct_axes_agg_size * rhs_non_reduct_axes_agg_size
    bkmn = bmn * reduction_axes_agg_size
    bytes_accessed = (
        # output: (bmn) * dtype_bytes
        bmn * output_dtype_bytes +
        # lhs: bkmn/N * dtype_bytes
        int(np.ceil(bkmn / rhs_tile_agg_size) * lhs_dtype_bytes) +
        # rhs: bkmn/M * dtype_bytes
        int(np.ceil(bkmn / lhs_tile_agg_size) * rhs_dtype_bytes)
    )

    return bytes_accessed


@lru_cache(maxsize=None)
def find_best_tile_shape_for_matmul(
    lhs_dtype_bytes: int,
    rhs_dtype_bytes: int,
    output_dtype_bytes: int,
    sa_dim: int,
    lhs_agg_size: int,
    rhs_agg_size: int,
    reduction_agg_size: int,
    batch_axes_agg_size: int,
    vmem_bytes: int,
    nc_compute_ns: int,
    freq_GHz: float,
    hbm_bw_GBps: float,
):
    def tile_size_bytes(M: int, N: int, K: int) -> int:
        ts_bytes = M * N * output_dtype_bytes + M * K * lhs_dtype_bytes + K * N * rhs_dtype_bytes
        logging.vlog(3, "M=%d, N=%d, K=%d -> tile_size_bytes=%d", M, N, K, ts_bytes)
        return ts_bytes

    def num_MXU_ops(M: int, N: int, K: int) -> int:
        '''
        Compute number of MXU push/matmul/pop operations for a given tile size M, N, and K.
        Assuming sa_dim*sa_dim*2*4byte is the min tile size for a MXU,
        and approximately 100 cycles for the MXU to consume two such input tiles
        and produce one such output tile, if the HBM latency is 500ns, the number of compute
        operations must be large enough to consume more than 5 times the min tile size to hide the HBM latency.
        With double buffering, the effectively require a total tile size satisfying num_MXU_ops > 10.
        '''
        num_ops = max(
            3,  # at least one push+matmul+pop
            # MN/sa_dim^2 output tiles, each of which has K/sa_dim pipelined push+matmul+pop.
            # The ceilings here account for MXU padding.
            int(np.ceil(M / sa_dim)) * int(np.ceil(N / sa_dim)) * int(np.ceil(K / sa_dim)),
        )
        logging.vlog(3, "M=%d, N=%d, K=%d -> num_MXU_ops=%d", M, N, K, num_ops)
        return num_ops

    def HBM_bytes_accessed(M: int, N: int) -> int:
        '''
        Compute the total HBM bytes accessed for a given tile size M, N, and K,
        and axes sizes m, n, and k.
        total num_accesses = b(mn)/(MN) * k(M+N) + bmn
            output = bmn
            lhs = bkmn/N
            rhs = bkmn/M
        '''
        mn = lhs_agg_size * rhs_agg_size
        kmn = mn * reduction_agg_size
        return batch_axes_agg_size * (
            mn * output_dtype_bytes
            + int(np.ceil(kmn / N) * lhs_dtype_bytes)
            + int(np.ceil(kmn / M) * rhs_dtype_bytes)
        )

    lhs_agg_factors = util.get_factors(lhs_agg_size)
    rhs_agg_factors = util.get_factors(rhs_agg_size)
    reduction_agg_factors = util.get_factors(reduction_agg_size)

    # requires at least around 10 MXU ops to hide HBM latency
    NUM_MXU_OPS_TO_HIDE_HBM_LATENCY = 10

    agg_tile_shapes_candidates = []
    num_ops_lower_bound = NUM_MXU_OPS_TO_HIDE_HBM_LATENCY
    while len(agg_tile_shapes_candidates) == 0:
        for lhs_agg_f in lhs_agg_factors: # M
            for rhs_agg_f in rhs_agg_factors: # N
                for reduct_agg_f in reduction_agg_factors: # K
                    num_ops = num_MXU_ops(lhs_agg_f, rhs_agg_f, reduct_agg_f)
                    ts_bytes = tile_size_bytes(lhs_agg_f, rhs_agg_f, reduct_agg_f)
                    num_accesses = HBM_bytes_accessed(lhs_agg_f, rhs_agg_f)
                    if num_ops > num_ops_lower_bound and ts_bytes <= vmem_bytes:
                        # All push/matmul/pop ops are pipelined. Each op is ~100 cycles.
                        # num_tiles = int(np.ceil((lhs_agg_size * rhs_agg_size) / (lhs_agg_f * rhs_agg_f)))
                        # MXU_cycles = 100 * (num_ops + 2) * num_tiles
                        MXU_cycles = nc_compute_ns * freq_GHz
                        mem_cycles = num_accesses / (hbm_bw_GBps * 1024 * 1024 * 1024 / (1e9 * freq_GHz))
                        op_cycles = max(MXU_cycles, mem_cycles)
                        agg_tile_shapes_candidates.append((num_ops, num_accesses, ts_bytes, op_cycles, MXU_cycles, mem_cycles, lhs_agg_f, rhs_agg_f, reduct_agg_f))
        num_ops_lower_bound -= 1
        assert num_ops_lower_bound >= 1, f"No valid tile shapes found (num_ops_lower_bound={num_ops_lower_bound})"
    assert len(agg_tile_shapes_candidates) > 0, f"No valid tile shapes found"
    # num_ops: larger the better; num_accesses: smaller the better; ts_bytes: smaller the better;
    # op_cycles: smaller the better; MXU_cycles: smaller the better; mem_cycles: smaller the better;
    # if MXU bound, choose the min mem_cycles in all best candidates; if mem bound, choose the min MXU_cycles in all best candidates;
    # priority in ascending order (the last candidate is the most preferred)
    agg_tile_shapes_candidates.sort(key=lambda x: (-x[3], -x[5] if x[4] > x[5] else -x[4], x[0], -x[1], -x[2]))
    # num_ops, num_accesses, ts_bytes, op_cycles, MXU_cycles, mem_cycles, lhs_agg_f, rhs_agg_f, reduct_agg_f = agg_tile_shapes_candidates[-1]
    return agg_tile_shapes_candidates[-1]


def get_best_tile_shapes_for_matmul_from_vmem_size(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
    update_tile_size_in_axes: bool = False,
) -> List[List[int]]:
    '''
    Find the best tile size to maximize data reuse (i.e., minimize number of HBM accesses) for a MatMul operator.

    For a BatchMatMul [b, m, k] * [b, k, n] -> [b, m, n], the number of HBM accesses is b * (mn)/MN * k(M+N).
    To minimize this, we need to find the best M and N that minimize (M+N)/MN = 1/M+1/N, while satisfying
    the vmem size constraint (i.e., all input/output tiles must fit in the vmem).

    Output Tile Size: MN
    Input M Tile Size: MK
    Input N Tile Size: KN
    '''
    logging.vlog(3, "Finding best tile shape for op: %s", I.name)

    nc_compute_ns = int(node_cost.stats.compute_time_ns)
    vmem_bytes = config.vmem_size_MB * 1024 * 1024
    lhs_dtype_bytes = util.get_size_bytes_from_dtype(I.input_axes[0][0].data_type)
    rhs_dtype_bytes = util.get_size_bytes_from_dtype(I.input_axes[1][0].data_type)
    output_dtype_bytes = util.get_size_bytes_from_dtype(I.output_axes[0].data_type)
    sa_dim = config.sa_dim

    lhs_axes = I.input_axes[0]
    rhs_axes = I.input_axes[1]
    output_axes = I.output_axes
    batch_axes, reduction_axes, lhs_non_reduct_axes, rhs_non_reduct_axes = separate_axes_by_type_for_matmul(lhs_axes, rhs_axes, output_axes)
    batch_axes_names = {ax.name for ax in batch_axes}

    batch_axes_sizes = [ax.size for ax in batch_axes]
    if len(batch_axes_sizes) == 0:
        batch_axes_agg_size = 1
    else:
        batch_axes_agg_size = int(np.prod(batch_axes_sizes))

    # pick a value for K that is preferrably a multiple of 128*2
    reduction_axes_sizes = [ax.size for ax in reduction_axes]
    reduction_agg_size = int(np.prod(reduction_axes_sizes))

    # Consider all possible M/N values that are factors of m/n, pick the one that minimizes 1/M+1/N.
    # Preferrably, M/N is a multiple of 128*2 to minimize MXU padding.
    lhs_non_reduct_axes_sizes = [ax.size for ax in lhs_non_reduct_axes]
    rhs_non_reduct_axes_sizes = [ax.size for ax in rhs_non_reduct_axes]
    lhs_agg_size = int(np.prod(lhs_non_reduct_axes_sizes))
    rhs_agg_size = int(np.prod(rhs_non_reduct_axes_sizes))

    # compute max SRAM size demand (the min SRAM size that maximizes data reuse)
    max_sram_size_output = output_dtype_bytes * lhs_agg_size * rhs_agg_size
    max_sram_size_lhs = lhs_dtype_bytes * lhs_agg_size * min(sa_dim * 8, reduction_agg_size)
    max_sram_size_rhs = rhs_dtype_bytes * min(sa_dim * 8, reduction_agg_size) * rhs_agg_size
    max_sram_size = max_sram_size_output + max_sram_size_lhs + max_sram_size_rhs
    assert isinstance(node_cost, Operator.EinsumOperator), \
        f"node_cost must be an EinsumOperator, got {type(node_cost)}"
    node_cost.stats.max_vmem_demand_bytes = max_sram_size

    # if the tensors are smaller than vmem size, then tile size is just the entire tensor
    lhs_size_bytes = lhs_dtype_bytes * lhs_agg_size * reduction_agg_size * batch_axes_agg_size
    rhs_size_bytes = rhs_dtype_bytes * reduction_agg_size * rhs_agg_size * batch_axes_agg_size
    output_size_bytes = output_dtype_bytes * lhs_agg_size * rhs_agg_size * batch_axes_agg_size
    tot_size_bytes = lhs_size_bytes + rhs_size_bytes + output_size_bytes
    if tot_size_bytes <= vmem_bytes:
        logging.vlog(
            3,
            "lhs_size_bytes=%d, rhs_size_bytes=%d, output_size_bytes=%d, tot_size_bytes=%d, vmem_bytes=%d",
            lhs_size_bytes, rhs_size_bytes, output_size_bytes, tot_size_bytes, vmem_bytes
        )
        return [
            [ax.size for ax in lhs_axes],
            [ax.size for ax in rhs_axes],
            [ax.size for ax in output_axes],
        ]

    (
        num_ops,
        num_accesses,
        ts_bytes,
        op_cycles,
        MXU_cycles,
        mem_cycles,
        lhs_agg_f,
        rhs_agg_f,
        reduct_agg_f,
    ) = find_best_tile_shape_for_matmul(
        lhs_dtype_bytes,
        rhs_dtype_bytes,
        output_dtype_bytes,
        sa_dim,
        lhs_agg_size,
        rhs_agg_size,
        reduction_agg_size,
        batch_axes_agg_size,
        vmem_bytes,
        nc_compute_ns,
        config.freq_GHz,
        config.hbm_bw_GBps,
    )

    logging.vlog(
        3,
        "%s: M=%d, N=%d, K=%d, num_ops=%d, bytes_accessed=%d, tile_size_bytes=%d"
        " -> op_cycles=%d, MXU_cycles=%d, mem_cycles=%d",
        I.result.name, lhs_agg_f, rhs_agg_f, reduct_agg_f, num_ops, num_accesses, ts_bytes,
        op_cycles, MXU_cycles, mem_cycles,
    )

    # factors that must be "split" across the corresponding groups of axes
    lhs_agg_split = int(np.ceil(lhs_agg_size / lhs_agg_f))
    rhs_agg_split = int(np.ceil(rhs_agg_size / rhs_agg_f))
    reduct_agg_split = int(np.ceil(reduction_agg_size / reduct_agg_f))

    ## convert aggregated axes back to individual axes
    ## split larger axes first, and then smaller ones
    lhs_tile_shape = [-1 for ax in lhs_axes]
    rhs_tile_shape = [-1 for ax in rhs_axes]
    output_tile_shape = [-1 for ax in output_axes]

    # assume batch axes are fully tiled, because they don't affect data reuse
    for ax in lhs_axes:
        if ax.name in batch_axes_names:
            lhs_tile_shape[ax.index] = 1
    for ax in rhs_axes:
        if ax.name in batch_axes_names:
            rhs_tile_shape[ax.index] = 1
    for ax in output_axes:
        if ax.name in batch_axes_names:
            output_tile_shape[ax.index] = 1

    def split_axes(
        axes_to_split: List[hlo_struct.HLOAxis],
        tile_shape: List[int],
        split: int,
        other_axes_to_update: List[hlo_struct.HLOAxis],
        other_tile_shape: List[int]
    ):
        axes_to_split.sort(key=lambda ax: ax.size, reverse=True)
        for ax in axes_to_split:
            gcd_split = int(np.gcd(ax.size, split))
            tile_shape[ax.index] = ax.size // gcd_split
            # find the corresponding ax.index in other_axes_to_update
            for ax2 in other_axes_to_update:
                if ax2.name == ax.name:
                    other_tile_shape[ax2.index] = tile_shape[ax.index]
                    break
            split //= gcd_split
        assert split == 1, f"split not expected: {split}"

    # lhs and output non reduction axes
    split_axes(lhs_non_reduct_axes, lhs_tile_shape, lhs_agg_split, output_axes, output_tile_shape)
    # rhs and output non reduction axes
    split_axes(rhs_non_reduct_axes, rhs_tile_shape, rhs_agg_split, output_axes, output_tile_shape)
    # lhs and rhs reduction axes
    split_axes(reduction_axes, lhs_tile_shape, reduct_agg_split, rhs_axes, rhs_tile_shape)

    if update_tile_size_in_axes:
        for ax in lhs_axes:
            ax.tile_size = lhs_tile_shape[ax.index]
        for ax in rhs_axes:
            ax.tile_size = rhs_tile_shape[ax.index]
        for ax in output_axes:
            ax.tile_size = output_tile_shape[ax.index]

    logging.vlog(
        3,
        "%s: lhs_tile_shape=%s, rhs_tile_shape=%s, output_tile_shape=%s",
        I.result.name, lhs_tile_shape, rhs_tile_shape, output_tile_shape,
    )

    return [lhs_tile_shape, rhs_tile_shape, output_tile_shape]


def compute_bytes_accessed_from_vmem_size_for_matmul(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
) -> int:
    lhs, rhs, output = get_best_tile_shapes_for_matmul_from_vmem_size(
        I, node_cost, config, update_tile_size_in_axes=True
    )
    assert isinstance(node_cost, Operator.EinsumOperator), \
        f"node_cost must be an EinsumOperator, got {type(node_cost)}"
    node_cost.stats.tile_shapes_str = str([lhs, rhs, output])
    ba = compute_bytes_accessed_from_tile_shape_matmul(
        I,
        node_cost,
        [
            lhs,
            rhs,
            output,
        ],
    )

    return ba


def get_best_tile_config_for_flash_attention_from_vmem_size(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
    update_tile_size_in_axes: bool = False,
) -> Tuple[int, int]:
    '''
    Returns (Bc, Br).
    Data reuse in FlashAttention is only affected by Bc.
    Assumes Bc and Br are at least @config.sa_dim to utilize the MXUs.
    This function simply picks the largest Bc that fits in the vmem size.
    The SRAM consumption of tile config (Bc, Br) = 2 * Bc * d_head + Br * d_head + 2 * Bc * Br <= vmem_size.
    This gives Bc <= (vmem_size - Br * d_head) / (2 * (Br + d_head)).
    '''
    if update_tile_size_in_axes:
        raise NotImplementedError("update_tile_size_in_axes is not implemented right now.")

    vmem_MB = config.vmem_size_MB
    sa_dim = config.sa_dim
    dtype_bytes = util.get_size_bytes_from_dtype(I.input_axes[0][0].data_type)
    batch, q_seqlen, kv_seqlen, num_heads, d_head = (
        get_axes_size_for_flash_attention(I, node_cost)
    )

    def SRAM_req_bytes(Bc: int, Br: int) -> int:
        return (2 * Bc * d_head + Br * d_head + 2 * Bc * Br) * dtype_bytes

    # FlashAttention requires minimum SRAM size that fits in the smallest tile.
    # Assumes Bc and Br is at least sa_dim to utilize the MXUs.
    min_SRAM_req = SRAM_req_bytes(sa_dim, sa_dim)
    assert min_SRAM_req <= vmem_MB * 1024 * 1024, f"SRAM is too small to implement FlashAttention: (requires at least {min_SRAM_req / 1024 / 1024} MB)"

    # compute max vmem demand,
    # i.e., min tile size that maximizes data reuse
    # max tile size is when Bc is maximized (i.e., == kv_seqlen).
    max_vmem_demand_bytes = SRAM_req_bytes(
        Bc=max(sa_dim, kv_seqlen),
        Br=sa_dim,
    )
    assert isinstance(node_cost, Operator.FlashAttentionOperator), \
        f"node_cost must be a FlashAttentionOperator, got {type(node_cost)}"
    node_cost.stats.max_vmem_demand_bytes = max_vmem_demand_bytes

    vmem_size = int(vmem_MB * 1024 * 1024 / dtype_bytes)
    Br = sa_dim
    Bc = max(
        sa_dim,
        min(
            kv_seqlen,
            (vmem_size - Br * d_head) // (2 * (Br + d_head)),
        )
    )

    return Bc, Br


def compute_bytes_accessed_from_tile_config_flash_attention(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    Bc: int,
    Br: int,  # unused
) -> int:
    '''
    @param Bc: tile size of kv_seqlen dimension (for the K and V matrices).
    @param Br: tile size of q_seqlen dimension (for the Q matrix).
    For the complete description of tile sizes @Br and @Bc, see https://arxiv.org/pdf/2205.14135.
    '''
    dtype_bytes = util.get_size_bytes_from_dtype(I.input_axes[0][0].data_type)
    batch, q_seqlen, kv_seqlen, num_heads, d_head = (
        get_axes_size_for_flash_attention(I, node_cost)
    )

    # 1. Calculate the traffic for each sequence and each attention head.
    num_Bc_tiles = int(np.ceil(kv_seqlen / Bc))
    KV_traffic = 2 * kv_seqlen * d_head
    Q_and_output_traffic = 2 * num_Bc_tiles * q_seqlen * d_head

    # 2. The total mem traffic is the sum for the entire batch and all heads.
    total_traffic = batch * num_heads * (KV_traffic + Q_and_output_traffic)

    return total_traffic * dtype_bytes


def compute_bytes_accessed_from_vmem_size_for_flash_attention(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
) -> int:
    '''
    Side effect: Set node_cost["tile_shapes"] to be [Bc, Br].
    '''
    dtype_bytes = util.get_size_bytes_from_dtype(I.input_axes[0][0].data_type)
    batch, q_seqlen, kv_seqlen, num_heads, d_head = (
        get_axes_size_for_flash_attention(I, node_cost)
    )

    # If total mem footprint (Q,K,V,S,O) of this op is smaller than SRAM size,
    # then do not perform tiling.
    Q_size = batch * q_seqlen * num_heads * d_head
    KV_size = 2 * batch * kv_seqlen * num_heads * d_head
    S_size = batch * kv_seqlen * q_seqlen * num_heads * d_head
    O_size = batch * q_seqlen * num_heads * d_head
    tot_size = (Q_size + KV_size + S_size + O_size) * dtype_bytes
    assert isinstance(node_cost, Operator.FlashAttentionOperator), \
        f"node_cost must be a FlashAttentionOperator, got {type(node_cost)}"
    if tot_size <= config.vmem_size_MB * 1024 * 1024:
        node_cost.stats.tile_shapes_str = str([kv_seqlen, q_seqlen])
        node_cost.stats.max_vmem_demand_bytes = tot_size
        return tot_size

    Bc, Br = get_best_tile_config_for_flash_attention_from_vmem_size(
        I, node_cost, config
    )
    node_cost.stats.tile_shapes_str = str([Bc, Br])
    ba = compute_bytes_accessed_from_tile_config_flash_attention(
        I, node_cost, Bc, Br,
    )

    return ba


def separate_axes_by_type_for_conv2d(
    input_axes: Sequence[hlo_struct.HLOAxis],
    kernel_axes: Sequence[hlo_struct.HLOAxis],
    output_axes: Sequence[hlo_struct.HLOAxis],
) -> Sequence[List[hlo_struct.HLOAxis]]:
    '''
    Returns (b_axes, ic_axes, oc_axes, kh_axes, kw_axes, oh_axes, ow_axes, ih_axes, iw_axes).
    Each [X]_axes is a List of HLOAxis objects.
    Each [X]_axes should only contain one axis, prioritizing the axis object
    from input_axes, output_axes, kernel_axes (descending order).
    '''
    b_axes = [ax for ax in input_axes if ax.name == "batch"]
    assert len(b_axes) == 1, f"Batch axes not found: {b_axes} in {input_axes}"
    ic_axes = [ax for ax in input_axes if ax.name == "input_channel"]
    assert len(ic_axes) == 1, f"Input channel axes not found: {ic_axes} in {input_axes}"
    oc_axes = [ax for ax in output_axes if ax.name == "output_channel"]
    assert len(oc_axes) == 1, f"Output channel axes not found: {oc_axes} in {output_axes}"
    kh_axes = [ax for ax in kernel_axes if ax.name == "spatial0"]
    assert len(kh_axes) == 1, f"Kernel spatial0 axes not found: {kh_axes} in {kernel_axes}"
    kw_axes = [ax for ax in kernel_axes if ax.name == "spatial1"]
    assert len(kw_axes) == 1, f"Kernel spatial1 axes not found: {kw_axes} in {kernel_axes}"
    oh_axes = [ax for ax in output_axes if ax.name == "spatial0"]
    assert len(oh_axes) == 1, f"Output spatial0 axes not found: {oh_axes} in {output_axes}"
    ow_axes = [ax for ax in output_axes if ax.name == "spatial1"]
    assert len(ow_axes) == 1, f"Output spatial1 axes not found: {ow_axes} in {output_axes}"
    ih_axes = [ax for ax in input_axes if ax.name == "spatial0"]
    assert len(ih_axes) == 1, f"Input spatial0 axes not found: {ih_axes} in {input_axes}"
    iw_axes = [ax for ax in input_axes if ax.name == "spatial1"]
    assert len(iw_axes) == 1, f"Input spatial1 axes not found: {iw_axes} in {input_axes}"

    return b_axes, ic_axes, oc_axes, kh_axes, kw_axes, oh_axes, ow_axes, ih_axes, iw_axes


def compute_bytes_accessed_from_tile_shape_conv2d(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    tile_shapes: Sequence[Sequence[int]],
) -> int:
    '''
    Compute HBM bytes accessed for Conv2d given input/output tensor and tile shapes.
    @param node_cost: the operator containing the node cost information
    @param tile_shapes: a list of list of integers representing the tile shapes for input, kernel, and output
    @return: the number of bytes accessed from/to HBM
    '''
    assert len(tile_shapes) == 3, f"tile_shape not expected: {tile_shapes}"
    input_tile_shape = tile_shapes[0]
    kernel_tile_shape = tile_shapes[1]
    output_tile_shape = tile_shapes[2]

    b_axes, ic_axes, oc_axes, kh_axes, kw_axes, oh_axes, ow_axes, ih_axes, iw_axes = (
        separate_axes_by_type_for_conv2d(I.input_axes[0], I.input_axes[1], I.output_axes)
    )
    b_ax, ic_ax, oc_ax, kh_ax, kw_ax, oh_ax, ow_ax, ih_ax, iw_ax = (
        b_axes[0], ic_axes[0], oc_axes[0], kh_axes[0], kw_axes[0], oh_axes[0], ow_axes[0], ih_axes[0], iw_axes[0]
    )
    b, ic, oc, kh, kw, oh, ow, ih, iw = (
        b_ax.size, ic_ax.size, oc_ax.size, kh_ax.size, kw_ax.size, oh_ax.size, ow_ax.size, ih_ax.size, iw_ax.size
    )

    B = input_tile_shape[b_ax.index]
    IC = input_tile_shape[ic_ax.index]
    OC = output_tile_shape[oc_ax.index]
    KH = kernel_tile_shape[kh_ax.index]
    KW = kernel_tile_shape[kw_ax.index]
    OH = output_tile_shape[oh_ax.index]
    OW = output_tile_shape[ow_ax.index]
    IH = input_tile_shape[ih_ax.index]
    IW = input_tile_shape[iw_ax.index]

    input_dtype_bytes = util.get_size_bytes_from_dtype(I.input_axes[0][0].data_type)
    kernel_dtype_bytes = util.get_size_bytes_from_dtype(I.input_axes[1][0].data_type)
    output_dtype_bytes = util.get_size_bytes_from_dtype(I.output_axes[0].data_type)

    num_tiles = int(np.ceil((b * oc * oh * ow) / (B * OC * OH * OW)))
    assert isinstance(node_cost, Operator.Conv2DOperator), \
        f"node_cost must be a Conv2DOperator, got {type(node_cost)}"
    node_cost.stats.num_tiles = num_tiles

    if "stride" in I.convolution_window:
        stride_ih = I.convolution_window["stride"][0]
        stride_iw = I.convolution_window["stride"][1]
    else:
        stride_ih = 1
        stride_iw = 1
    if stride_ih >= kh:
        real_IH = OH * kh
    else:
        real_IH = OH * stride_ih + kh - 1
    if stride_iw >= kw:
        real_IW = OW * kw
    else:
        real_IW = OW * stride_iw + kw - 1

    if num_tiles == 1:
        num_inputs = B * ic * ih * iw  # not tiled since the op fits in vmem
    else:
        num_inputs = B * ic * real_IH * real_IW
    num_kernel = ic * OC * kh * kw
    num_outputs = B * OC * OH * OW
    bytes_accessed = num_tiles * (
        num_inputs * input_dtype_bytes
        + num_outputs * output_dtype_bytes
        + num_kernel * kernel_dtype_bytes
    )

    return bytes_accessed


def get_best_tile_shapes_for_conv2d_from_vmem_size(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
    update_tile_size_in_axes: bool = False,
) -> List[List[int]]:
    '''
    Find the best tile size to maximize data reuse (i.e., minimize number of HBM accesses) for a Conv2d operator.

    For an unrolled Conv2d [b, ic, ih, iw], [ic, oc, kh, kw] -> [b, oc, oh, ow], we assume:
        1. kh and kw are not tiled since they are usually very small (e.g., 3x3 or 5x5).
        2. For now, we don't consider dilated conv or specially written depth-wise conv.
        3. We ignore output padding for now.
        4. We assume the entire padded input tile is loaded into vmem. For example, to compute an output tile
           of size [5, 5], if the kernel is [3, 3], then the input tile size is [7, 7] ([oh+kh-1, ow+kw-1]).
        5. The axes are unrolled into MatMul as follows:
                batch axes: []
                reduction axes: [ic, kh, kw]
                non-reduction axes: [b, oc, oh, ow], where m = oc, n = b*oh*ow, k = ic*kh*kw.
           Unrolled axes are used to compute num_MXU_ops.

    For more about unrolling Conv2d, see: https://lumetta.web.engr.illinois.edu/408-S19/slide-copies/ece408-lecture12-S19-ZJUI.pdf

    With tile size [B, IC, OH+kh-1, OW+kw-1], [IC, OC, kh, kw] -> [B, OC, OH, OW], the number of HBM accesses per output tile is
        HBM_{tile} = B * ic * real_IH * real_IW + ic * OC * kh * kw + B * OC * OH * OW.
    Here, real_IH and real_IW accounts for strided convolution:
        real_IH = (OH * kh) if (stride_ih >= kh) else (OH * stride_ih + kh - 1).
        real_IW = (OW * kw) if (stride_iw >= kw) else (OW * stride_iw + kw - 1).
    There are N_{tile} = (b * oc * oh * ow) / (B * OC * OH * OW) output tiles.
    The total number of HBM accesses is HBM_{total} = N_{tile} * HBM_{tile}.
    '''
    logging.vlog(3, "Finding best tile shape for op: %s", I.name)

    nc_compute_ns = int(node_cost.stats.compute_time_ns)
    vmem_bytes = config.vmem_size_MB * 1024 * 1024
    input_dtype_bytes = util.get_size_bytes_from_dtype(I.input_axes[0][0].data_type)
    kernel_dtype_bytes = util.get_size_bytes_from_dtype(I.input_axes[1][0].data_type)
    output_dtype_bytes = util.get_size_bytes_from_dtype(I.output_axes[0].data_type)
    sa_dim = config.sa_dim

    b_axes, ic_axes, oc_axes, kh_axes, kw_axes, oh_axes, ow_axes, ih_axes, iw_axes = (
        separate_axes_by_type_for_conv2d(I.input_axes[0], I.input_axes[1], I.output_axes)
    )
    b_ax, ic_ax, oc_ax, kh_ax, kw_ax, oh_ax, ow_ax, ih_ax, iw_ax = (
        b_axes[0], ic_axes[0], oc_axes[0], kh_axes[0], kw_axes[0], oh_axes[0], ow_axes[0], ih_axes[0], iw_axes[0]
    )
    b = b_ax.size
    ic = ic_ax.size
    oc = oc_ax.size
    kh = kh_ax.size
    kw = kw_ax.size
    oh = oh_ax.size
    ow = ow_ax.size
    ih = ih_ax.size
    iw = iw_ax.size

    if "stride" in I.convolution_window:
        stride_ih = I.convolution_window["stride"][0]
        stride_iw = I.convolution_window["stride"][1]
    else:
        stride_ih = 1
        stride_iw = 1

    def tile_size_bytes(B: int, IC: int, OC: int, OH: int, OW: int) -> int:
        if stride_ih >= kh:
            real_IH = OH * kh
        else:
            real_IH = OH * stride_ih + kh - 1
        if stride_iw >= kw:
            real_IW = OW * kw
        else:
            real_IW = OW * stride_iw + kw - 1
        input_bytes = B * IC * real_IH * real_IW * input_dtype_bytes + IC * OC * kh * kw * kernel_dtype_bytes
        output_bytes = B * OC * OH * OW * output_dtype_bytes
        ts_bytes = input_bytes + output_bytes
        logging.vlog(3, "B=%d, IC=%d, OC=%d, OH=%d, OW=%d, kh=%s, kw=%d -> tile_size_bytes=%d", B, IC, OC, OH, OW, kh, kw, ts_bytes)
        return ts_bytes

    # compute max tile size demand (min tile size that maximizes data reuse)
    # After unrolling to matmul, this is equivalent to maximizing b, oc, oh, ow, and minimizing ic, ih, iw.
    max_tile_size = tile_size_bytes(b, min(sa_dim * 8, ic), oc, oh, ow)
    assert isinstance(node_cost, Operator.Conv2DOperator), \
        f"node_cost must be a Conv2DOperator, got {type(node_cost)}"
    node_cost.stats.max_vmem_demand_bytes = max_tile_size

    # if the tensors are smaller than vmem size, then tile size is just the entire tensor
    input_size_bytes = b * ic * ih * iw * input_dtype_bytes
    kernel_size_bytes = ic * oc * kh * kw * kernel_dtype_bytes
    output_size_bytes = b * oc * oh * ow * output_dtype_bytes
    tot_size_bytes = input_size_bytes + kernel_size_bytes + output_size_bytes
    if tot_size_bytes <= vmem_bytes:
        input_tile = [ax.size for ax in I.input_axes[0]]
        kernel_tile = [ax.size for ax in I.input_axes[1]]
        output_tile = [ax.size for ax in I.output_axes]
        logging.vlog(
            3,
            "input_size_bytes=%d, kernel_size_bytes=%d, output_size_bytes=%d, tot_size_bytes=%d, vmem_bytes=%d"
            " -> input_tile=%s, kernel_tile=%s, output_tile=%s",
            input_size_bytes, kernel_size_bytes, output_size_bytes, tot_size_bytes, vmem_bytes,
            input_tile, kernel_tile, output_tile,
        )
        return [input_tile, kernel_tile, output_tile]

    def num_MXU_ops(B: int, IC: int, OC: int, OH: int, OW: int) -> int:
        '''
        Compute number of MXU push/matmul/pop operations for a given tile size.
        '''
        # unrolled Conv2d into MatMul
        M = OC
        K = IC * kh * kw
        N = B * OH * OW
        num_ops = max(
            3,  # at least one push+matmul+pop
            # MN/sa_dim^2 output tiles, each of which has K/sa_dim pipelined push+matmul+pop.
            # The ceilings here account for MXU padding.
            int(np.ceil(M / sa_dim)) * int(np.ceil(N / sa_dim)) * int(np.ceil(K / sa_dim)),
        )
        logging.vlog(
            3,
            "B=%d, IC=%d, OC=%d, OH=%d, OW=%d, kh=%s, kw=%d"
            " -> M=%d, K=%d, N=%d"
            " -> num_MXU_ops=%d",
            M, K, N, B, IC, OC, OH, OW, kh, kw, num_ops
        )
        return num_ops

    def HBM_bytes_accessed(B: int, IC: int, OC: int, OH: int, OW: int) -> int:
        '''
        Compute the total HBM bytes accessed for a given tile size.
        '''
        if stride_ih >= kh:
            real_IH = OH * kh
        else:
            real_IH = OH * stride_ih + kh - 1
        if stride_iw >= kw:
            real_IW = OW * kw
        else:
            real_IW = OW * stride_iw + kw - 1
        num_inputs = B * ic * real_IH * real_IW * input_dtype_bytes
        num_kernel = ic * OC * kh * kw * kernel_dtype_bytes
        num_outputs = B * OC * OH * OW * output_dtype_bytes
        num_tiles = int(np.ceil((b * oc * oh * ow) / (B * OC * OH * OW)))
        logging.vlog(
            3,
            "B=%d, IC=%d, OC=%d, OH=%d, OW=%d, kh=%s, kw=%d, stride=(%d,%d)"
            " -> num_inputs=%d, num_kernel=%d, num_outputs=%d, num_tiles=%d",
            B, IC, OC, OH, OW, kh, kw, stride_ih, stride_iw,
            num_inputs, num_kernel, num_outputs, num_tiles,
        )
        return num_tiles * (num_inputs + num_outputs + num_kernel)

    tile_shape_candidates = []
    b_factors = util.get_factors(b)
    ic_factors = util.get_factors(ic)
    oc_factors = util.get_factors(oc)
    oh_factors = util.get_factors(oh)
    ow_factors = util.get_factors(ow)
    for B in b_factors:
        for OH in oh_factors:
            for OW in ow_factors:
                for OC in oc_factors:
                    for IC in ic_factors:
                        num_ops = num_MXU_ops(B, IC, OC, OH, OW)
                        ts_bytes = tile_size_bytes(B, IC, OC, OH, OW)
                        num_accesses = HBM_bytes_accessed(B, IC, OC, OH, OW)
                        if num_ops > 10 and ts_bytes <= vmem_bytes:
                            # All push/matmul/pop ops are pipelined. Each op is ~100 cycles.
                            num_tiles = int(np.ceil((b * oc * oh * ow) / (B * OC * OH * OW)))
                            MXU_cycles = (
                                nc_compute_ns * config.freq_GHz
                                if nc_compute_ns > 0
                                else 100 * (num_ops + 2) * num_tiles
                            )
                            mem_cycles = num_accesses / (config.hbm_bw_GBps * 1024 * 1024 * 1024 / (1e9 * config.freq_GHz))
                            op_cycles = max(MXU_cycles, mem_cycles)
                            tile_shape_candidates.append((num_ops, num_accesses, ts_bytes, op_cycles, MXU_cycles, mem_cycles, B, IC, OC, OH, OW))
    assert len(tile_shape_candidates) > 0, "No valid tile shapes found"
    # num_ops: larger the better; num_accesses: smaller the better; ts_bytes: smaller the better;
    # op_cycles: smaller the better; MXU_cycles: smaller the better; mem_cycles: smaller the better;
    # if MXU bound, choose the min mem_cycles in all best candidates; if mem bound, choose the min MXU_cycles in all best candidates;
    # priority in ascending order (the last candidate is the most preferred)
    tile_shape_candidates.sort(key=lambda x: (-x[3], -x[5] if x[4] > x[5] else -x[4], x[0], -x[1], -x[2]))

    num_ops, bytes_accessed, tile_bytes, op_cycles, MXU_cycles, mem_cycles, B, IC, OC, OH, OW = tile_shape_candidates[-1]

    logging.vlog(
        3,
        "%s: B=%d, IC=%d, OC=%d, OH=%d, OW=%d, num_ops=%d, bytes_accessed=%d, tile_size_bytes=%d"
        " -> op_cycles=%d, MXU_cycles=%d, mem_cycles=%d",
        I.result.name, B, IC, OC, OH, OW, num_ops, bytes_accessed, tile_bytes,
        op_cycles, MXU_cycles, mem_cycles,
    )

    # convert candidate tile shape into tensor tile shapes
    input_tile_shape = [ax.size for ax in I.input_axes[0]]
    kernel_tile_shape = [ax.size for ax in I.input_axes[1]]
    output_tile_shape = [ax.size for ax in I.output_axes]

    def update_tile_shape(
        tile_shape: list[int],
        axes: list[hlo_struct.HLOAxis],
        dim_name: str,
        tile_size: int,
    ):
        for ax in axes:
            if ax.name == dim_name:
                tile_shape[ax.index] = tile_size
                if update_tile_size_in_axes:
                    ax.tile_size = tile_size

    # B
    update_tile_shape(input_tile_shape, I.input_axes[0], "batch", B)
    update_tile_shape(output_tile_shape, I.output_axes, "batch", B)
    # IC
    update_tile_shape(input_tile_shape, I.input_axes[0], "input_channel", IC)
    update_tile_shape(kernel_tile_shape, I.input_axes[1], "input_channel", IC)
    # OC
    update_tile_shape(output_tile_shape, I.output_axes, "output_channel", OC)
    update_tile_shape(kernel_tile_shape, I.input_axes[1], "output_channel", OC)
    # IH
    update_tile_shape(
        input_tile_shape,
        I.input_axes[0],
        "spatial0",
        OH * kh if stride_ih >= kh else OH * stride_ih + kh - 1
    )
    # IW
    update_tile_shape(
        input_tile_shape,
        I.input_axes[0],
        "spatial1",
        OW * kw if stride_iw >= kw else OW * stride_iw + kw - 1
    )
    # OH
    update_tile_shape(output_tile_shape, I.output_axes, "spatial0", OH)
    # OW
    update_tile_shape(output_tile_shape, I.output_axes, "spatial1", OW)

    logging.vlog(
        3,
        "%s: input_tile_shape=%s, kernel_tile_shape=%s, output_tile_shape=%s",
        I.result.name, input_tile_shape, kernel_tile_shape, output_tile_shape,
    )

    return [input_tile_shape, kernel_tile_shape, output_tile_shape]


def compute_bytes_accessed_from_vmem_size_for_conv2d(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
) -> int:
    input_feature, kernel, output_feature = get_best_tile_shapes_for_conv2d_from_vmem_size(
        I, node_cost, config, update_tile_size_in_axes=True
    )
    assert isinstance(node_cost, Operator.Conv2DOperator), \
        f"node_cost must be a Conv2DOperator, got {type(node_cost)}"
    node_cost.stats.tile_shapes_str = str([input_feature, kernel, output_feature])
    ba = compute_bytes_accessed_from_tile_shape_conv2d(
        I,
        node_cost,
        [
            input_feature,
            kernel,
            output_feature,
        ],
    )

    return ba


def compute_bytes_accessed_from_tensor_sizes(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
) -> int:
    '''
    Return the sum of input and output tensor sizes in bytes.
    # Added consideration for num elements per byte in bytes_accessed equation.
    '''
    input_shapes = [[ax.size for ax in axes] for axes in I.input_axes]
    output_shape = [ax.size for ax in I.output_axes]
    input_elem_bytes = [util.get_size_bytes_from_dtype(axes[0].data_type) for axes in I.input_axes]
    output_elem_bytes = util.get_size_bytes_from_dtype(I.output_axes[0].data_type)
    bytes_accessed = (
        # inputs
        sum(int(bytes_per_elem * np.prod(tensor)) for tensor, bytes_per_elem in zip(input_shapes, input_elem_bytes))
        # outputs
        + int(np.prod(output_shape) * output_elem_bytes)
    )
    return bytes_accessed


def compute_bytes_accessed_from_vmem_size(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
) -> int:
    node_cost.stats.parsed_op_type = I.metadata["op_type"]
    if I.isConvolution():
        if I.metadata["op_type"] == "Conv2D":
            return compute_bytes_accessed_from_vmem_size_for_conv2d(I, node_cost, config)
        elif I.metadata["op_type"] in EINSUM_OP_TYPES:
            return compute_bytes_accessed_from_vmem_size_for_matmul(I, node_cost, config)
        elif I.metadata["op_type"] == "FlashAttention":
            return compute_bytes_accessed_from_vmem_size_for_flash_attention(I, node_cost, config)
        # NOTE: add other MXU ops here
        else:
            raise ValueError(f"Op type not supported: {I.metadata['op_type']}")
    else:  # assume not affected by vmem size
        orig_bytes_accessed = int(node_cost.stats.memory_traffic_bytes)
        if orig_bytes_accessed > 0:
            # If the bytes accessed is already computed, return it directly.
            return orig_bytes_accessed
        else:
            # compute from input/output tensor sizes
            return compute_bytes_accessed_from_tensor_sizes(I, node_cost)


def update_node_cost_memory_traffic_and_time(
    nc: Operator.Operator,
    new_bytes_accessed: int,
    hbm_bw_GBps: float,
):
    '''
    Update the memory traffic and time in the node cost based on @new_bytes_accessed.
    '''
    nc.stats.memory_traffic_bytes = new_bytes_accessed
    nc.stats.memory_time_ns = int(np.ceil(new_bytes_accessed / (hbm_bw_GBps * 1024 * 1024 * 1024) * 1e9))
    nc.stats.execution_time_ns = max(
        int(nc.stats.execution_time_ns), int(nc.stats.memory_time_ns)
    )

    # update Bound By field
    if nc.stats.memory_time_ns == nc.stats.execution_time_ns:
        nc.stats.bounded_by = "Memory"
    else:
        nc.stats.bounded_by = "Compute"


def get_compute_memory_boundedness(
    node_costs: list[Operator.Operator | dict[str, Any]] | Operator.Operator | dict[str, Any]
) -> Tuple[float, float]:
    """Returns (compute %, pure mem bound %)"""
    if isinstance(node_costs, (Operator.Operator, dict)):
        node_costs = [node_costs]
    if isinstance(node_costs[0], dict):
        # If node_costs is a list of dicts, convert them to Operator objects
        node_costs = [Operator.from_csv_dict(nc) for nc in node_costs]  # type: ignore
    assert all(isinstance(nc, Operator.Operator) for nc in node_costs), \
        "All node_costs must be Operator objects."

    total_execution_time_ns = util.get_total_execution_time_ns_from_ops(node_costs)  # type: ignore
    pure_mem_time_ns = sum([
        int(nc.stats.execution_time_ns) - int(nc.stats.compute_time_ns)  # type: ignore
        for nc in node_costs
        if nc.stats.bounded_by == "Memory"  # type: ignore
    ])

    compute_time = total_execution_time_ns - pure_mem_time_ns
    return compute_time / total_execution_time_ns, pure_mem_time_ns / total_execution_time_ns


def get_compute_memory_boundedness_for_node_cost(
    node_cost: Operator.Operator | dict[str, Any],
) -> Tuple[float, float]:
    '''Returns (compute %, pure mem bound %)'''
    return get_compute_memory_boundedness([node_cost])


def compute_node_cost_mxu_time_from_num_ops(
    num_ops: int,
    config: ChipConfig,
) -> int:
    freq_GHz = config.freq_GHz
    sa_dim = config.sa_dim
    pipeline_II = sa_dim / 128 * 100  # at least one push+matmul+pop latency
    total_mxu_time = max(
        int(np.ceil(pipeline_II / freq_GHz)),
        int(
            np.ceil(
                (num_ops + 2)  # the initiation interval of push+matmul+pop pipeline is 2
                * sa_dim / config.num_sa  # sa_dim cycles per sa_dim*sa_dim MatMul
                / freq_GHz # scale by frequency
            )
        ),
        1,
    )
    return total_mxu_time


def compute_node_cost_vpu_time_from_num_ops(
    num_ops: int,
    config: ChipConfig,
) -> int:
    # TODO: use VPU dim to compute VPU time
    total_vpu_time = int(
        np.ceil(num_ops / config.num_vu)  # 1 cycle per VPU op
        / config.freq_GHz  # scale by frequency
    )
    return max(1, total_vpu_time)


def compute_node_cost_compute_time_for_conv2d(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
) -> tuple[int, int]:
    # get dimensions for this conv2d
    b_axes, ic_axes, oc_axes, kh_axes, kw_axes, oh_axes, ow_axes, ih_axes, iw_axes = (
        separate_axes_by_type_for_conv2d(I.input_axes[0], I.input_axes[1], I.output_axes)
    )
    b_ax, ic_ax, oc_ax, kh_ax, kw_ax, oh_ax, ow_ax, ih_ax, iw_ax = (
        b_axes[0], ic_axes[0], oc_axes[0], kh_axes[0], kw_axes[0], oh_axes[0], ow_axes[0], ih_axes[0], iw_axes[0]
    )
    b, ic, oc, kh, kw, oh, ow, ih, iw = (
        b_ax.size, ic_ax.size, oc_ax.size, kh_ax.size, kw_ax.size, oh_ax.size, ow_ax.size, ih_ax.size, iw_ax.size
    )
    sa_dim = config.sa_dim
    # unroll Conv2d into MatMul
    m = oc
    k = ic * kh * kw
    n = b * oh * ow
    num_mxu_ops = (
        int(np.ceil(m / sa_dim)) * int(np.ceil(n / sa_dim)) * int(np.ceil(k / sa_dim))
    )
    assert isinstance(node_cost, Operator.Conv2DOperator), \
        f"node_cost must be a Conv2DOperator, got {type(node_cost)}"
    node_cost.stats.einsum_B_size = b
    node_cost.stats.einsum_M_size = m
    node_cost.stats.einsum_N_size = n
    node_cost.stats.einsum_K_size = k
    node_cost.stats.num_sa_ops = num_mxu_ops
    vu_ops_multiplier = int(np.ceil(sa_dim / 128 * 8 * 2))
    num_vpu_ops = num_mxu_ops * vu_ops_multiplier

    total_mxu_time = compute_node_cost_mxu_time_from_num_ops(
        num_mxu_ops, config
    )
    total_vpu_time = compute_node_cost_vpu_time_from_num_ops(
        num_vpu_ops, config
    )

    return total_mxu_time, total_vpu_time


def compute_node_cost_compute_time_for_matmul(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
) -> tuple[int, int]:
    lhs_axes, rhs_axes = I.input_axes
    output_axes = I.output_axes
    batch_axes, reduction_axes, lhs_non_reduct_axes, rhs_non_reduct_axes = separate_axes_by_type_for_matmul(lhs_axes, rhs_axes, output_axes)
    b = int(np.prod([ax.size for ax in batch_axes]))
    m = int(np.prod([ax.size for ax in lhs_non_reduct_axes]))
    n = int(np.prod([ax.size for ax in rhs_non_reduct_axes]))
    k = int(np.prod([ax.size for ax in reduction_axes]))
    assert isinstance(node_cost, Operator.EinsumOperator), \
        f"node_cost must be a EinsumOperator, got {type(node_cost)}"
    node_cost.stats.einsum_B_size = b
    node_cost.stats.einsum_M_size = m
    node_cost.stats.einsum_N_size = n
    node_cost.stats.einsum_K_size = k
    sa_dim = config.sa_dim
    vu_ops_multiplier = int(np.ceil(sa_dim / 128 * 8 * 2))

    def try_run_matmul_on_mxu():
        '''Try execute this matmul op using MXU.'''
        # number of MXU MatMul ops (each op computes sa_dim*sa_dim, sa_dim*sa_dim -> sa_dim*sa_dim matmul)
        total_mxu_ops = b * (
            int(np.ceil(m / sa_dim)) * int(np.ceil(n / sa_dim)) * int(np.ceil(k / sa_dim))
        )

        node_cost.stats.num_sa_ops = total_mxu_ops
        # number of VPU accumulation ops (each op adds a 8*128*2 vector)
        total_vpu_ops = total_mxu_ops * vu_ops_multiplier

        total_mxu_time = compute_node_cost_mxu_time_from_num_ops(
            total_mxu_ops, config
        )
        total_vpu_time = compute_node_cost_vpu_time_from_num_ops(
            total_vpu_ops, config
        )
        return total_mxu_time, total_vpu_time

    def try_run_matmul_on_vpu():
        '''Try execute this matmul op using VPU.'''
        # number of VPU MatMul ops (each op computes 8*128, 8*128 -> 8*128 add/mul)
        total_vpu_ops = b * min(
            # try different tile mapping strategies for the 8*128 VPU,
            # use the one with minimum compute time
            int(np.ceil(m / 8)) * int(np.ceil(n / 128)) * int(np.ceil(k / 8)),
            int(np.ceil(m / 128)) * int(np.ceil(n / 8)) * int(np.ceil(k / 8)),
            int(np.ceil(m / 8)) * int(np.ceil(n / 8)) * int(np.ceil(k / 128)),
        ) * 2  # mul+add requires 2 instructions

        total_vpu_time = compute_node_cost_vpu_time_from_num_ops(
            total_vpu_ops, config
        )
        return 0, total_vpu_time

    total_mxu_time_using_mxu, total_vpu_time_using_mxu = try_run_matmul_on_mxu()
    total_mxu_time_using_vpu, total_vpu_time_using_vpu = try_run_matmul_on_vpu()
    total_exe_time_using_mxu = max(total_mxu_time_using_mxu, total_vpu_time_using_mxu)
    total_exe_time_using_vpu = max(total_mxu_time_using_vpu, total_vpu_time_using_vpu)
    if not config.use_vu_for_small_matmul or total_exe_time_using_mxu <= 4 * total_exe_time_using_vpu:
        return total_mxu_time_using_mxu, total_vpu_time_using_mxu
    else:
        return total_mxu_time_using_vpu, total_vpu_time_using_vpu


def get_axes_size_for_flash_attention(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,  # unused
) -> tuple[int, int, int, int, int]:
    '''
    Returns (batch, q_seqlen, kv_seqlen, num_heads, d_head).
    Assumes the inputs are in the order of Q, K, V in @param I.
    Assumes the dimensions of each input is [batch, seqlen, num_heads, d_head]
    '''
    Q_shape = [ax.size for ax in I.input_axes[0]]
    K_shape = [ax.size for ax in I.input_axes[1]]
    V_shape = [ax.size for ax in I.input_axes[2]]
    assert len(Q_shape) == 4 and len(K_shape) == 4 and len(V_shape) == 4, f"Invalid input shapes: {Q_shape}, {K_shape}, {V_shape}"
    assert Q_shape[0] == K_shape[0] == V_shape[0], f"Batch size mismatch: {Q_shape[0]}, {K_shape[0]}, {V_shape[0]}"
    batch = Q_shape[0]
    assert Q_shape[2] == K_shape[2] == V_shape[2], f"Num heads mismatch: {Q_shape[2]}, {K_shape[2]}, {V_shape[2]}"
    num_heads = Q_shape[2]
    assert Q_shape[3] == K_shape[3] == V_shape[3], f"Head dim mismatch: {Q_shape[3]}, {K_shape[3]}, {V_shape[3]}"
    d_head = Q_shape[3]
    assert K_shape[1] == V_shape[1], f"Seq len mismatch: {K_shape[1]}, {V_shape[1]}"
    q_seqlen = Q_shape[1]
    kv_seqlen = K_shape[1]

    return batch, q_seqlen, kv_seqlen, num_heads, d_head


def compute_node_cost_compute_time_for_flash_attention(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,  # unused
    config: ChipConfig,
) -> tuple[int, int]:
    '''
    Compute time should be the same as normal attention.
    MXU time = MatMul time.
    VPU time = MatMul aggregation time + softmax time of QK.
    Assumes the inputs are in the order of Q, K, V in @param I.
    Assumes the dimensions of each input is [batch, seqlen, num_heads, d_head]
    '''
    batch, q_seqlen, kv_seqlen, num_heads, d_head = get_axes_size_for_flash_attention(I, node_cost)
    sa_dim = config.sa_dim

    # Assuem each VU op is 128*8 elements,
    # compute how many VU ops are required to accumulate SA output based on sa_dim
    # TODO: add VU dims to config
    vu_op_multiplier = int(np.ceil(sa_dim / 128 * 8 * 2))

    mxu_time = 0
    vpu_time = 0

    # QK MatMul
    QK_num_mxu_ops = num_heads * batch * (
        int(np.ceil(kv_seqlen / sa_dim)) * int(np.ceil(q_seqlen / sa_dim)) * int(np.ceil(d_head / sa_dim))
    )
    mxu_time += compute_node_cost_mxu_time_from_num_ops(QK_num_mxu_ops, config)
    vpu_time += compute_node_cost_vpu_time_from_num_ops(QK_num_mxu_ops * vu_op_multiplier, config)

    # Softmax
    vu_softmax_time = compute_node_cost_vpu_time_from_num_ops(
        int(np.ceil(4 * batch * num_heads * q_seqlen * kv_seqlen / 128 / 8)),
        config,
    )
    vpu_time += vu_softmax_time

    # save vu softmax time in stats
    assert isinstance(node_cost, Operator.FlashAttentionOperator), \
        f"node_cost must be a FlashAttentionOperator, got {type(node_cost)}"
    node_cost.stats.vu_softmax_time_ns = vu_softmax_time

    # QK_V MatMul
    QK_V_num_mxu_ops = num_heads * batch * (
        int(np.ceil(kv_seqlen / sa_dim)) * int(np.ceil(q_seqlen / sa_dim)) * int(np.ceil(d_head / sa_dim))
    )
    mxu_time += compute_node_cost_mxu_time_from_num_ops(QK_V_num_mxu_ops, config)
    vpu_time += compute_node_cost_vpu_time_from_num_ops(QK_V_num_mxu_ops * vu_op_multiplier, config)

    node_cost.stats.einsum_B_size = batch * num_heads
    node_cost.stats.einsum_M_size = kv_seqlen
    node_cost.stats.einsum_N_size = q_seqlen
    node_cost.stats.einsum_K_size = d_head
    node_cost.stats.num_sa_ops = QK_num_mxu_ops + QK_V_num_mxu_ops

    return mxu_time, vpu_time


def compute_node_cost_compute_time_for_collective(
    op_name: str,
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
) -> tuple[int, int]:
    total_mxu_time = 0          # No matrix ops
    num_ops = node_cost.stats.flop_count
    # Each VU computes 8 * 128 elements per cycle
    vu_speed = config.num_vu * 8 * 128  # TODO: add VU dims to config
    total_vpu_time = int(np.ceil(num_ops / vu_speed))

    return total_mxu_time, total_vpu_time


def compute_node_cost_compute_time_for_elementwise(
    op_name: str,
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
) -> tuple[int, int]:
    total_mxu_time = 0
    input_size = int(np.prod([ax.size for ax in I.input_axes[0]]))

    # Each VU computes 8 * 128 elements per cycle
    vu_speed = config.num_vu * 8 * 128  # TODO: add VU dims to config
    if op_name in ["LayerNorm", "GroupNorm"]:
        total_vpu_time = int(np.ceil(8 * input_size / vu_speed))
    elif op_name == "RMSNorm":
        # RMSNorm is similar to LayerNorm, but more efficient
        total_vpu_time = int(np.ceil(6 * input_size / vu_speed))
    elif op_name == "Softmax":
        total_vpu_time = int(np.ceil(4 * input_size / vu_speed))
    elif op_name in ["Add", "Mul", "Abs", "Pointwise Mul."]:
        total_vpu_time = int(np.ceil(input_size / vu_speed))
    else:
        raise ValueError(f"Op name not supported: {op_name}")
    total_vpu_time = int(np.ceil(total_vpu_time / config.freq_GHz))  # convert to ns

    return total_mxu_time, total_vpu_time


def compute_node_cost_compute_time_for_up_down_sample(
    op_name: str,
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
) -> tuple[int, int]:
    total_mxu_time = 0
    input_size = int(np.prod([ax.size for ax in I.input_axes[0]]))
    output_size = int(np.prod([ax.size for ax in I.output_axes]))
    vu_speed = config.num_vu * 8 * 128  # TODO: add VU dims to config
    if op_name == "Upsample":
        # currently, assume nearest neighbor value, which is a memcpy operation
        total_vpu_time = 0
    elif op_name == "AvgPool2d":
        # Each input element needs to be aggregated to the corresponding output element (input_size flops)
        # Each aggregated value needs to be divided by the number of corresponding input elements (output_size flops)
        total_vpu_time = int(np.ceil((output_size + input_size) / vu_speed))
    else:
        raise ValueError(f"Op name not supported: {op_name}")
    total_vpu_time = int(np.ceil(total_vpu_time / config.freq_GHz))  # convert to ns

    return total_mxu_time, total_vpu_time


def compute_node_cost_compute_time(
    I: hlo_struct.HLOInstruction,
    node_cost: Operator.Operator,
    config: ChipConfig,
) -> tuple[int, int]:
    '''
    Return (MXU, VPU) Time
    based on the op type and tensor shapes in @I and @node_cost.
    Op type is given by the 'OpType' entry in the base_op dict.
    '''
    # print("compute time op:", I, I.metadata)
    op_type = I.metadata["op_type"]
    node_cost.stats.parsed_op_type = op_type
    if I.isConvolution():
        if op_type == "Conv2D":
            return compute_node_cost_compute_time_for_conv2d(I, node_cost, config)
        elif op_type in EINSUM_OP_TYPES:
            return compute_node_cost_compute_time_for_matmul(I, node_cost, config)
        elif op_type == "FlashAttention":
            return compute_node_cost_compute_time_for_flash_attention(I, node_cost, config)
        # NOTE: add other MXU ops here
        else:
            raise ValueError(f"Op type not supported: {op_type}")
    elif op_type in ["LayerNorm", "GroupNorm", "RMSNorm", "Softmax", "Add", "Mul", "Abs", "Pointwise Mul."]:
        return compute_node_cost_compute_time_for_elementwise(
            op_type, I, node_cost, config
        )
    elif op_type in ["Upsample", "AvgPool2d"]:
        return compute_node_cost_compute_time_for_up_down_sample(
            op_type, I, node_cost, config
        )
    elif op_type in ["AllGather", "InterChipCommInput", "InterChipCommOutput", "AllToAll", "KVSwapInput", "KVSwapOutput"]:
        # Communication Ops involve no compute.
        return (0, 0)

    elif op_type in ["ReduceScatter", "AllReduce"]:
        return compute_node_cost_compute_time_for_collective(
            op_type, I, node_cost, config
        )
    # NOTE: add other non-MXU ops here
    else:
        if node_cost.stats.sa_time_ns > 0 or node_cost.stats.vu_time_ns > 0:
            # If already computed elsewhere, return them directly.
            return node_cost.stats.sa_time_ns, node_cost.stats.vu_time_ns
        elif node_cost.stats.flop_count > 0 and not I.isConvolution():
            # If FLOP count is available for a VPU op, use it to compute VPU time.
            vu_speed = config.num_vu * 8 * 128  # TODO: add VU dims to config
            total_vpu_time = int(np.ceil(node_cost.stats.flop_count / vu_speed))
            total_vpu_time = int(np.ceil(total_vpu_time / config.freq_GHz))
            return 0, total_vpu_time
        else:
            raise NotImplementedError(f"Op type {op_type} not supported")


# def compute_op_compute_time(
#     I: hlo_struct.HLOInstruction,
#     op: Operator.Operator,
#     config: ChipConfig,
# ) -> tuple[int, int]:
#     '''
#     Return (MXU, VPU) Time
#     based on the op type and tensor shapes in @I and @op.
#     '''
#     nc = Operator.to_csv_dict(op)
#     return compute_node_cost_compute_time(I, nc, config)


def update_node_cost_compute_time(
    I: hlo_struct.HLOInstruction,
    node_cost: dict[str, Any],
    mxu_time: int,
    vpu_time: int,
):
    '''
    TODO: modify this function if we need it to support Operator.Operator in the future.
    Update the MXU, VPU, Compute, and total execution time in the node cost
    based on the given mxu and vpu time.
    '''
    # update mxu and vpu time
    node_cost["Total MXU time (ns)"] = mxu_time
    node_cost["Total VPU time (ns)"] = vpu_time

    # update total compute time
    node_cost["Compute Time (ns)"] = max(mxu_time, vpu_time)

    # update total execution time
    node_cost["Execution Time (ns)"] = max(
        int(node_cost["Execution Time (ns)"]), int(node_cost["Compute Time (ns)"])
    )
    node_cost["Total execution time (ns)"] = max(
        int(node_cost["Total execution time (ns)"]), int(node_cost["Compute Time (ns)"])
    )

    # update Bound By field
    if node_cost["Memory Time (ns)"] == node_cost["Execution Time (ns)"]:
        node_cost["Bound By"] = "External Memory"
    else:
        node_cost["Bound By"] = "Compute"
