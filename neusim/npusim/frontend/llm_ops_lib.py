from math import floor, ceil
from os import name
import re
from typing import Any, Sequence
from absl import flags, logging
import numpy as np
from copy import deepcopy

import neusim.npusim.backend.util as util
from neusim.configs.models.LLMConfig import DeepSeekConfig
from neusim.configs.chips.ChipConfig import ChipConfig

from neusim.configs.models.ModelConfig import ModelConfig
from neusim.npusim.frontend.Operator import Conv2DOperator, EinsumOperator, FlashAttentionOperator, Tensor, Axis, Operator, OpType, OpcodeType
from neusim.npusim.frontend.util import get_bisection_bw_per_chip_GBps


def format_conv2d_config_str(
    input_shape: Sequence[int],
    kernel_shape: Sequence[int],
    output_shape: Sequence[int],
    einsum_expr: str,
    window: str,
    dtype: str = "DT_FLOAT32",
    memory_placement: Sequence[int] = [0, 0, 0],
) -> str:
    '''
    @param input_a_shape: Shape of input.
    @param input_b_shape: Shape of kernel.
    @param output_shape: Shape of output.
    @param einsum_expr: Einsum dimension expression. E.g., "BLND;NDM->BLM".
    @param window: Window config. E.g., "{size=3x3 stride=2x2 pad=0_1x0_1}".
    @param dtype: Data type.
    @param memory_placement: Memory placement. 0 for HBM, 1 for SRAM.
    '''
    input_shape_str = "x".join([str(i) for i in input_shape])
    kernel_shape_str = "x".join([str(i) for i in kernel_shape])
    output_shape_str = "x".join([str(i) for i in output_shape])
    memory_placement_str = "_".join([str(i) for i in memory_placement])
    return f"Conv2D(a={input_shape_str},b={kernel_shape_str},c={output_shape_str},eq={einsum_expr},window={window},memory_placements={memory_placement_str},type={dtype})"


def format_einsim_config_str(
    input_a_shape: Sequence[int],
    input_b_shape: Sequence[int],
    einsum_expr: str,
    dtype: str = "DT_BFLOAT16",
    memory_placement: Sequence[int] = [0, 0, 0],
) -> str:
    '''
    @param input_a_shape: Shape of input A.
    @param input_b_shape: Shape of input B.
    @param einsum_expr: Einsum dimension expression. E.g., "BLND;NDM->BLM".
    @param dtype: Data type.
    @param memory_placement: Memory placement. 0 for HBM, 1 for SRAM.
    '''
    input_a_shape_str = "x".join([str(i) for i in input_a_shape])
    input_b_shape_str = "x".join([str(i) for i in input_b_shape])
    memory_placement_str = "_".join([str(i) for i in memory_placement])
    return f"XlaEinsum(a={input_a_shape_str},b={input_b_shape_str},eq={einsum_expr},memory_placements={memory_placement_str},type={dtype})"


def format_input_tensor_shapes(
    input_shapes: Sequence[Sequence[int] | int],
    dtype: str | Sequence[str] = "DT_BFLOAT16",
) -> str:
    if isinstance(input_shapes[0], int):
        input_shapes = [input_shapes]  # type: ignore
    if isinstance(dtype, str):
        dtype = [dtype] * len(input_shapes)
    return ",".join([f"{dt}:[{','.join([str(i) for i in shape])}]" for dt, shape in zip(dtype, input_shapes)]) # type: ignore


def format_output_tensor_shape(
    output_shape: Sequence[int],
    dtype: str = "DT_BFLOAT16",
) -> str:
    return f"[{dtype}:({','.join([str(i) for i in output_shape])})]"


def format_op_name(
    name: str,
    description: str,
) -> str:
    new_name = description + "_" + name
    new_name.replace(" ", "")
    regex = re.compile(r"[^a-zA-Z0-9]")
    new_name = regex.sub("", new_name)
    return new_name


def get_einsum_output_shape(
    input_a_shape: Sequence[int],
    input_b_shape: Sequence[int],
    einsum_expr: str,
) -> list[int]:
    einsum_expr_split = einsum_expr.split("->")
    input_a_axes, input_b_axes = einsum_expr_split[0].split(";")
    output_axes = einsum_expr_split[1]

    input_a_axes_set = {
        (axis, size)
        for axis, size in zip(input_a_axes, input_a_shape)
    }
    input_b_axes_set = {
        (axis, size)
        for axis, size in zip(input_b_axes, input_b_shape)
    }

    output_shape = []
    for axis in output_axes:
        found = False
        for ax, size in input_a_axes_set:
            if axis == ax:
                output_shape.append(size)
                found = True
                break
        if found:
            continue
        for ax, size in input_b_axes_set:
            if axis == ax:
                output_shape.append(size)
                found = True
                break
        if found:
            continue
        # Some axis are not found in inputs; there must be sth wrong.
        raise ValueError(f"Invalid einsum expression: {einsum_expr}")
    return output_shape


def get_flash_attention_output_shape(
    Q_shape: Sequence[int],
    K_shape: Sequence[int],
    V_shape: Sequence[int],
) -> list[int]:
    '''
    Get the output shape of a flash attention operation.
    Assumes the input shape dim order is [batch, seqlen, num_heads, d_model].
    @param Q_shape: Shape of Q matrix.
    @param K_shape: Shape of K matrix.
    @param V_shape: Shape of V matrix.
    '''
    assert Q_shape[0] == K_shape[0] == V_shape[0], "Batch size mismatch."
    batch = Q_shape[0]
    assert Q_shape[2] == K_shape[2] == V_shape[2], "num_heads mismatch."
    num_heads = Q_shape[2]
    assert Q_shape[3] == K_shape[3] == V_shape[3], "d_model mismatch."
    d_model = Q_shape[2]
    assert K_shape[1] == V_shape[1], "KV sequence length mismatch."
    q_seqlen = Q_shape[1]
    output_shape = [batch, q_seqlen, num_heads, d_model]
    return output_shape


def get_flops_flash_attention(
    Q_shape: Sequence[int],
    K_shape: Sequence[int],
    V_shape: Sequence[int],
) -> int:
    '''
    FLOP count of flash attention in forward pass is the same as normal attention.
    Assumes the input shape dim order is [batch, seqlen, num_heads, d_model].
    '''
    assert Q_shape[0] == K_shape[0] == V_shape[0], "Batch size mismatch."
    batch = Q_shape[0]
    assert Q_shape[2] == K_shape[2] == V_shape[2], "num_heads mismatch."
    num_heads = Q_shape[2]
    assert Q_shape[3] == K_shape[3] == V_shape[3], "d_model mismatch."
    d_model = Q_shape[2]
    assert K_shape[1] == V_shape[1], "KV sequence length mismatch."
    q_seqlen = Q_shape[1]
    kv_seqlen = K_shape[1]

    QK_flops = num_heads * batch * q_seqlen * d_model * kv_seqlen * 2
    softmax_QK_flops = get_flops_unary_op(
        "Softmax",
        [num_heads, batch, q_seqlen, kv_seqlen],
    )
    QK_V_flops = num_heads * batch * q_seqlen * d_model * kv_seqlen * 2

    return QK_flops + softmax_QK_flops + QK_V_flops


def get_flops_conv2d(
    input_shape: Sequence[int],  # unused
    kernel_shape: Sequence[int],
    output_shape: Sequence[int],
    einsum_expr: str,
    window: str,  # unused
) -> int:
    '''
    Get the number of FLOPs for a convolution operation.
    The FLOPs is simply (the number of output channels) * (the number of output elements) *
    (kernel size) * (the number of input channels) * (batch size) * 2. ("2" for mul+add.)
    @param input_shape: Shape of input.
    @param kernel_shape: Shape of kernel.
    @param output_shape: Shape of output.
    @param einsum_expr: Einsum dimension expression. E.g., "bf01;01io->bf01".
    @param window: Window config. E.g., "{size=3x3 stride=2x2 pad=0_1x0_1}".
    '''
    einsum_expr_split = einsum_expr.split("->")
    # input_axes, kernel_axes = einsum_expr_split[0].split(";")
    output_axes = einsum_expr_split[1]

    # get output size and batch size
    output_size = 1
    batch_size = 1
    for i, ax in enumerate(output_axes):
        if ax.isdigit():
            output_size *= output_shape[i]
        elif ax == 'b':
            batch_size *= output_shape[i]

    # tot_kernel_size = kernel_size * input_channel * output_channel
    tot_kernel_size = int(np.prod(kernel_shape))

    return batch_size * output_size * tot_kernel_size * 2


def get_flops_einsum(
    input_a_shape: Sequence[int],
    input_b_shape: Sequence[int],
    einsum_expr: str,
) -> int:
    '''
    Get the number of FLOPs for an einsum expression.
    @param input_a_shape: Shape of input A.
    @param input_b_shape: Shape of input B.
    @param einsum_expr: Einsum dimension expression. E.g., "BLND;NDM->BLM".
    '''
    einsum_expr_split = einsum_expr.split("->")
    input_a_axes, input_b_axes = einsum_expr_split[0].split(";")
    output_axes = einsum_expr_split[1]

    input_a_axes_set = {
        (axis, size)
        for axis, size in zip(input_a_axes, input_a_shape)
    }
    input_b_axes_set = {
        (axis, size)
        for axis, size in zip(input_b_axes, input_b_shape)
    }
    output_shape = get_einsum_output_shape(input_a_shape, input_b_shape, einsum_expr)
    output_axes_set = {
        (axis, size)
        for axis, size in zip(output_axes, output_shape)
    }
    all_axes = input_a_axes_set.union(input_b_axes_set).union(output_axes_set)

    # find batch axes: axes that exist in all variables
    batch_axes = input_a_axes_set.intersection(input_b_axes_set).intersection(output_axes_set)

    # find reduction axes: axes that exist in input_a and input_b but not in output
    reduction_axes = input_a_axes_set.union(input_b_axes_set).difference(output_axes_set)

    # find non-reduction axes: everything else
    non_reduction_axes = all_axes.difference(reduction_axes).difference(batch_axes)

    # calculate aggregate axes size
    batch_size = int(np.prod([x[1] for x in batch_axes]))
    reduction_size = int(np.prod([x[1] for x in reduction_axes]))
    non_reduction_size = int(np.prod([x[1] for x in non_reduction_axes]))

    # calculate flops
    return batch_size * reduction_size * non_reduction_size * 2


def get_flops_unary_op(op_name: str, input_shape: Sequence[int]) -> int:
    if op_name == "Softmax":
        return 4 * int(np.prod(input_shape))
    elif op_name in ["LayerNorm", "GroupNorm"]:
        return 8 * int(np.prod(input_shape))
    elif op_name == "RMSNorm":
        return 6 * int(np.prod(input_shape))
    elif op_name == "Pointwise Mul.":
        return 1 * int(np.prod(input_shape))
    else:
        raise ValueError(f"Unsupported unary op: {op_name}")


def get_flops_elementwise_binary_op(op_name: str, input_shape: Sequence[int]) -> int:
    if op_name in ["Add", "Mul"]:
        return int(np.prod(input_shape))
    else:
        raise ValueError(f"Unsupported elementwise binary op: {op_name}")


def get_weight_size_conv2d(
    kernel_shape: Sequence[int], dtype: str = "DT_FLOAT32"
):
    '''
    kernel shape: [input_ch, output_ch, *spatial shape]
    '''
    return int(np.prod(kernel_shape)) * util.get_size_bytes_from_dtype(dtype)


def compute_reduce_scatter_elem_per_chip(
    elem_per_chip: int,
    dim_sizes: Sequence[int],
) -> int:
    tot_elem = 0
    cur_stage_elem = elem_per_chip
    for dim_size in dim_sizes:
        tot_elem += cur_stage_elem
        cur_stage_elem = ceil(cur_stage_elem / dim_size)
    return tot_elem


def compute_all_gather_elem_per_chip(
    elem_per_chip: int,
    dim_sizes: Sequence[int],
) -> int:
    # all gather traffic volume is computed in the same way as reduce scatter,
    # except for the reverse stage order
    tot_elem = elem_per_chip * int(np.prod(dim_sizes))
    return compute_reduce_scatter_elem_per_chip(tot_elem, dim_sizes)


def compute_all_reduce_elem_per_chip(
    elem_per_chip: int,
    dim_sizes: Sequence[int],
) -> int:
    # all reduce traffic is the sum of all gather and reduce scatter
    return 2 * compute_reduce_scatter_elem_per_chip(elem_per_chip, dim_sizes)


def create_weight_update(
    config: ModelConfig,
    param_count: int,
    ici_data_parallel_axes: Sequence[int],
    fusion_id_start: int = 0
) -> list[Operator]:
    ops: list[Operator] = []
    fusion_id = fusion_id_start
    # global_batch_size = batch_size * data_parallelism_degree
    ici_bandwidth = config.ici_bw_GBps
    dcn_bandwidth = config.dcn_bw_GBps
    dcn_data_parallel_degree = config.data_parallel_degree_dcn

    ici_dp_degree = int(np.prod(ici_data_parallel_axes))
    tot_dp_degree = ici_dp_degree * dcn_data_parallel_degree

    # all gather the computed gradients
    if tot_dp_degree > 1:
        data_parallel_axes = list(ici_data_parallel_axes)
        if dcn_data_parallel_degree > 1:
            data_parallel_axes += [dcn_data_parallel_degree]
        axes_bandwidth = [ici_bandwidth] * len(ici_data_parallel_axes) + [dcn_bandwidth]
        ops.append(
            create_all_reduce_op(
                # no need to divide param_count by tensor_parallelism_degree
                # since it's already done in op["Weight Size"]
                input_shape=[1, param_count, 1],
                parallelism_axes=data_parallel_axes,
                axes_bandwidth=axes_bandwidth,
                config=config,
                fusion_id_start=fusion_id,
                count=1,
                description="WeightUpdate"
            )
        )
        fusion_id += 1

    # perform optimization step w/ optimizer states
    ops.extend(
        create_optim_op(
            input_shape=[param_count],
            # dtype=dtype,
            fusion_id=fusion_id,
            count=1,
            description="WeightUpdateOptimizerStates",
        )
    )
    fusion_id = ops[-1].fusion_id + 1

    # weight update: memory bound, so modeled as write to HBM
    weight_update_op = create_output_op(
        input_shape=[param_count],
        fusion_id=fusion_id,
        count=1,
        description="WeightUpdateWriteHBM",
    )
    weight_update_op.op_type = OpType.VPU
    ops.append(weight_update_op)
    fusion_id += 1

    return ops


def create_optim_op(
    input_shape: Sequence[int],
    dtype: str = "DT_BFLOAT16",
    fusion_id: int = 0,
    description: str = "input",
    name: str = "input",
    count: int = 1,
) -> list[Operator]:
    ops: list[Operator] = []
    # load optimizer states from memory (~4x weights)

    op = create_input_op(
        input_shape=[4*i for i in input_shape], # just assuming opt. takes 4x the space as weights
        dtype=dtype,
        fusion_id=fusion_id,
        description=description + ": Load optimizer states from HBM",
        name=name,
        count=count,
    )
    op.op_type = OpType.VPU

    ops.append(op)

    return ops


def create_conv2d_op(
    input_shape: Sequence[int],
    kernel_shape: Sequence[int],
    output_shape: Sequence[int],
    einsum_expr: str,
    window: str,
    dtype: str = "DT_FLOAT32",
    memory_placement: Sequence[int] = [0, 0, 0],
    fusion_id: int = 0,
    description: str = "conv2d",
    name: str = "conv2d",
    count: int = 1,
) -> Operator:
    '''
    @param input_shape: Shape of input.
    @param kernel_shape: Shape of kernel.
    @param output_shape: Shape of output.
    @param einsum_expr: Einsum dimension expression. E.g., "bf01;01io->bf01".
        b: batch;
        f: input channel in lhs; output channel in output;
        i: input channel;
        o: output channel;
        0/1: spatial dimensions (W/H).
    @param window: Window config. E.g., "{size=3x3 stride=2x2 pad=0_1x0_1}".
        size: kernel size;
        stride: conv stride (1x1 means non-strided conv);
        pad: output padding.
    @param dtype: Data type.
    @param memory_placement: Memory placement. 0 for HBM, 1 for SRAM.
    '''
    final_op = Conv2DOperator()

    final_op.fusion_id = fusion_id
    final_op.description = description
    final_op.config_str = format_conv2d_config_str(
        input_shape, kernel_shape, output_shape, einsum_expr, window, dtype, memory_placement
    )
    final_op.name = format_op_name(name, description)
    final_op.op_type = OpType.MXU
    final_op.stats.count = count
    final_op.stats.flop_count = get_flops_conv2d(input_shape, kernel_shape, output_shape, einsum_expr, window)
    final_op.input_tensor_shape_str = format_input_tensor_shapes([input_shape, kernel_shape], dtype)
    final_op.output_tensor_shape_str = format_output_tensor_shape(output_shape, dtype)
    final_op.opcode = "Conv2D"
    final_op.opcode_type = OpcodeType.CONV2D
    final_op.stats.weight_size_bytes = get_weight_size_conv2d(kernel_shape=kernel_shape, dtype=dtype)
    return final_op


def create_einsum_op_bwd(
    input_a_shape: Sequence[int],
    input_b_shape: Sequence[int],
    einsum_expr: str,
    dtype: str = "DT_BFLOAT16",
    memory_placement: Sequence[int] = [0, 0, 0],
    fusion_id: int = 0,
    description: str = "einsum",
    name: str = "einsum",
    count: int = 1,
) -> list[Operator]:
    '''
    @param input_a_shape: Shape of input A.
    @param input_b_shape: Shape of input B.
    @param einsum_expr: Einsum dimension expression. E.g., "BLND;NDM->BLM".
    @param dtype: Data type.
    @param memory_placement: Memory placement. 0 for HBM, 1 for SRAM.
    '''
    x_grad_op = EinsumOperator()

    x_grad_op.fusion_id = fusion_id
    x_grad_op.description = description
    x_grad_op.config_str = format_einsim_config_str(
        input_a_shape, input_b_shape, einsum_expr, dtype, memory_placement
    )
    x_grad_op.op_type = OpType.MXU
    x_grad_op.stats.count = count
    x_grad_op.stats.flop_count = get_flops_einsum(input_a_shape, input_b_shape, einsum_expr)
    x_grad_op.input_tensor_shape_str = format_input_tensor_shapes([input_a_shape, input_b_shape], dtype)
    output_shape = get_einsum_output_shape(input_a_shape, input_b_shape, einsum_expr)
    x_grad_op.output_tensor_shape_str = format_output_tensor_shape(output_shape, dtype)
    x_grad_op.name = format_op_name(name, description) + "XGrad"
    x_grad_op.opcode = "Einsum"
    x_grad_op.opcode_type = OpcodeType.EINSUM

    x_grad_op.input_tensors += [
        Tensor.from_shape("input_a", input_a_shape, dtype),
        Tensor.from_shape("input_b", input_b_shape, dtype),
    ]
    x_grad_op.output_tensors += [
        Tensor.from_shape("output", output_shape, dtype)
    ]

    # Tentative weight size -- NOTE: this does not apply to transformer einsums, which get their
    # Weight Size field overwritten in the create_multi_head...attn function.
    x_grad_op.stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod(input_b_shape))

    fusion_id += 1

    y_grad_op = EinsumOperator()

    y_grad_op.fusion_id = fusion_id
    y_grad_op.description = description
    y_grad_op.config_str = format_einsim_config_str(
        input_a_shape, input_b_shape, einsum_expr, dtype, memory_placement
    )
    y_grad_op.op_type = OpType.MXU
    y_grad_op.stats.count = count
    y_grad_op.stats.flop_count = get_flops_einsum(input_a_shape, input_b_shape, einsum_expr)
    y_grad_op.input_tensor_shape_str = format_input_tensor_shapes([input_a_shape, input_b_shape], dtype)
    output_shape = get_einsum_output_shape(input_a_shape, input_b_shape, einsum_expr)
    y_grad_op.output_tensor_shape_str = format_output_tensor_shape(output_shape, dtype)
    y_grad_op.name = format_op_name(name, description) + "YGrad"
    y_grad_op.opcode = "Einsum"
    y_grad_op.opcode_type = OpcodeType.EINSUM

    y_grad_op.input_tensors += [
        Tensor.from_shape("input_a", input_a_shape, dtype),
        Tensor.from_shape("input_b", input_b_shape, dtype),
    ]
    y_grad_op.output_tensors += [
        Tensor.from_shape("output", output_shape, dtype)
    ]

    # Tentative weight size -- NOTE: this does not apply to transformer einsums, which get their
    # Weight Size field overwritten in the create_multi_head...attn function.
    y_grad_op.stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod(input_b_shape))

    return [x_grad_op, y_grad_op]


def create_einsum_op(
    input_a_shape: Sequence[int],
    input_b_shape: Sequence[int],
    einsum_expr: str,
    dtype: str = "DT_BFLOAT16",
    memory_placement: Sequence[int] = [0, 0, 0],
    fusion_id: int = 0,
    description: str = "einsum",
    name: str = "einsum",
    count: int = 1,
) -> Operator:
    '''
    @param input_a_shape: Shape of input A.
    @param input_b_shape: Shape of input B.
    @param einsum_expr: Einsum dimension expression. E.g., "BLND;NDM->BLM".
    @param dtype: Data type.
    @param memory_placement: Memory placement. 0 for HBM, 1 for SRAM.
    '''
    final_op = EinsumOperator()

    final_op.fusion_id = fusion_id
    final_op.description = description
    final_op.config_str = format_einsim_config_str(
        input_a_shape, input_b_shape, einsum_expr, dtype, memory_placement
    )
    final_op.name = name
    final_op.op_type = OpType.MXU
    final_op.stats.count = count
    final_op.stats.flop_count = get_flops_einsum(input_a_shape, input_b_shape, einsum_expr)

    input_tensors = [
        Tensor.from_shape("input_a", input_a_shape, dtype),
        Tensor.from_shape("input_b", input_b_shape, dtype),
    ]
    output_shape = get_einsum_output_shape(input_a_shape, input_b_shape, einsum_expr)
    output_tensor = Tensor.from_shape("output", output_shape, dtype)
    final_op.input_tensors = input_tensors
    final_op.output_tensors = [output_tensor]
    final_op.input_tensor_shape_str = format_input_tensor_shapes([t.shape for t in input_tensors], dtype)
    final_op.output_tensor_shape_str = format_output_tensor_shape(output_shape, dtype)
    final_op.name = format_op_name(name, description)
    final_op.opcode = "Einsum"
    final_op.opcode_type = OpcodeType.EINSUM

    # Tentative weight size -- NOTE: this does not apply to transformer einsums, which get their
    # Weight Size field overwritten in the create_multi_head...attn function.
    final_op.stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod(input_b_shape))
    return final_op


def create_unary_op(
    input_shape: Sequence[int],
    op_name: str,
    memory_placement: Sequence[int] = [0, 0],
    dtype: str = "DT_BFLOAT16",
    fusion_id: int = 0,
    description: str | None = None,
    name: str | None = None,
    count: int = 1,
) -> Operator:
    '''
    @param input_shape: Shape of input.
    @param memory_placement: Memory placement. 0 for HBM, 1 for SRAM.
    @param dtype: Data type.
    '''
    name = name or op_name
    description = description or op_name
    input_shape_str = "x".join([str(i) for i in input_shape])
    memory_placement_str = "_".join([str(i) for i in memory_placement])
    config_str = f"{op_name}(x={input_shape_str},memory_placements={memory_placement_str},type={dtype})"
    final_op = Operator()
    final_op.fusion_id = fusion_id
    final_op.description = description
    final_op.config_str = config_str
    final_op.name = format_op_name(name, description)
    final_op.op_type = OpType.VPU
    final_op.stats.count = count
    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape, dtype)
    final_op.output_tensor_shape_str = format_output_tensor_shape(input_shape, dtype)
    final_op.stats.flop_count = get_flops_unary_op(op_name, input_shape)
    final_op.opcode = op_name
    final_op.opcode_type = OpcodeType.ELEMENTWISE

    final_op.input_tensors.append(
        Tensor.from_shape("input", input_shape, dtype)
    )
    final_op.output_tensors.append(
        Tensor.from_shape("output", input_shape, dtype)
    )

    # if(op_name == "LayerNorm"):
    #     final_op["Weight Size"] = util.get_size_bytes_from_dtype(dtype) * 2 \
    #                             * int(np.prod([num_heads, d_head, d_query]))

    return final_op


def create_embedding_bag(
    batch_size: int,
    num_indices: Sequence[int],
    table_sizes: Sequence[int],
    embedding_dim: int,
    reduction_op: str = "sum",
    memory_placement: Sequence[int] = [0, 0],
    dtype: str = "DT_FLOAT",
    fusion_id: int = 0,
    name: str | None = None,
    description: str | None = None,
    count: int = 1,
) -> list[Operator]:
    '''
    Create an embedding bag op (like the one in PyTorch) for a single chip.
    Assumes parallelism shardings are alredy applied to the input dimensions.
    By default, the embedding bag op sums up all fetched entries into the final output.
    Computes the compute time (MXU=0, VPU=computed) and HBM memory traffic/time.
    '''
    op = Operator()
    op.fusion_id = fusion_id
    op.description = description or "EmbeddingBag"
    op.config_str = f"EmbeddingBag(batch={batch_size},table_sizes={table_sizes},num_indices={num_indices},embedding_dim={embedding_dim},reduction={reduction_op},memory_placements={memory_placement},type={dtype})"
    op.name = name or "EmbeddingBag"
    op.op_type = OpType.VPU
    op.opcode_type = OpcodeType.EMBEDDING_BAG
    op.opcode = "EmbeddingBag"
    op.stats.count = count

    # Input Tensor 1: [batch_size, num_tables, max num_indices for each table]
    # Input Tensor 2: All tables
    # Output Tensor: [batch_size, num_tables, embed_dim]
    # Embedding Table: [table_size, embed_dim]
    op.input_tensors.append(
        Tensor.from_shape(
            "indices",
            [batch_size, len(table_sizes), max(num_indices)],
            "DT_INT64",
        )
    )
    op.input_tensors += [
        Tensor.from_shape(
            f"embedding_table{i}",
            [table_size, embedding_dim],
            dtype,
        )
        for i, table_size in enumerate(table_sizes)
    ]
    op.input_tensor_shape_str = format_input_tensor_shapes(
        [tensor.shape for tensor in op.input_tensors],
        [tensor.dtype for tensor in op.input_tensors],
    )
    op.output_tensors.append(
        Tensor.from_shape(
            "embedding_output",
            [batch_size, len(table_sizes), embedding_dim],
            dtype,
        )
    )
    op.output_tensor_shape_str = format_output_tensor_shape(op.output_tensors[0].shape, op.output_tensors[0].dtype)

    # HBM time
    bytes_per_element = util.get_size_bytes_from_dtype(dtype)
    num_entries = batch_size * sum(num_indices)
    op.stats.memory_traffic_bytes = (
        num_entries * 8  # read indices; indices are int64
        + num_entries * embedding_dim * bytes_per_element  # fetch embedding vectors
        + batch_size * len(table_sizes) * embedding_dim * bytes_per_element  # write output
    )

    # VPU FLOP count and compute time
    if reduction_op == "sum":
        op.stats.flop_count = num_entries * embedding_dim * 2
    else:
        raise ValueError(f"Unsupported reduction op: {reduction_op}")

    return [op]


def create_softmax_op_bwd(
    input_shape: Sequence[int],
    op_name: str,
    memory_placement: Sequence[int] = [0, 0],
    dtype: str = "DT_BFLOAT16",
    fusion_id: int = 0,
    description: str | None = None,
    name: str | None = None,
    count: int = 1,
) -> list[Operator]:
    # dl/dx_p = dl/dy_p * func'(x_p) -> func = e^x, func' = e^x (known)
    # only need to do mul -> 1-flop unary op
    op = create_unary_op(
        input_shape=input_shape,
        op_name="Pointwise Mul.",
        dtype=dtype,
        memory_placement=[1, 1],
        name="Softmax Backprop",
        count=count,
        fusion_id=fusion_id,
        description=description,
    )
    return [op]


def create_layernorm_op_bwd(
    input_shape: Sequence[int],
    op_name: str,
    memory_placement: Sequence[int] = [0, 0],
    dtype: str = "DT_BFLOAT16",
    fusion_id: int = 0,
    description: str | None = None,
    name: str | None = None,
    count: int = 1,
) -> list[Operator]:

    ops: list[Operator] = []

    # dl/dB = sum(dl/dy) -> summation
    ops.append(
        create_unary_op(
            input_shape=input_shape,
            op_name="Softmax",
            dtype=dtype,
            memory_placement=[1, 1], # TODO what is this
            name="Softmax Backprop.",
            count=1,
            fusion_id=fusion_id,
        )
    )

    # dl/dy = einsum
    ops.append(
        create_einsum_op(
            input_a_shape=input_shape,
            input_b_shape=input_shape,
            einsum_expr="BLV;BLV->BV",
            dtype=dtype,
            memory_placement=[0, 0, 1],
            name="MatMul: dl/dy",
            description=(f"softmax-bwd-dl/dy-{fusion_id}"),
            count=1,
            fusion_id=fusion_id,
        )
    )

    # e1 = einsum
    ops.append(
        create_einsum_op(
            input_a_shape=input_shape,
            input_b_shape=input_shape,
            einsum_expr="BLN;BLN->BL",
            dtype=dtype,
            memory_placement=[0, 0, 1],
            name="MatMul: e1",
            description=(f"softmax-bwd-e1-{fusion_id}"),
            count=1,
            fusion_id=fusion_id,
        )
    )

    # e2 = einsum
    ops.append(
        create_einsum_op(
            input_a_shape=input_shape,
            input_b_shape=input_shape,
            einsum_expr="BLN;BLN->BL",
            dtype=dtype,
            memory_placement=[0, 0, 1],
            name="MatMul: e2",
            description=(f"softmax-bwd-e2-{fusion_id}"),
            count=1,
            fusion_id=fusion_id,
        )
    )

    # dl/dx = dl/dy*Y/signma - 1/H*e1 - 1/H*mu*e2
    # is dl/dY * Y a matmul? -> yes, and then just scaling
    return ops


def create_elementwise_binary_op(
    input_shape: Sequence[int],
    op_name: str,
    memory_placement: Sequence[int] = [0, 0, 0],
    dtype: str = "DT_BFLOAT16",
    fusion_id: int = 0,
    description: str | None = None,
    name: str | None = None,
    count: int = 1,
) -> Operator:
    '''
    @param input_shape: Shape of input.
    @param memory_placement: Memory placement. 0 for HBM, 1 for SRAM.
    @param op_name: Name of the operation.
    @param dtype: Data type.
    '''
    name = name or op_name
    description = description or op_name
    input_shape_str = "x".join([str(i) for i in input_shape])
    memory_placement_str = "_".join([str(i) for i in memory_placement])
    config_str = f"{op_name}(a={input_shape_str},b={input_shape_str},memory_placements={memory_placement_str},type={dtype})"
    final_op = Operator()
    final_op.fusion_id = fusion_id
    final_op.description = description
    final_op.config_str = config_str
    final_op.op_type = OpType.VPU
    final_op.stats.count = count
    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape, dtype)
    final_op.output_tensor_shape_str = format_output_tensor_shape(input_shape, dtype)
    final_op.stats.flop_count = get_flops_elementwise_binary_op(op_name, input_shape)
    final_op.name = format_op_name(name, description)
    final_op.opcode = op_name
    final_op.opcode_type = OpcodeType.ELEMENTWISE

    final_op.input_tensors += [
        Tensor.from_shape("input_a", input_shape, dtype),
        Tensor.from_shape("input_b", input_shape, dtype),
    ]
    final_op.output_tensors += [
        Tensor.from_shape("output", input_shape, dtype)
    ]

    return final_op


def create_input_op(
    input_shape: Sequence[int],
    dtype: str = "DT_BFLOAT16",
    fusion_id: int = 0,
    description: str = "input",
    name: str = "input",
    count: int = 1,
) -> Operator:
    '''
    @param input_shape: Shape of input.
    @param memory_placement: Memory placement. 0 for HBM, 1 for SRAM.
    @param dtype: Data type.
    '''
    input_shape_str = "x".join([str(i) for i in input_shape])
    config_str = f"Abs(x={input_shape_str},type={dtype})"
    final_op = Operator()
    final_op.fusion_id = fusion_id
    final_op.description = description
    final_op.config_str = config_str
    final_op.op_type = OpType.OTHER
    final_op.stats.count = count
    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape, dtype)
    final_op.output_tensor_shape_str = format_output_tensor_shape(input_shape, dtype)
    final_op.stats.flop_count = 0
    final_op.name = format_op_name(name, description)
    final_op.opcode = "Input"
    final_op.opcode_type = OpcodeType.OTHER

    final_op.input_tensors.append(
        Tensor.from_shape("input", input_shape, dtype)
    )
    final_op.output_tensors.append(
        Tensor.from_shape("output", input_shape, dtype)
    )

    return final_op


def create_output_op(
    input_shape: Sequence[int],
    dtype: str = "DT_BFLOAT16",
    fusion_id: int = 0,
    description: str = "output",
    name: str = "output",
    count: int = 1,
) -> Operator:
    '''
    @param input_shape: Shape of input.
    @param memory_placement: Memory placement. 0 for HBM, 1 for SRAM.
    @param dtype: Data type.
    '''
    input_shape_str = "x".join([str(i) for i in input_shape])
    config_str = f"Abs(x={input_shape_str},type={dtype})"
    final_op = Operator()
    final_op.fusion_id = fusion_id
    final_op.description = description
    final_op.config_str = config_str
    final_op.name = name
    final_op.op_type = OpType.OTHER
    final_op.stats.count = count
    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape, dtype)
    final_op.output_tensor_shape_str = format_output_tensor_shape(input_shape, dtype)
    final_op.stats.flop_count = 0
    final_op.name = format_op_name(name, description)
    final_op.opcode = "Output"
    final_op.opcode_type = OpcodeType.OTHER

    final_op.input_tensors.append(
        Tensor.from_shape("input", input_shape, dtype)
    )
    final_op.output_tensors.append(
        Tensor.from_shape("output", input_shape, dtype)
    )

    return final_op


def create_fourier_embedding_op(
    batch_size: int,
    seqlen: int,
    feature_dim: int,
    num_freqs: int,
    num_layers: int,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
) -> Operator:
    '''Fourier embedding op (e.g., used in GLIGEN to process spatial grounding features).'''
    name = op_name = "FourierEmbedding"
    description = "FourierEmbedding"

    input_shape = [batch_size, seqlen, feature_dim]
    output_shape = [batch_size, seqlen, num_freqs * 2 * feature_dim]
    memory_placement = [0, 0, 0]

    input_shape_str = "x".join([str(i) for i in input_shape])
    # output_shape_str = "x".join([str(i) for i in output_shape])
    memory_placement_str = "_".join([str(i) for i in memory_placement])
    config_str = f"{op_name}(a={input_shape_str},num_freqs={num_freqs},memory_placements={memory_placement_str},type={dtype})"

    final_op = Operator()
    final_op.fusion_id = fusion_id_start
    final_op.description = description
    final_op.config_str = config_str
    final_op.op_type = OpType.VPU
    final_op.stats.count = num_layers
    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape, dtype)
    final_op.output_tensor_shape_str = format_output_tensor_shape(output_shape, dtype)
    # sin+cos for each freq for each feature
    final_op.stats.flop_count = 2 * num_freqs * feature_dim * seqlen * batch_size
    final_op.name = format_op_name(name, description)
    final_op.opcode = op_name
    final_op.opcode_type = OpcodeType.OTHER

    return final_op


def create_upsample_op(
    batch_size: int,
    input_channels: int,
    input_spatial_shape: Sequence[int],
    scale_factor: int,
    num_layers: int,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description: str = "Upsample",
) -> Operator:
    '''Upsample op (e.g., used in GLIGEN/UNet to upsample spatial features).'''
    name = op_name = "Upsample"

    input_shape = [batch_size, input_channels] + list(input_spatial_shape)
    output_shape = [batch_size, input_channels] + [
        x * scale_factor for x in input_spatial_shape
    ]
    memory_placement = [0, 0, 0]

    input_shape_str = "x".join([str(i) for i in input_shape])
    # output_shape_str = "x".join([str(i) for i in output_shape])
    memory_placement_str = "_".join([str(i) for i in memory_placement])
    config_str = f"{op_name}(a={input_shape_str},scale_factor={scale_factor},memory_placements={memory_placement_str},type={dtype})"

    final_op = Operator()
    final_op.fusion_id = fusion_id_start
    final_op.description = description
    final_op.config_str = config_str
    final_op.op_type = OpType.VPU
    final_op.stats.count = num_layers
    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape, dtype)
    final_op.output_tensor_shape_str = format_output_tensor_shape(output_shape, dtype)
    final_op.stats.flop_count = 0
    final_op.name = format_op_name(name, description)
    final_op.opcode = op_name
    final_op.opcode_type = OpcodeType.UP_DOWN_SAMPLE

    final_op.input_tensors += [
        Tensor.from_shape("input", input_shape, dtype)
    ]
    final_op.output_tensors += [
        Tensor.from_shape("output", output_shape, dtype)
    ]

    return final_op


def create_downsample_op(
    batch_size: int,
    input_channels: int,
    input_spatial_shape: Sequence[int],
    scale_factor: int,
    num_layers: int,
    use_conv: bool = False,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description: str = "Downsample",
) -> Operator:
    '''Upsample op (e.g., used in GLIGEN/UNet to downsample spatial features).'''
    name = op_name = "Downsample"
    if use_conv:
        op_name = "Conv2D"
    else:
        op_name = "AvgPool2d"

    input_shape = [batch_size, input_channels] + list(input_spatial_shape)
    output_shape = [batch_size, input_channels] + [
        max(1, x // scale_factor) for x in input_spatial_shape
    ]
    memory_placement = [0, 0, 0]

    input_shape_str = "x".join([str(i) for i in input_shape])
    # output_shape_str = "x".join([str(i) for i in output_shape])
    memory_placement_str = "_".join([str(i) for i in memory_placement])
    if use_conv:
        config_str = format_conv2d_config_str(
            input_shape=input_shape,
            kernel_shape=[input_channels, input_channels, scale_factor, scale_factor],
            output_shape=output_shape,
            einsum_expr="bf01;io01->bf01",
            window="{" + f"size={scale_factor}x{scale_factor} stride=1x1 pad=0_1x0_1" + "}",
            dtype=dtype,
            memory_placement=memory_placement,
        )
    else:
        config_str = f"{op_name}(a={input_shape_str},scale_factor={scale_factor},memory_placements={memory_placement_str},type={dtype})"

    final_op = Conv2DOperator() if use_conv else Operator()
    final_op.fusion_id = fusion_id_start
    final_op.description = description
    final_op.config_str = config_str
    final_op.op_type = OpType.MXU if use_conv else OpType.VPU
    final_op.stats.count = num_layers
    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape, dtype)
    final_op.output_tensor_shape_str = format_output_tensor_shape(output_shape, dtype)
    final_op.stats.flop_count = (
        get_flops_conv2d(
            input_shape,
            [input_channels, input_channels, scale_factor, scale_factor],
            output_shape,
            "bf01;io01->bf01",
            "{" + f"size={scale_factor}x{scale_factor} stride=1x1 pad=0_1x0_1" + "}"
        )
        if use_conv else
        int(np.prod(input_shape) + np.prod(output_shape))
    )
    final_op.name = format_op_name(name, description)
    final_op.opcode = op_name
    final_op.opcode_type = OpcodeType.CONV2D if use_conv else OpcodeType.UP_DOWN_SAMPLE

    final_op.input_tensors += [
        Tensor.from_shape("input", input_shape, dtype)
    ]
    final_op.output_tensors += [
        Tensor.from_shape("output", output_shape, dtype)
    ]

    return final_op


def create_input_transfer_op(
    input_shape: Sequence[int],
    config: ModelConfig | ChipConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description: str = "InputOp",
    name: str = "InterChipInputOp",
    count: int = 1
) -> Operator:
    '''
        Simulates processor accepting data from an upstream node.
        Meant to be used for passing data between pipeline stages.
    '''

    return create_ic_transfer_op(
        input_shape=input_shape,
        config=config,
        dtype=dtype,
        fusion_id_start=fusion_id_start,
        description=description,
        name=name,
        count=count,
        transfer_type="Input"
    )


def create_output_transfer_op(
    input_shape: Sequence[int],
    config: ModelConfig | ChipConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description: str = "OutputOp",
    name: str = "InterChipOutputOp",
    count: int = 1
) -> Operator:
    '''
        Simulates processor passing data to a downstream node.
        Meant to be used for passing data between pipeline stages.
    '''

    return create_ic_transfer_op(
        input_shape=input_shape,
        config=config,
        dtype=dtype,
        fusion_id_start=fusion_id_start,
        description=description,
        name=name,
        count=count,
        transfer_type="Output"
    )


def create_ic_transfer_op(
    input_shape: Sequence[int],
    config: ModelConfig | ChipConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description: str = "XferOp",
    name: str = "InterChipXferOp",
    count: int = 1,
    transfer_type: str = "Input",
) -> Operator:
    '''
    Creates an interchip transfer operation with the specified transfer type.
    '''
    allowable_transfer_types = ["Input", "Output"]
    assert(transfer_type in allowable_transfer_types), \
    f"Invalid transfer type. Use one of the following: {allowable_transfer_types}."

    bytes_per_elem = util.get_size_bytes_from_dtype(dtype)
    final_op = Operator()
    transfer_size = int(np.prod(np.array(input_shape)) * bytes_per_elem)    # Cast to int for JSON serializability

    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape)
    final_op.fusion_id = fusion_id_start
    final_op.description = description
    final_op.name = format_op_name(name, description)
    final_op.opcode = f"InterChipXfer{transfer_type}"
    final_op.opcode_type = OpcodeType.OTHER

    final_op.stats.ici_traffic_outbound_bytes = transfer_size if transfer_type == "Output" else 0
    final_op.stats.ici_traffic_inbound_bytes = transfer_size if transfer_type == "Input" else 0
    final_op.stats.ici_time_ns = max(
        ceil(transfer_size / config.ici_bw_GBps),
        config.ici_latency_ns,
    )
    final_op.op_type = OpType.ICI_NO_COMPUTE
    final_op.stats.count = count
    final_op.config_str = f"InterChipComm{transfer_type}({input_shape})"
    final_op.output_tensor_shape_str = format_output_tensor_shape(input_shape)

    final_op.stats.memory_traffic_bytes = (
        final_op.stats.ici_traffic_outbound_bytes
        + final_op.stats.ici_traffic_inbound_bytes
    )

    return final_op


def create_kvswap_op(
    input_shape: Sequence[int],
    config: ModelConfig | ChipConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description: str = "XferOp",
    name: str = "InterChipXferOp",
    count: int = 1,
    transfer_type: str = "Input",
) -> Operator:
    '''
    Creates a KV cache swap op.
    '''
    allowable_transfer_types = ["Input", "Output"]
    assert(transfer_type in allowable_transfer_types), \
    f"Invalid transfer type. Use one of the following: {allowable_transfer_types}."

    bytes_per_elem = util.get_size_bytes_from_dtype(dtype)
    final_op = Operator()
    transfer_size = int(np.prod(np.array(input_shape)) * bytes_per_elem)    # Cast to int for JSON serializability

    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape)
    final_op.fusion_id = fusion_id_start
    final_op.description = description
    final_op.name = format_op_name(name, description)
    final_op.opcode = f"KVSwap{transfer_type}"
    final_op.opcode_type = OpcodeType.OTHER

     # just use ici traffic fields to emulate PCIe stats for compatibility and simplicity
    final_op.stats.ici_traffic_outbound_bytes = transfer_size if transfer_type == "Output" else 0
    final_op.stats.ici_traffic_inbound_bytes = transfer_size if transfer_type == "Input" else 0
    final_op.stats.ici_time_ns = max(
        ceil(transfer_size * 1e9 / 1024 / 1024 / 1024 / config.pcie_bw_GBps),
        config.ici_latency_ns,
    )
    final_op.stats.pcie_time_ns = max(
        ceil(transfer_size * 1e9 / 1024 / 1024 / 1024 / config.pcie_bw_GBps),
        config.pcie_latency_ns,
    )
    final_op.op_type = OpType.ICI_NO_COMPUTE  # use ICI_NO_COMPUTE type for now, although it is actually not ICI
    final_op.stats.count = count
    final_op.config_str = f"KVSwap{transfer_type}({input_shape})"
    final_op.output_tensor_shape_str = format_output_tensor_shape(input_shape)

    final_op.stats.memory_traffic_bytes = (
        final_op.stats.ici_traffic_outbound_bytes
        + final_op.stats.ici_traffic_inbound_bytes
    )

    return final_op


def create_all_to_all_op(
    input: Tensor,
    config: ModelConfig,
    bisection_bw: float,
    num_parallelism: int,
    dtype: str = "DT_BFLOAT16",
    fusion_id: int = 0,
    description: str = "AllToAll",
    name: str = "AllToAll",
    count: int = 1,
) -> Operator:
    '''
    @input: Input tensor.
    @bisection_bw: Bisection bandwidth per chip.
    @num_parallelism: Parallelism degrees.
    '''
    input_shape = input.shape
    bytes_per_elem = util.get_size_bytes_from_dtype(dtype)
    elems_per_chip = int(np.prod(np.array(input_shape)))
    agg_bw = bisection_bw

    # num_transfers = 2 * (parallelism_degree - 1)
    transfer_size = bytes_per_elem * elems_per_chip            # in Bytes
    input_shape_str = "x".join([str(i) for i in input_shape])

    final_op = Operator()
    final_op.input_tensors.append(input)
    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape)
    final_op.fusion_id = fusion_id
    final_op.description = description
    final_op.name = format_op_name(name, description)
    final_op.opcode = "AllToAll"
    final_op.stats.ici_traffic_outbound_bytes = transfer_size # traffic for each TPU
    final_op.stats.ici_traffic_inbound_bytes = transfer_size
    final_op.stats.ici_time_ns = max(
        ceil(
            max(
                final_op.stats.ici_traffic_outbound_bytes,
                final_op.stats.ici_traffic_inbound_bytes,
            ) / agg_bw
        ),
        # assume in the worst case, each shard needs to send (num_parallelism - 1) messages (to each other shard)
        config.ici_latency_ns * (num_parallelism - 1),
    )
    final_op.stats.memory_traffic_bytes = (
        final_op.stats.ici_traffic_outbound_bytes
        + final_op.stats.ici_traffic_inbound_bytes
    )

    final_op.output_tensors.append(
        Tensor.from_shape("alltoall_output", input_shape, dtype=dtype)
    )
    final_op.output_tensor_shape_str = format_output_tensor_shape(input_shape)
    final_op.stats.count = count
    final_op.op_type = OpType.ICI_NO_COMPUTE
    output_shape_str = "x".join([str(i) for i in input_shape])
    final_op.config_str = f"AllToAll({input_shape_str}->{output_shape_str})"

    return final_op


def create_all_reduce_op(
    input_shape: Sequence[int],
    parallelism_axes: Sequence[int],
    axes_bandwidth: Sequence[float],
    config: ModelConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description: str = "AllReduce",
    name: str = "AllReduce",
    count: int = 1,
) -> Operator:
    '''
        input shape = shape of tensor for each chip.
        Each processor needs to receive data from other processors.
    '''
    logging.vlog(3, "############## AR Begin ############")
    logging.vlog(3, "AR Input: %s", input_shape)
    bytes_per_elem = util.get_size_bytes_from_dtype(dtype)
    elems_per_chip = int(np.prod(np.array(input_shape)))
    elems_per_chip = compute_all_reduce_elem_per_chip(elems_per_chip, parallelism_axes)
    parallelism_degree = int(np.prod(parallelism_axes))
    agg_bw = sum(axes_bandwidth)

    # num_transfers = 2 * (parallelism_degree - 1)
    transfer_size = bytes_per_elem * elems_per_chip            # in Bytes
    input_shape_str = "x".join([str(i) for i in input_shape])

    final_op = Operator()
    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape)
    final_op.fusion_id = fusion_id_start
    final_op.description = description
    final_op.name = format_op_name(name, description)
    final_op.opcode = "AllReduce"
    final_op.stats.ici_traffic_outbound_bytes = transfer_size # traffic for each TPU
    final_op.stats.ici_traffic_inbound_bytes = transfer_size
    # TODO: DCN time is also included here; maybe separate them in the future.
    final_op.stats.ici_time_ns = max(
        ceil(
            max(
                final_op.stats.ici_traffic_inbound_bytes,
                final_op.stats.ici_traffic_outbound_bytes,
            ) / agg_bw
        ),
        # all reduce steps is (num_shards - 1) * 2
        # TODO: currently assumes all axes have the same latency. Need to consider DCN latency separately.
        config.ici_latency_ns * (parallelism_degree - 1) * 2,
    )
    final_op.stats.memory_traffic_bytes = (
        final_op.stats.ici_traffic_outbound_bytes
        + final_op.stats.ici_traffic_inbound_bytes
    )
    # final_op["Output Shape"]            = deepcopy(list(input_shape))
    # final_op["Output Shape"][gather_dim] *= parallelism_degree # Gathering concats all copies along specified dim

    final_op.output_tensor_shape_str = format_output_tensor_shape(input_shape)
    final_op.stats.count = count
    final_op.op_type = OpType.VPU
    final_op.opcode_type = OpcodeType.COLLECTIVE_REDUCE
    output_shape_str = "x".join([str(i) for i in input_shape])
    final_op.config_str = f"AllReduce({input_shape_str}->{output_shape_str})"

    final_op.input_tensors.append(
        Tensor.from_shape("allreduce_input", input_shape, dtype=dtype)
    )
    final_op.output_tensors.append(
        Tensor.from_shape("allreduce_output", input_shape, dtype=dtype)
    )

    logging.vlog(3, "AR Output: %s", input_shape)  # all-reduce has the same input/output shapes
    logging.vlog(3, "############## AR End ############")
    return final_op


def create_all_gather_op(
    input_shape: Sequence[int],
    parallelism_axes: Sequence[int],
    axes_bandwidth: Sequence[float],
    gather_dim: int,
    config: ModelConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description: str = "AllGather",
    name: str = "AllGather",
    count: int = 1
) -> Operator:
    '''
        input shape = shape of tensor for each tpu
        Each processor needs to receive data from other processors.
    '''
    logging.vlog(3, "############## AG Begin ############")
    logging.vlog(3, "AG Input: %s", input_shape)
    bytes_per_elem = util.get_size_bytes_from_dtype(dtype)
    elems_per_chip = int(np.prod(np.array(input_shape)))         # Multiply the batch, ch, spatial dims for each tpu chip
    elems_per_chip = compute_all_gather_elem_per_chip(elems_per_chip, parallelism_axes)
    agg_bw = sum(axes_bandwidth)
    parallelism_degree = int(np.prod(parallelism_axes))

    # num_transfers = parallelism_degree - 1                  # Need to get data from other nodes.
    transfer_size = bytes_per_elem * elems_per_chip         # in Bytes
    input_shape_str = "x".join([str(i) for i in input_shape])

    final_op = Operator()
    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape)
    final_op.fusion_id = fusion_id_start
    final_op.description = description
    final_op.name = name
    final_op.name = format_op_name(name, description)
    final_op.opcode = "AllGather"
    final_op.opcode_type = OpcodeType.COLLECTIVE_NO_COMPUTE
    final_op.stats.ici_traffic_outbound_bytes = transfer_size # traffic for each TPU
    final_op.stats.ici_traffic_inbound_bytes = transfer_size
    # TODO: DCN time is also included here; maybe separate them in the future.
    final_op.stats.ici_time_ns = max(
        ceil(
            max(
                final_op.stats.ici_traffic_inbound_bytes,
                final_op.stats.ici_traffic_outbound_bytes,
            ) / agg_bw
        ),
        # all gather steps is (num_shards - 1).
        # TODO: currently assumes all axes have the same latency. Need to consider DCN latency separately.
        config.ici_latency_ns * (parallelism_degree - 1),
    )
    final_op.stats.memory_traffic_bytes = (
        final_op.stats.ici_traffic_outbound_bytes
        + final_op.stats.ici_traffic_inbound_bytes
    )
    # final_op["Output Shape"]            = deepcopy(list(input_shape))                     # Shallow copy -- input_shape will also be modified.
    # final_op["Output Shape"][gather_dim] *= parallelism_degree # Gathering concats all copies along specified dim
    output_shape = deepcopy(list(input_shape))
    output_shape[gather_dim] *= parallelism_degree
    final_op.output_tensor_shape_str = format_output_tensor_shape(output_shape)
    final_op.stats.count = count
    final_op.op_type = OpType.ICI_NO_COMPUTE
    output_shape_str = "x".join([str(i) for i in output_shape])
    final_op.config_str = f"AllGather({input_shape_str}->{output_shape_str})"

    final_op.input_tensors.append(
        Tensor.from_shape("allgather_input", input_shape, dtype=dtype)
    )
    final_op.output_tensors.append(
        Tensor.from_shape("allgather_output", output_shape, dtype=dtype)
    )

    logging.vlog(3, "AG Output: %s", output_shape)
    logging.vlog(3, "############## AG End ############")
    return final_op


def create_reduce_scatter_op(
    input_shape: Sequence[int],
    parallelism_axes: Sequence[int],
    axes_bandwidth: Sequence[float],
    reduction_dim: int,
    config: ModelConfig | ChipConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description: str = "ReduceScatter",
    name: str = "ReduceScatter",
    count: int = 1
) -> Operator:

    '''
    Each node will iteratively receive portion of the data from all other nodes, performing reduction at each stage.
    At each step, the node will also send out its portion of data to another node.
    '''

    logging.vlog(3, "############## RS Begin ############")
    logging.vlog(3, "RS Input: %s", input_shape)

    bytes_per_elem = util.get_size_bytes_from_dtype(dtype)
    num_total_elems = np.prod(np.array(input_shape))        # Number of elements in overall array; product of input dims.
    parallelism_degree = int(np.prod(parallelism_axes))
    agg_bw = sum(axes_bandwidth)
    elems_per_chip = ceil(num_total_elems / parallelism_degree)   # Each chip will only be responsible for a portion of elems.
    elems_per_chip = compute_reduce_scatter_elem_per_chip(elems_per_chip, parallelism_axes)
    # num_transfers = parallelism_degree - 1                  # Need to get data from other t-1 nodes.
    transfer_size = int(bytes_per_elem * elems_per_chip)         # in Bytes
    final_op = Operator()
    final_op.input_tensor_shape_str = format_input_tensor_shapes(input_shape)
    final_op.fusion_id = fusion_id_start
    final_op.description = description
    final_op.name = format_op_name(name, description)
    final_op.opcode = "ReduceScatter"
    # final_op["ICI/NVLink time"]         = time_per_transfer * num_transfers  * 2 # x 2 for inbound + outbound
    final_op.stats.ici_traffic_inbound_bytes = transfer_size # traffic for each TPU
    final_op.stats.ici_traffic_outbound_bytes = transfer_size
    # TODO: DCN time is also included here; maybe separate them in the future.
    final_op.stats.ici_time_ns = max(
        ceil(
            max(
                final_op.stats.ici_traffic_inbound_bytes,
                final_op.stats.ici_traffic_outbound_bytes,
            ) / agg_bw
        ),
        # reeduce scatter steps is (num_shards - 1).
        # TODO: currently assumes all axes have the same latency. Need to consider DCN latency separately.
        config.ici_latency_ns * (parallelism_degree - 1),
    )
    final_op.stats.flop_count = ceil(elems_per_chip * (parallelism_degree - 1)) # Need t - 1 additions to reduce all outputs.
    final_op.stats.memory_traffic_bytes = (
        final_op.stats.ici_traffic_outbound_bytes
        + final_op.stats.ici_traffic_inbound_bytes
    )

    output_shape = deepcopy(list(input_shape))  # B S D
    output_shape[reduction_dim] = ceil(output_shape[reduction_dim] / parallelism_degree)  # B S D/t Scatter op splits results along reduction dim.
    final_op.output_tensor_shape_str = format_output_tensor_shape(output_shape)
    final_op.input_tensors.append(
        Tensor.from_shape("reduce_scatter_input", input_shape, dtype=dtype)
    )
    final_op.output_tensors.append(
        Tensor.from_shape("reduce_scatter_output", output_shape, dtype=dtype)
    )
    final_op.stats.count = count
    final_op.op_type = OpType.VPU
    final_op.opcode_type = OpcodeType.COLLECTIVE_REDUCE
    final_op.config_str = f"ReduceScatter({[str(i) for i in input_shape]}->{[str(i) for i in output_shape]})"

    logging.vlog(3, "RS Output: %s", output_shape)
    logging.vlog(3, "############## RS End ############")

    return final_op


def create_multi_head_latent_attention(
    batch_size: int,
    config: ModelConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    is_decode: bool = False,
    description_prefix: str = "",
    use_flash_attention: bool = False,
    tensor_parallelism_axes: Sequence[int] = [1],
) -> list[Operator]:
    '''MLA attention layer in DeepSeek models.'''

    assert isinstance(config, DeepSeekConfig), \
        f"Latent attention is only supported for DeepSeek models. Got {config.__class__}"
    assert not use_flash_attention, \
        "Flash attention is not supported for multi-head latent attention for now."

    fusion_id = fusion_id_start
    ops: list[Operator] = []

    if is_decode:  # decode
        count = config.num_layers * config.output_seqlen
        seqlen = config.decode_width
        descript_prefix = description_prefix + "Attention-serving-decode-"
    else:  # prefill
        count = config.num_layers
        seqlen = config.input_seqlen
        descript_prefix = description_prefix + "Attention-serving-prefill-"


    # Preprocess input dimensions
    tensor_parallelism_degree = int(np.prod(tensor_parallelism_axes))
    num_local_heads = ceil(config.num_heads / tensor_parallelism_degree)
    d_model = config.d_model

    h_t_shape = [batch_size, seqlen, d_model]
    kv_shape = [batch_size, seqlen, config.kv_lora_rank]
    k_pe_shape = [batch_size, seqlen, config.qk_rope_head_dim]
    wkv_a_shape = [d_model, config.kv_lora_rank + config.qk_rope_head_dim]  # W^{DKV} matrix
    wkv_b_shape = [config.kv_lora_rank, num_local_heads * (config.qk_nope_head_dim + config.v_head_dim)]  # W^{Q} matrix, W^{UK} is absorbed into W^{Q}
    wo_shape = [num_local_heads * config.v_head_dim, d_model]  # output linear projection
    wq_a_shape = [d_model, config.q_lora_rank]  # W^{DQ} matrix
    wq_b_shape = [config.q_lora_rank, config.qk_head_dim * num_local_heads]  # W^{UQ} matrix
    q_nope_shape = [batch_size, seqlen, num_local_heads, config.qk_nope_head_dim]
    q_pe_shape = [batch_size, seqlen, num_local_heads, config.qk_rope_head_dim]

    if tensor_parallelism_degree > 1:
        ag_op = create_all_gather_op(
            input_shape=[batch_size, seqlen, d_model],
            parallelism_axes=tensor_parallelism_axes,
            axes_bandwidth=[config.ici_bw_GBps] * len(tensor_parallelism_axes),
            gather_dim=2,
            config=config,
            fusion_id_start=fusion_id,
            count=count,
            description=f"AllGatherMHA-{fusion_id}"
        )
        batch_size, seqlen, d_model = ag_op.output_tensors[0].shape  # B, S, D/t -> AllGather = B, S, D
        ops.append(ag_op)
        fusion_id += 1

    # q = self.wq_b(self.q_norm(self.wq_a(x)))
    # q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
    # q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    # q_pe = apply_rotary_emb(q_pe, freqs_cis)
    ops.append(
        create_einsum_op(
            input_a_shape=h_t_shape,
            input_b_shape=wq_a_shape,
            einsum_expr="BSM;MD->BSD",
            dtype="DT_BFLOAT16",
            fusion_id=fusion_id,
            description=f"{descript_prefix}q_wq_a_linear",
            name="wqax = self.wq_a(x)",
            count=count,
        )
    )
    fusion_id += 1
    ops.append(
        create_unary_op(
            input_shape=[batch_size, seqlen, config.q_lora_rank],
            op_name="RMSNorm",
            dtype="DT_FLOAT32",
            fusion_id=fusion_id,
            description=f"{descript_prefix}q_norm",
            name="q_norm_wqax = self.q_norm(wqax)",
            count=count,
        )
    )
    fusion_id += 1
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, config.q_lora_rank],
            input_b_shape=wq_b_shape,  # this is local wq_b_shape after tensor parallelism
            einsum_expr="BSR;RD->BSD",
            dtype="DT_BFLOAT16",
            fusion_id=fusion_id,
            description=f"{descript_prefix}q_wq_b_linear",
            name="q = self.wq_b(q_norm_wqax)",
            count=count,
        )
    )
    fusion_id += 1
    ops.append(
        create_elementwise_binary_op(
            input_shape=q_pe_shape,
            op_name="Mul",
            dtype="DT_FLOAT32",
            fusion_id=fusion_id,
            description=f"{descript_prefix}apply_rotary_emb",
            name="q_pe = apply_rotary_emb(q_pe, freqs_cis)",
            count=count,
        )
    )
    fusion_id += 1

    # kv = self.wkv_a(x)
    # kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    # k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
    ops.append(
        create_einsum_op(
            input_a_shape=h_t_shape,
            input_b_shape=wkv_a_shape,
            einsum_expr="BSM;MK->BSK",
            dtype="DT_BFLOAT16",
            fusion_id=fusion_id,
            description=f"{descript_prefix}kv_k_pe_linear",
            name="kv = self.wkv_a(x); kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)",
            count=count,
        )
    )
    fusion_id += 1
    ops.append(
        create_elementwise_binary_op(
            input_shape=k_pe_shape,
            op_name="Mul",
            dtype="DT_FLOAT32",
            fusion_id=fusion_id,
            description=f"{descript_prefix}apply_rotary_emb",
            name="k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)",
            count=count,
        )
    )
    fusion_id += 1

    # sequence parallelism: allgather the kv and pe cache before proceeding
    # assume SP uses the TP axes (which shards the attention heads)
    if tensor_parallelism_degree > 1:
        ops.append(
            create_all_gather_op(
                input_shape=kv_shape,
                parallelism_axes=tensor_parallelism_axes,
                axes_bandwidth=[config.ici_bw_GBps] * len(tensor_parallelism_axes),
                gather_dim=1,  # gather along seqlen dim
                config=config,
                fusion_id_start=fusion_id,
                dtype="DT_FLOAT8",
                count=count,
                description=f"AllGatherKV-{fusion_id}"
            )
        )
        fusion_id += 1
        ops.append(
            create_all_gather_op(
                input_shape=k_pe_shape,
                parallelism_axes=tensor_parallelism_axes,
                axes_bandwidth=[config.ici_bw_GBps] * len(tensor_parallelism_axes),
                gather_dim=1,  # gather along seqlen dim
                config=config,
                fusion_id_start=fusion_id,
                dtype="DT_FLOAT32",
                count=count,
                description=f"AllGatherK_PE-{fusion_id}"
            )
        )
        fusion_id += 1

    # wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
    # q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
    ops.append(
        create_einsum_op(
            input_a_shape=q_nope_shape,  # [batch_size, seqlen, num_local_heads, config.qk_nope_head_dim]
            input_b_shape=[num_local_heads, config.qk_nope_head_dim, config.kv_lora_rank],  # wkv_b[:, :self.qk_nope_head_dim]
            einsum_expr="BSHD;HDC->BSHC",
            dtype="DT_BFLOAT16",
            fusion_id=fusion_id,
            description=f"{descript_prefix}q_nope_wkv_b_linear",
            name="wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank); q_nope = torch.einsum('bshd,hdc->bshc', q_nope, wkv_b[:, :self.qk_nope_head_dim])",
            count=count,
        )
    )
    fusion_id += 1

    # self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
    # self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)
    ops.append(
        create_unary_op(
            input_shape=kv_shape,
            op_name="RMSNorm",
            dtype="DT_FLOAT32",
            fusion_id=fusion_id,
            description=f"{descript_prefix}kv_norm",
            name="kv_norm = self.kv_norm(kv)",
            count=count,
        )
    )
    fusion_id += 1

    # scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
    #             torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, num_local_heads, config.kv_lora_rank],
            input_b_shape=(
                [batch_size, config.input_seqlen + config.output_seqlen, config.kv_lora_rank]
                if is_decode else
                [batch_size, config.input_seqlen, config.kv_lora_rank]
            ),
            einsum_expr="BSHC;BTC->BSHT",
            dtype="DT_BFLOAT16",
            fusion_id=fusion_id,
            description=f"{descript_prefix}q_nope_kv_cache_einsum",
            name="q_nope_kv = torch.einsum('bshc,btc->bsht', q_nope, self.kv_cache[:bsz, :end_pos])",
            count=count,
        )
    )
    fusion_id += 1
    ops.append(
        create_einsum_op(
            input_a_shape=q_pe_shape,  # [batch_size, seqlen, num_local_heads, config.qk_rope_head_dim]
            input_b_shape=(
                [batch_size, config.input_seqlen + config.output_seqlen, config.qk_rope_head_dim]
                if is_decode else
                [batch_size, config.input_seqlen, config.qk_rope_head_dim]
            ),
            einsum_expr="BSHR;BTR->BSHT",
            dtype="DT_BFLOAT16",
            fusion_id=fusion_id,
            description=f"{descript_prefix}q_pe_pe_einsum",
            name="q_pe_pe = torch.einsum('bshr,btr->bsht', q_pe, self.pe_cache[:bsz, :end_pos])",
            count=count,
        )
    )
    fusion_id += 1

    # x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
    ops.append(
        create_einsum_op(
            input_a_shape=(
                [batch_size, seqlen, num_local_heads, config.input_seqlen + config.output_seqlen]
                if is_decode else
                [batch_size, seqlen, num_local_heads, config.input_seqlen]
            ),
            input_b_shape=(
                [batch_size, config.input_seqlen + config.output_seqlen, config.kv_lora_rank]
                if is_decode else
                [batch_size, config.input_seqlen, config.kv_lora_rank]
            ),
            einsum_expr="BSHT;BTC->BSHC",
            dtype="DT_BFLOAT16",
            fusion_id=fusion_id,
            description=f"{descript_prefix}scores_kv_cache_einsum",
            name="x = torch.einsum('bsht,btc->bshc', scores, self.kv_cache[:bsz, :end_pos])",
            count=count,
        )
    )
    fusion_id += 1

    # x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, num_local_heads, config.kv_lora_rank],
            input_b_shape=[num_local_heads, config.v_head_dim, config.kv_lora_rank],  # wkv_b[:, -self.v_head_dim:]
            einsum_expr="BSHC;HDC->BSHD",
            dtype="DT_BFLOAT16",
            fusion_id=fusion_id,
            description=f"{descript_prefix}x_wkv_b_linear",
            name="x = torch.einsum('bshc,hdc->bshd', x, wkv_b[:, -self.v_head_dim:])",
            count=count,
        )
    )
    fusion_id += 1

    # x = self.wo(x.flatten(2))
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, num_local_heads * config.v_head_dim],
            input_b_shape=wo_shape,  # [num_local_heads * config.v_head_dim, d_model]
            einsum_expr="BSD;DM->BSM",
            dtype="DT_BFLOAT16",
            fusion_id=fusion_id,
            description=f"{descript_prefix}x_wo_linear",
            name="x = self.wo(x.flatten(2))",
            count=count,
        )
    )
    fusion_id += 1

    #RS
    if tensor_parallelism_degree > 1:
        rs_op = create_reduce_scatter_op(
            input_shape=[batch_size, seqlen, d_model],
            parallelism_axes=tensor_parallelism_axes,
            axes_bandwidth=[config.ici_bw_GBps] * len(tensor_parallelism_axes),
            reduction_dim=2,
            config=config,
            fusion_id_start=fusion_id,
            count=count,
            description=f"Reduce attention and scatter before RMSNorm",
            name=f"ReduceScatterMHA-{fusion_id}"
        )
        ops.append(rs_op)
        fusion_id += 1
        rs_dim = rs_op.output_tensors[0].shape

    ops.append(
        create_unary_op(
            input_shape=[batch_size, seqlen, d_model],
            op_name="RMSNorm",
            dtype=dtype,
            name="Y_norm = RMSNorm(y)",
            description=(descript_prefix + "Attention_RMSNorm"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    fusion_id += 1

    #AG
    if tensor_parallelism_degree > 1:
        assert rs_dim
        ag_op = create_all_gather_op(
            input_shape=rs_dim,
            parallelism_axes=tensor_parallelism_axes,
            axes_bandwidth=[config.ici_bw_GBps] * len(tensor_parallelism_axes),
            gather_dim=2,
            config=config,
            fusion_id_start=fusion_id,
            count=count,
            description=f"AllGatherMHA-{fusion_id}"
        )
        ops.append(ag_op)
        fusion_id += 1

    return ops


def create_multi_head_attention_bwd(
    batch_size: int,
    input_seqlen: int,
    output_seqlen: int,
    decode_width: int,
    num_heads: int,
    d_model: int,
    d_head: int,
    num_layers: int,
    config: ModelConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    is_decode: bool = False,
    description_prefix: str = "",
    type: str = "self-attention",
    q_seqlen: int | None = None,
    kv_seqlen: int | None = None,
    d_query: int | None = None,
    d_key: int | None = None,
    d_value: int | None = None,
    use_flash_attention: bool = False,
    tensor_parallelism_axes: Sequence[int] = [1],
    ici_bw_GBps: float = 900.0
) -> list[Operator]:
    '''
    IMPORTANT NOTE: All parallelized attention ops must be created by calling this function.
    It acts as a wrapper to preprocess the tensor dimensions using parallelism degrees before
    passing to attention ops (create_multihead_self/cross_attention), which assume input
    dimensions have already been parallelized.
    '''

    #Preprocess input dimensions
    tensor_parallelism_degree = int(np.prod(tensor_parallelism_axes))
    num_heads = ceil(num_heads / tensor_parallelism_degree)
    if d_model is not None:
        d_model_parallel = ceil(d_model / tensor_parallelism_degree)
    if d_query is not None:
        d_query = ceil(d_query / tensor_parallelism_degree)

    if not is_decode and use_flash_attention:
        # use flash attention for prefill
        # call the cross attention function with the same q and kv seqlen
        # makes this a self attention

    #     return create_multi_head_cross_attention(
    #         batch_size,
    #         q_seqlen or input_seqlen,
    #         kv_seqlen or input_seqlen,
    #         num_heads,
    #         d_query or d_model_parallel,
    #         d_key or d_model,
    #         d_value or d_model,
    #         d_head,
    #         num_layers,
    #         dtype,
    #         fusion_id_start,
    #         description_prefix,
    #         use_flash_attention,
    #         tensor_parallelism_degree= tensor_parallelism_degree,
    #         ici_bw_GBps= ici_bw_GBps
    #     )
        raise ValueError(f"Not yet supported attention type: flash-attn-{type}")
    elif type == "self-attention":
        return create_multi_head_self_attention_bwd(
            batch_size,
            input_seqlen,
            output_seqlen,
            decode_width,
            num_heads,
            d_model_parallel,
            d_head,
            num_layers,
            config,
            dtype,
            fusion_id_start,
            description_prefix,
            tensor_parallelism_axes=tensor_parallelism_axes,
            ici_bw_GBps=ici_bw_GBps
        )
    elif type == "cross-attention":
    #     return create_multi_head_cross_attention_bwd(
    #         batch_size,
    #         q_seqlen or input_seqlen,
    #         kv_seqlen or input_seqlen,
    #         num_heads,
    #         d_query or d_model_parallel,
    #         d_key or d_model,
    #         d_value or d_model,
    #         d_head,
    #         num_layers,
    #         dtype,
    #         fusion_id_start,
    #         description_prefix,
    #         use_flash_attention,
    #         tensor_parallelism_degree = tensor_parallelism_degree,
    #         ici_bw_GBps = ici_bw_GBps
    #     )
        raise ValueError(f"Not yet supported attention type: {type}")
    else:
        raise ValueError(f"Unsupported attention type: {type}")


def create_multi_head_attention(
    batch_size: int,
    input_seqlen: int,
    output_seqlen: int,
    decode_width: int,
    num_heads: int,
    d_model: int,
    d_head: int,
    num_layers: int,
    config: ModelConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    is_decode: bool = False,
    description_prefix: str = "",
    type: str = "self-attention",
    q_seqlen: int | None = None,
    kv_seqlen: int | None = None,
    d_query: int | None = None,
    d_key: int | None = None,
    d_value: int | None = None,
    use_flash_attention: bool = False,
    tensor_parallelism_axes: Sequence[int] = [1],
    ici_bw_GBps: float = 900.0,
) -> list[Operator]:
    '''
    IMPORTANT NOTE: All parallelized attention ops must be created by calling this function.
    It acts as a wrapper to preprocess the tensor dimensions using parallelism degrees before
    passing to attention ops (create_multihead_self/cross_attention), which assume input
    dimensions have already been parallelized.
    '''

    # Preprocess input dimensions
    tensor_parallelism_degree = int(np.prod(tensor_parallelism_axes))
    num_heads = ceil(num_heads / tensor_parallelism_degree)
    if d_model is not None:
        d_model_parallel = ceil(d_model / tensor_parallelism_degree)
    if d_query is not None:
        d_query = ceil(d_query / tensor_parallelism_degree)

    if not is_decode and use_flash_attention:
        # use flash attention for prefill
        # call the cross attention function with the same q and kv seqlen
        # makes this a self attention
        return create_multi_head_cross_attention(
            batch_size,
            q_seqlen or input_seqlen,
            kv_seqlen or input_seqlen,
            num_heads,
            d_query or d_model_parallel,
            d_key or d_model,
            d_value or d_model,
            d_head,
            config,
            num_layers,
            dtype,
            fusion_id_start,
            description_prefix,
            use_flash_attention,
            tensor_parallelism_axes=tensor_parallelism_axes,
            ici_bw_GBps=ici_bw_GBps
        )
    elif type == "self-attention":
        return create_multi_head_self_attention(
            batch_size,
            input_seqlen,
            output_seqlen,
            decode_width,
            num_heads,
            d_model_parallel,
            d_head,
            num_layers,
            config,
            dtype,
            fusion_id_start,
            is_decode,
            description_prefix,
            tensor_parallelism_axes=tensor_parallelism_axes,
            ici_bw_GBps = ici_bw_GBps
        )
    elif type == "cross-attention":
        return create_multi_head_cross_attention(
            batch_size,
            q_seqlen or input_seqlen,
            kv_seqlen or input_seqlen,
            num_heads,
            d_query or d_model_parallel,
            d_key or d_model,
            d_value or d_model,
            d_head,
            config,
            num_layers,
            dtype,
            fusion_id_start,
            description_prefix,
            use_flash_attention,
            tensor_parallelism_axes=tensor_parallelism_axes,
            ici_bw_GBps=ici_bw_GBps
        )
    else:
        raise ValueError(f"Unsupported attention type: {type}")


def create_multi_head_normal_attention_block(
    batch_size: int,
    q_seqlen: int,
    kv_seqlen: int,
    num_heads: int,
    d_head: int,
    num_layers: int = 1,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description_prefix: str = "",
) -> list[Operator]:
    ops: list[Operator] = []
    fusion_id = fusion_id_start

    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, q_seqlen, num_heads, d_head],
            input_b_shape=[batch_size, kv_seqlen, num_heads, d_head],
            einsum_expr="BLND;BSND->BLSN",
            dtype=dtype,
            memory_placement=[0, 0, 1],
            name="MatMul: attnWeights = Q*K",
            description=(description_prefix + f"-Attention_Softmax(Q*K)*V-{fusion_id}"),
            count=num_layers,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = 0      # No parameters used here.

    ops.append(
        create_unary_op(
            input_shape=[batch_size, q_seqlen, kv_seqlen, num_heads],
            op_name="Softmax",
            dtype=dtype,
            memory_placement=[1, 1],
            name="attnWeights = Softmax(attnWeights)",
            description=(description_prefix + f"-Attention_Softmax(Q*K)*V-{fusion_id}"),
            count=num_layers,
            fusion_id=fusion_id,
        )
    )
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, q_seqlen, kv_seqlen, num_heads],
            input_b_shape=[batch_size, kv_seqlen, num_heads, d_head],
            einsum_expr="BLSN;BSND->BLND",
            dtype=dtype,
            memory_placement=[1, 0, 0],
            name="MatMul: attnAvg = attnWeights * V",
            description=(description_prefix + f"-Attention_Softmax(Q*K)*V-{fusion_id}"),
            count=num_layers,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = 0      # No parameters used here.

    ops.append(
        create_elementwise_binary_op(
            input_shape=[batch_size, q_seqlen, num_heads, d_head],
            op_name="Add",
            dtype=dtype,
            name="attn_avg = Add(([B/d][L/l][M/mh] + [B/d][L/l][M/mh]))",
            description=(description_prefix + f"-Attention_Softmax(Q*K)*V-{fusion_id}"),
            count=num_layers,
            fusion_id=fusion_id,
        )
    )

    return ops


def create_multi_head_flash_attention_op(
    Q_shape: Sequence[int],
    K_shape: Sequence[int],
    V_shape: Sequence[int],
    dtype: str = "DT_BFLOAT16",
    memory_placement: Sequence[int] = [0, 0, 0],
    fusion_id: int = 0,
    description: str = "FlashAttention",
    name: str = "FlashAttention",
    count: int = 1,
) -> Operator:
    final_op = FlashAttentionOperator()

    def get_shape_str(shape: Sequence[int]) -> str:
        return "x".join([str(i) for i in shape])

    final_op.fusion_id = fusion_id
    final_op.description = description
    final_op.config_str = f"FlashAttention(q={get_shape_str(Q_shape)},k={get_shape_str(K_shape)},v={get_shape_str(V_shape)},memory_placements={memory_placement},type={dtype})"
    final_op.op_type = OpType.MXU
    final_op.stats.count = count
    final_op.stats.flop_count = get_flops_flash_attention(Q_shape, K_shape, V_shape)

    final_op.input_tensors += [
        Tensor.from_shape("Q", Q_shape, dtype=dtype),
        Tensor.from_shape("K", K_shape, dtype=dtype),
        Tensor.from_shape("V", V_shape, dtype=dtype),
    ]
    final_op.input_tensor_shape_str = format_input_tensor_shapes([Q_shape, K_shape, V_shape], dtype)
    output_shape = get_flash_attention_output_shape(Q_shape, K_shape, V_shape)
    final_op.output_tensors.append(
        Tensor.from_shape("attn_avg", output_shape, dtype=dtype)
    )
    final_op.output_tensor_shape_str = format_output_tensor_shape(output_shape, dtype)

    final_op.name = format_op_name(name, description)
    final_op.opcode = "FlashAttention"
    final_op.opcode_type = OpcodeType.FLASH_ATTENTION

    return final_op


def create_multi_head_flash_attention_block(
    batch_size: int,
    q_seqlen: int,
    kv_seqlen: int,
    num_heads: int,
    d_head: int,
    num_layers: int = 1,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description_prefix: str = "",
) -> list[Operator]:
    ops = [
        create_multi_head_flash_attention_op(
            Q_shape=[batch_size, q_seqlen, num_heads, d_head],
            K_shape=[batch_size, kv_seqlen, num_heads, d_head],
            V_shape=[batch_size, kv_seqlen, num_heads, d_head],
            dtype=dtype,
            fusion_id=fusion_id_start,
            description=(description_prefix + f"-FlashAttention-{fusion_id_start}"),
            count=num_layers,
        )
    ]
    return ops

# Generates operations from perspective of single TPU.
# Accepts parameters as though there is no tensor parallelism
# Handles "splitting" internally
# If d is data parallelism factor and t is tensor parallelism factor:

# batch size = b/d
#...
def create_multi_head_cross_attention(
    batch_size: int,
    q_seqlen: int,
    kv_seqlen: int,
    num_heads: int,
    d_query: int,
    d_key: int,
    d_value: int,
    d_head: int,
    config: ModelConfig,
    num_layers: int = 1,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description_prefix: str = "",
    use_flash_attention: bool = False,
    tensor_parallelism_axes: Sequence[int] = [1],
    ici_bw_GBps: float = 900.0
) -> list[Operator]:
    ops: list[Operator] = []
    fusion_id = fusion_id_start

    ag_dim = None
    # If using tensor parallelism, need to perform an all-gather here. Each TPU takes B,S,D/t to B,S,D
    tensor_parallelism_degree = int(np.prod(tensor_parallelism_axes))
    if tensor_parallelism_degree > 1:
        ag_op = create_all_gather_op(
            input_shape=[batch_size, q_seqlen, d_query],
            parallelism_axes=tensor_parallelism_axes,
            axes_bandwidth=[ici_bw_GBps] * len(tensor_parallelism_axes),
            gather_dim=2,
            config=config,
            fusion_id_start=fusion_id,
            count=num_layers,
            description="Concatenate dims before attn",
            name=f"AllGatherMHA-{fusion_id}"
        )
        batch_size, q_seqlen, d_query = ag_op.output_tensors[0].shape
        ops.append(ag_op)
        fusion_id += 1

    # Q/K/V linear projections
    ops.append(
        create_einsum_op(
            input_a_shape= [batch_size, q_seqlen, d_query],
            input_b_shape= [d_query, num_heads, d_head],
            einsum_expr="BLM;MND->BLND",
            dtype=dtype,
            name="MatMul: Q",
            description=(description_prefix + f"-Q-{fusion_id}"),
            count=num_layers,
            fusion_id=fusion_id,
        )
    )
    # ops[-1]["Weight Size"] = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_query, d_head]))
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, kv_seqlen, d_key],
            input_b_shape=[d_key, num_heads, d_head],
            einsum_expr="BLM;MND->BLND",
            dtype=dtype,
            name="MatMul: K",
            description=(description_prefix + f"-K-{fusion_id}"),
            count=num_layers,
            fusion_id=fusion_id,
        )
    )
    # ops[-1]["Weight Size"] = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_key, d_head]))
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, kv_seqlen, d_value],
            input_b_shape=[d_value, num_heads, d_head],
            einsum_expr="BLM;MND->BLND",
            dtype=dtype,
            name="MatMul: V",
            description=(description_prefix + f"-V-{fusion_id}"),
            count=num_layers,
            fusion_id=fusion_id,
        )
    )
    # ops[-1]["Weight Size"] = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_value, d_head]))
    fusion_id += 1

    # attention
    if use_flash_attention:
        ops += create_multi_head_flash_attention_block(
            batch_size=batch_size,
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
            num_heads=num_heads,
            d_head=d_head,
            num_layers=num_layers,
            dtype=dtype,
            fusion_id_start=fusion_id,
            description_prefix=description_prefix,
        )
    else:
        ops += create_multi_head_normal_attention_block(
            batch_size=batch_size,
            q_seqlen=q_seqlen,
            kv_seqlen=kv_seqlen,
            num_heads=num_heads,
            d_head=d_head,
            num_layers=num_layers,
            dtype=dtype,
            fusion_id_start=fusion_id,
            description_prefix=description_prefix,
        )
    fusion_id = ops[-1].fusion_id + 1

    # output linear projection
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, q_seqlen, num_heads, d_head],
            input_b_shape=[num_heads, d_head, d_query],
            einsum_expr="BLND;NDM->BLM",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: attnOutput = attnAvg * W_o",
            description=(description_prefix + f"-Attention_output-{fusion_id}"),
            count=num_layers,
            fusion_id=fusion_id,
        )
    )
    fusion_id += 1
    # ops[-1]["Weight Size"] = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_head, d_query]))

    rs_dim = None
    # All Reduce if using tensor parallelism
    if tensor_parallelism_degree > 1:
        rs_op = create_reduce_scatter_op(
            input_shape=[batch_size, q_seqlen, d_query],
            parallelism_axes=tensor_parallelism_axes,
            axes_bandwidth=[ici_bw_GBps] * len(tensor_parallelism_axes),
            reduction_dim=2,
            config=config,
            fusion_id_start=fusion_id,
            count=num_layers,
            name=f"AllReduce(ReduceScatter)-{fusion_id}",
            description=f"Reduce and split attention results"
        )
        rs_dim = rs_op.output_tensors[0].shape     # d_query gets split
        ops.append(rs_op)
        fusion_id += 1
        # rs_dim = rs_op["Output Shape"]

    # final LayerNorm
    ops.append(
        create_unary_op(
            input_shape= rs_dim if rs_dim else [batch_size, q_seqlen, d_query], # [batch_size, q_seqlen, d_query]
            op_name="LayerNorm",
            dtype=dtype,
            name="Y_norm = LayerNorm(y)",
            description=(description_prefix + f"-Attention_layernorm-{fusion_id}"),
            fusion_id=fusion_id,
            count=num_layers,
        )
    )

    #AG after LN
    if tensor_parallelism_degree > 1:
        assert rs_dim
        ag_op = create_all_gather_op(
            input_shape=rs_dim,
            parallelism_axes=tensor_parallelism_axes,
            axes_bandwidth=[ici_bw_GBps] * len(tensor_parallelism_axes),
            gather_dim=2,
            config=config,
            fusion_id_start=fusion_id,
            count=num_layers,
            description=f"Gather results after Layernorm",
            name=f"AllReduce(AllGather)-{fusion_id}"
        )
        ops.append(ag_op)
        fusion_id += 1

    return ops


def create_multi_head_gq_attention_bwd(
    batch_size: int,
    input_seqlen: int,
    output_seqlen: int,
    decode_width: int,
    num_heads: int,
    d_model: int,
    d_head: int,
    num_layers: int,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    is_decode: bool = False,
    description_prefix: str = "",
    tensor_parallelism_degree: int = 1,
    ici_bw_GBps: float = 900.0
) -> list[Operator]:

    num_kv_heads = 8 # TODO this is constant for our purposes but make an input anyways

    ops: list[Operator] = []
    fusion_id = fusion_id_start
    if is_decode:  # decode
        count = num_layers * output_seqlen
        seqlen = decode_width
        descript_prefix = description_prefix + "Attention-serving-decode-"
    else:  # prefill
        count = num_layers
        seqlen = input_seqlen
        descript_prefix = description_prefix + "Fwd-Attention_encoder-"

    # output layer norm
    ops.extend(
        create_layernorm_op_bwd(
            input_shape=[batch_size, seqlen, num_heads, d_model],
            op_name="LayerNorm",
            dtype=dtype,
            name="Y_norm = LayerNorm(y)",
            description=(descript_prefix + "Attention_layernorm"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    # einsum combining attention scores w/ tensor
    ops.extend(
        create_einsum_op_bwd(
            input_a_shape=[batch_size, num_kv_heads, num_heads, input_seqlen, input_seqlen],
            input_b_shape=[batch_size, num_heads, input_seqlen, d_head],
            einsum_expr="BGHNS,BHSD->BGHND",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: similarity scores",
            description=(descript_prefix + "Attention_output"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = 0
    # softmax
    ops.extend(
        create_softmax_op_bwd(
            input_shape=[batch_size, num_kv_heads, num_heads, input_seqlen, input_seqlen], # TODO what is n vs s
            op_name="Softmax",
            dtype=dtype,
            name="attnWeights = Softmax(attnWeights)",
            description=(descript_prefix + "Softmax(Q*K)*V"),
            count=count,
            fusion_id=fusion_id,
        )
    )

    # compute similarity scores
    ops.extend(
        create_einsum_op_bwd(
            input_a_shape=[batch_size, num_kv_heads, num_heads, input_seqlen, d_head],
            input_b_shape=[batch_size, num_heads, input_seqlen, d_head],
            einsum_expr="BGHND,BHSD->BGHNS",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: similarity scores",
            description=(descript_prefix + "Attention_output"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = 0

    return ops

def create_multi_head_gq_attention(
    batch_size: int,
    input_seqlen: int,
    output_seqlen: int,
    decode_width: int,
    num_heads: int,
    d_model: int,
    d_head: int,
    num_layers: int,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    is_decode: bool = False,
    description_prefix: str = "",
    tensor_parallelism_degree: int = 1,
    ici_bw_GBps: float = 900.0
) -> list[Operator]:
    ops: list[Operator] = []
    fusion_id = fusion_id_start
    if is_decode:  # decode
        count = num_layers * output_seqlen
        seqlen = decode_width
        descript_prefix = description_prefix + "Attention-serving-decode-"
    else:  # prefill
        count = num_layers
        seqlen = input_seqlen
        descript_prefix = description_prefix + "Fwd-Attention_encoder-"

    # scaled dot product gqa (this is single head)
    # [
    # Einstein notation:
    # - b: batch size
    # - n / s: sequence length
    # - h: number of heads
    # - g: number of groups
    num_kv_heads = 8
    # - d: dimension of query/key/value
    # # compute similarity (big einsum)
    # similarity = einsum(query, key, "b g h n d, b h s d -> b g h n s")
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, num_kv_heads, num_heads, input_seqlen, d_head],
            input_b_shape=[batch_size, num_heads, input_seqlen, d_head],
            einsum_expr="BGHND,BHSD->BGHNS",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: similarity scores",
            description=(descript_prefix + "Attention_output"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = 0
    # ops[-1]["Weight Size"] = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_head, d_model]))
    fusion_id = ops[-1].fusion_id + 1
    mask = None
    if is_decode: # decode (causal)
        # mask out ones
        pass
    # extend mask based on inputs (both decode, prefill)

    # softmax
    ops.append(
        create_unary_op(
            input_shape=[batch_size, num_kv_heads, num_heads, input_seqlen, input_seqlen], # TODO what is n vs s
            op_name="Softmax",
            dtype=dtype,
            name="attnWeights = Softmax(attnWeights)",
            description=(descript_prefix + "Softmax(Q*K)*V"),
            count=count,
            fusion_id=fusion_id,
        )
    )

    # dropout (Nop? )

    # einsum (combine attn w tensor)
    # out = einsum(attention, value, "b g h n s, b h s d -> b g h n d")
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, num_kv_heads, num_heads, input_seqlen, input_seqlen],
            input_b_shape=[batch_size, num_heads, input_seqlen, d_head],
            einsum_expr="BGHNS,BHSD->BGHND",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: similarity scores",
            description=(descript_prefix + "Attention_output"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = 0

    # ]

    # layer norm
    # BNHD
    ops.append(
        create_unary_op(
            input_shape=[batch_size, seqlen, num_heads, d_model],
            op_name="LayerNorm",
            dtype=dtype,
            name="Y_norm = LayerNorm(y)",
            description=(descript_prefix + "Attention_layernorm"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    # output projection (noP? )

    return ops


def create_multi_head_self_attention_bwd(
    batch_size: int,
    input_seqlen: int,
    output_seqlen: int,
    decode_width: int,
    num_heads: int,
    d_model: int,
    d_head: int,
    num_layers: int,
    config: ModelConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description_prefix: str = "",
    tensor_parallelism_axes: Sequence[int] = [1],
    ici_bw_GBps: float = 900.0,
) -> list[Operator]:
    ops: list[Operator] = []
    fusion_id = fusion_id_start
    count = num_layers
    seqlen = input_seqlen
    tensor_parallelism_degree = int(np.prod(tensor_parallelism_axes))
    descript_prefix = description_prefix + "Bwd-Attention_encoder-"

    ops.extend(
        create_layernorm_op_bwd(
            input_shape=[batch_size, seqlen, d_model],
            op_name="LayerNorm",
            dtype=dtype,
            name="Y_norm = LayerNorm(y)",
            description=(descript_prefix + "Attention_layernorm"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    fusion_id = ops[-1].fusion_id + 1

    ops.extend(
        create_einsum_op_bwd(
            input_a_shape=[batch_size, seqlen, num_heads, d_head],
            input_b_shape=[num_heads, d_head, d_model],
            einsum_expr="BLND;NDM->BLM",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: attnOutput (2) = attnAvg (2) * W_o (1)",
            description=(descript_prefix + "Attention_output"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    # TODO should I be overwriting?
    # ops[-1]["Weight Size"] = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_head, d_model]))
    fusion_id = ops[-1].fusion_id + 1

    # TODO BW of elementwise add

    ops.extend(
        create_einsum_op_bwd(
            input_a_shape=[batch_size, input_seqlen, input_seqlen, num_heads],
            input_b_shape=[batch_size, input_seqlen, num_heads, d_head],
            einsum_expr="BLSN;BSND->BLND",
            dtype=dtype,
            memory_placement=[1, 0, 0],
            name="MatMul: attnAvg (2) = attnWeights (2) * V_all_gathered (2)",
            description=(descript_prefix + "Attention_Softmax(Q*K)*V"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-2].stats.weight_size_bytes = 0 # No weights for (QK)*V
    ops[-1].stats.weight_size_bytes = 0

    ops.extend(
        create_softmax_op_bwd(
            input_shape=[batch_size, input_seqlen, input_seqlen, num_heads],
            op_name="Softmax",
            dtype=dtype,
            memory_placement=[1, 1],
            name="attnWeights = Softmax(attnWeights)",
            description=(descript_prefix + "Attention_Softmax(Q*K)*V"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = 0 # TODO is this even necessary? -> since it's a softmax...

    ops.extend(
        create_einsum_op_bwd(
            input_a_shape=[batch_size, input_seqlen, num_heads, d_head],
            input_b_shape=[batch_size, input_seqlen, num_heads, d_head],
            einsum_expr="BLND;BSND->BLSN", # L = S
            dtype=dtype,
            memory_placement=[0, 0, 1],
            name="MatMul: attnWeights (2) = Q (2) * K_all_gathered (2)",
            description=(descript_prefix + "Attention_Softmax(Q*K)*V"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-2].stats.weight_size_bytes = 0
    ops[-1].stats.weight_size_bytes = 0

    # Q, K, V via Wq, Wk, Wv -> all @ same fusion idx
    ops.extend(
        create_einsum_op_bwd(
            input_a_shape=[batch_size, seqlen, d_model],
            input_b_shape=[d_model, num_heads, d_head],
            einsum_expr="BLM;MND->BLND",
            dtype=dtype,
            name="MatMul: Q (2) = x_norm_all_gathered (2) * W_q (1)",
            description=(descript_prefix + "Q/K/V"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-2].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_model, d_head]))
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_model, d_head]))

    ops.extend(
        create_einsum_op_bwd(
            input_a_shape=[batch_size, input_seqlen, d_model],
            input_b_shape=[d_model, num_heads, d_head],
            einsum_expr="BLM;MND->BLND",
            dtype=dtype,
            name="MatMul: K (2) = x_norm_all_gathered (2) * W_k (1)",
            description=(descript_prefix + "Q/K/V"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-2].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_model, d_head]))
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_model, d_head]))

    ops.extend(
        create_einsum_op_bwd(
            input_a_shape=[batch_size, input_seqlen, d_model],
            input_b_shape=[d_model, num_heads, d_head],
            einsum_expr="BLM;MND->BLND",
            dtype=dtype,
            name="MatMul: V (2) = x_norm_all_gathered (2) * W_v (1)",
            description=(descript_prefix + "Q/K/V"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-2].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_model, d_head]))
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_model, d_head]))

    fusion_id = ops[-1].fusion_id + 1

    if tensor_parallelism_degree > 1:
        ops.append(
            create_all_reduce_op(
                input_shape=[batch_size, input_seqlen, d_model],
                parallelism_axes=tensor_parallelism_axes,
                axes_bandwidth=[ici_bw_GBps] * len(tensor_parallelism_axes),
                config=config,
                dtype=dtype,
                fusion_id_start=fusion_id,
                count=1,
                description=descript_prefix,
            )
        )

    return ops


def create_multi_head_self_attention(
    batch_size: int,
    input_seqlen: int,
    output_seqlen: int,
    decode_width: int,
    num_heads: int,
    d_model: int,
    d_head: int,
    num_layers: int,
    config: ModelConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    is_decode: bool = False,
    description_prefix: str = "",
    tensor_parallelism_axes: Sequence[int] = [1],
    ici_bw_GBps: float = 900.0
) -> list[Operator]:
    ops: list[Operator] = []
    fusion_id = fusion_id_start
    if is_decode:  # decode
        count = num_layers * output_seqlen
        seqlen = decode_width
        descript_prefix = description_prefix + "Attention-serving-decode-"
    else:  # prefill
        count = num_layers
        seqlen = input_seqlen
        descript_prefix = description_prefix + "Fwd-Attention_encoder-"

    rs_dim = None
    ag_dim = None
    tensor_parallelism_degree = int(np.prod(tensor_parallelism_axes))
    if tensor_parallelism_degree > 1:
        ag_op = create_all_gather_op(
            input_shape=[batch_size, seqlen, d_model],
            parallelism_axes=tensor_parallelism_axes,
            axes_bandwidth=[ici_bw_GBps] * len(tensor_parallelism_axes),
            gather_dim=2,
            config=config,
            fusion_id_start=fusion_id,
            count=count,
            description=f"AllGatherMHA-{fusion_id}"
        )
        batch_size, seqlen, d_model = ag_op.output_tensors[0].shape  # B, S, D/t -> AllGather = B, S, D
        ops.append(ag_op)
        fusion_id += 1

    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, d_model],
            input_b_shape=[d_model, num_heads, d_head],
            einsum_expr="BLM;MND->BLND",
            dtype=dtype,
            name="MatMul: Q (2) = x_norm_all_gathered (2) * W_q (1)",
            description=(descript_prefix + "Q/K/V"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_model, d_head]))
    if is_decode:  # use KV cache for decode stage
        ops.append(
            create_einsum_op(
                input_a_shape=[batch_size, decode_width, d_model],
                input_b_shape=[2, d_model, num_heads, d_head],
                einsum_expr="BLM;TMND->BTLND",
                dtype=dtype,
                name="MatMul: KV (2) = x_norm (2) * W_kv (1)",
                description=(descript_prefix + "Q/K/V"),
                count=count,
                fusion_id=fusion_id,
            )
        )
        ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([2, num_heads, d_model, d_head]))
    else:  # prefill
        ops.append(
            create_einsum_op(
                input_a_shape=[batch_size, input_seqlen, d_model],
                input_b_shape=[d_model, num_heads, d_head],
                einsum_expr="BLM;MND->BLND",
                dtype=dtype,
                name="MatMul: K (2) = x_norm_all_gathered (2) * W_k (1)",
                description=(descript_prefix + "Q/K/V"),
                count=count,
                fusion_id=fusion_id,
            )
        )
        ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_model, d_head]))
        ops.append(
            create_einsum_op(
                input_a_shape=[batch_size, input_seqlen, d_model],
                input_b_shape=[d_model, num_heads, d_head],
                einsum_expr="BLM;MND->BLND",
                dtype=dtype,
                name="MatMul: V (2) = x_norm_all_gathered (2) * W_v (1)",
                description=(descript_prefix + "Q/K/V"),
                count=count,
                fusion_id=fusion_id,
            )
        )
        ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_model, d_head]))
    fusion_id += 1
    if is_decode:  # use KV cache for decode stage
        ops.append(
            create_einsum_op(
                input_a_shape=[batch_size, decode_width, num_heads, d_head],
                input_b_shape=[batch_size, input_seqlen, num_heads, d_head],
                einsum_expr="BLND;BSND->BLSN",
                dtype=dtype,
                name="MatMul: QK_prefix (2) = Q (2) * Kcache (2)",
                description=(descript_prefix + "Softmax(Q*K)*V"),
                count=count,
                fusion_id=fusion_id,
            )
        )
        ops[-1].stats.weight_size_bytes = 0
        ops.append(
            create_einsum_op(
                input_a_shape=[batch_size, decode_width, num_heads, d_head],
                input_b_shape=[batch_size, output_seqlen, num_heads, d_head],
                einsum_expr="BLND;BSND->BLSN",
                dtype=dtype,
                name="MatMul: QK_suffix (2) = Q (2) * Ksuffix (2)",
                description=(descript_prefix + "Softmax(Q*K)*V"),
                count=count,
                fusion_id=fusion_id,
            )
        )
        ops[-1].stats.weight_size_bytes = 0
        ops.append(
            create_unary_op(
                input_shape=[batch_size, decode_width, input_seqlen + output_seqlen, num_heads],
                op_name="Softmax",
                dtype=dtype,
                name="attnWeights = Softmax(attnWeights)",
                description=(descript_prefix + "Softmax(Q*K)*V"),
                count=count,
                fusion_id=fusion_id,
            )
        )
        ops[-1].stats.weight_size_bytes = 0
        ops.append(
            create_einsum_op(
                input_a_shape=[batch_size, decode_width, input_seqlen, num_heads],
                input_b_shape=[batch_size, input_seqlen, num_heads, d_head],
                einsum_expr="BLSN;BSND->BLND",
                dtype=dtype,
                name="MatMul: attn_avg_prefix (2) = QK_prefix (2) * Vcache (2)",
                description=(descript_prefix + "Softmax(Q*K)*V"),
                count=count,
                fusion_id=fusion_id,
            )
        )
        ops[-1].stats.weight_size_bytes = 0
        ops.append(
            create_einsum_op(
                input_a_shape=[batch_size, decode_width, output_seqlen, num_heads],
                input_b_shape=[batch_size, output_seqlen, num_heads, d_head],
                einsum_expr="BLSN;BSND->BLND",
                dtype=dtype,
                name="MatMul: attn_avg_suffix (2) = QK_suffix (2) * Vsuffix (2)",
                description=(descript_prefix + "Softmax(Q*K)*V"),
                count=count,
                fusion_id=fusion_id,
            )
        )
        ops[-1].stats.weight_size_bytes = 0
    else:  # prefill
        ops.append(
            create_einsum_op(
                input_a_shape=[batch_size, input_seqlen, num_heads, d_head],
                input_b_shape=[batch_size, input_seqlen, num_heads, d_head],
                einsum_expr="BLND;BSND->BLSN",
                dtype=dtype,
                memory_placement=[0, 0, 1],
                name="MatMul: attnWeights (2) = Q (2) * K_all_gathered (2)",
                description=(descript_prefix + "Attention_Softmax(Q*K)*V"),
                count=count,
                fusion_id=fusion_id,
            )
        )
        ops[-1].stats.weight_size_bytes = 0
        ops.append(
            create_unary_op(
                input_shape=[batch_size, input_seqlen, input_seqlen, num_heads],
                op_name="Softmax",
                dtype=dtype,
                memory_placement=[1, 1],
                name="attnWeights = Softmax(attnWeights)",
                description=(descript_prefix + "Attention_Softmax(Q*K)*V"),
                count=count,
                fusion_id=fusion_id,
            )
        )
        ops[-1].stats.weight_size_bytes = 0
        ops.append(
            create_einsum_op(
                input_a_shape=[batch_size, input_seqlen, input_seqlen, num_heads],
                input_b_shape=[batch_size, input_seqlen, num_heads, d_head],
                einsum_expr="BLSN;BSND->BLND",
                dtype=dtype,
                memory_placement=[1, 0, 0],
                name="MatMul: attnAvg (2) = attnWeights (2) * V_all_gathered (2)",
                description=(descript_prefix + "Attention_Softmax(Q*K)*V"),
                count=count,
                fusion_id=fusion_id,
            )
        )
        ops[-1].stats.weight_size_bytes = 0
    ops.append(
        create_elementwise_binary_op(
            input_shape=[batch_size, seqlen, num_heads, d_head],
            op_name="Add",
            dtype=dtype,
            name=(
                "attn_avg = Add(([B/d]beamS[N/h]D + [B/d]beamS[N/h]D))"
                if is_decode else
                "attn_avg = Add(([B/d][L/l][M/mh] + [B/d][L/l][M/mh]))"
            ),
            description=(
                (descript_prefix + "Softmax(Q*K)*V")
                if is_decode else
                (descript_prefix + "Attention_Softmax(Q*K)*V")
            ),
            count=count,
            fusion_id=fusion_id,
        )
    )
    fusion_id += 1
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, num_heads, d_head],
            input_b_shape=[num_heads, d_head, d_model],
            einsum_expr="BLND;NDM->BLM",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: attnOutput (2) = attnAvg (2) * W_o (1)",
            description=(descript_prefix + "Attention_output"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    # ops[-1]["Weight Size"] = util.get_size_bytes_from_dtype(dtype) * int(np.prod([num_heads, d_head, d_model]))
    fusion_id += 1

    #RS
    if tensor_parallelism_degree > 1:
        rs_op = create_reduce_scatter_op(
            input_shape=[batch_size, seqlen, d_model],
            parallelism_axes=tensor_parallelism_axes,
            axes_bandwidth=[ici_bw_GBps] * len(tensor_parallelism_axes),
            reduction_dim=2,
            config=config,
            fusion_id_start=fusion_id,
            count=count,
            description=f"Reduce attention and scatter before LN",
            name=f"AllReduceMHA(ReduceScatter)-{fusion_id}"
        )
        ops.append(rs_op)
        fusion_id += 1
        rs_dim = rs_op.output_tensors[0].shape

    ops.append(
        create_unary_op(
            input_shape=[batch_size, seqlen, d_model],
            op_name="LayerNorm",
            dtype=dtype,
            name="Y_norm = LayerNorm(y)",
            description=(descript_prefix + "Attention_layernorm"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    fusion_id += 1

    #AG
    if tensor_parallelism_degree > 1:
        assert rs_dim
        ag_op = create_all_gather_op(
            input_shape=rs_dim,
            parallelism_axes=tensor_parallelism_axes,
            axes_bandwidth=[ici_bw_GBps] * len(tensor_parallelism_axes),
            gather_dim=2,
            config=config,
            fusion_id_start=fusion_id,
            count=count,
            description=f"AllGatherMHA-{fusion_id}"
        )
        ops.append(ag_op)

    return ops


def create_ffn_matmul_default(
    batch_size: int,
    seqlen: int,
    d_model: int,
    d_ff: int,
    count: int,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    is_decode: bool = False,
    description_prefix: str = "",
    tensor_parallelism_degree = 1,
) -> list[Operator]:
    ops: list[Operator] = []
    fusion_id = fusion_id_start
    if is_decode:  # decode
        descript_prefix = description_prefix + "Fwd-FFN-serving-decoder-"
    else:  # prefill
        descript_prefix = description_prefix + "Fwd-FFN_encoder-"

    # FFi
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, d_model],
            input_b_shape=[d_model, d_ff],
            einsum_expr="BLM;MH->BLH",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: h (2) = y_norm (2) * W_FFi (1)",
            description=(descript_prefix + "FFinput"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([d_model, d_ff])) # Compute weights in bytes.
    fusion_id += 1

    # FFo
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, d_ff],
            input_b_shape=[d_ff, d_model],
            einsum_expr="BLH;HM->BLM",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: ffn (2) = h (2) * W_FFo (1)",
            description=(descript_prefix + "FFinput"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([d_model, d_ff]))

    return ops


def create_ffn_matmul_llama_bwd(
    batch_size: int,
    seqlen: int,
    d_model: int,
    d_ff: int,
    count: int,
    tensor_parallelism_axes: Sequence[int],
    ici_bandwidth: float,
    config: ModelConfig,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description_prefix: str = "",
) -> list[Operator]:
    ops: list[Operator] = []
    fusion_id = fusion_id_start
    descript_prefix = description_prefix + "Bwd-FFN-encoder-"
    tensor_parallelism_degree = int(np.prod(tensor_parallelism_axes))

    # Down
    ops.extend(
        create_einsum_op_bwd(
            input_a_shape=[batch_size, seqlen, d_ff],
            input_b_shape=[d_ff, d_model],
            einsum_expr="BLH;HM->BLM",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            # name="MatMul: ff_down (2) = h_gate_up (2) * W_FFdown (1)",
            description=(descript_prefix + "FFdown"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([d_model, d_ff]))

    # Gate * Up -> TODO we don't have a bwd implemented for elementwise mul but I think should be similar to fwd?
    ops.append(
        create_elementwise_binary_op(
            input_shape=[batch_size, seqlen, d_ff],
            op_name="Mul",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="h_gate_up (2) = h_gate (2) * h_up (1)",
            description=(descript_prefix + "FFgate_up"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    fusion_id += 1

    # Gate
    ops.extend(
        create_einsum_op_bwd(
            input_a_shape=[batch_size, seqlen, d_model],
            input_b_shape=[d_model, d_ff],
            einsum_expr="BLM;MH->BLH",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            # name="MatMul: ff_down (2) = h_gate_up (2) * W_FFdown (1)",
            description=(descript_prefix + "FFgate"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([d_model, d_ff]))

    # Up
    ops.extend(
        create_einsum_op_bwd(
            input_a_shape=[batch_size, seqlen, d_model],
            input_b_shape=[d_model, d_ff],
            einsum_expr="BLM;MH->BLH",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            # name="MatMul: ff_down (2) = h_gate_up (2) * W_FFdown (1)",
            description=(descript_prefix + "FFup"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([d_model, d_ff]))

    if tensor_parallelism_degree > 1:
        ops.append(
            create_all_reduce_op(
                input_shape=[batch_size, seqlen, d_model],
                parallelism_axes=tensor_parallelism_axes,
                axes_bandwidth=[ici_bandwidth] * len(tensor_parallelism_axes),
                config=config,
                fusion_id_start=fusion_id,
                count=1,
                description=descript_prefix,
            )
        )

    return ops


def create_ffn_matmul_llama(
    batch_size: int,
    seqlen: int,
    d_model: int,
    d_ff: int,
    count: int,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    is_decode: bool = False,
    description_prefix: str = "",
) -> list[Operator]:
    ops: list[Operator] = []
    fusion_id = fusion_id_start
    if is_decode:  # decode
        descript_prefix = description_prefix + "Fwd-FFN-serving-decoder-"
    else:  # prefill
        descript_prefix = description_prefix + "Fwd-FFN-encoder-"

    # Gate
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, d_model],
            input_b_shape=[d_model, d_ff],
            einsum_expr="BLM;MH->BLH",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: h_gate (2) = y_norm (2) * W_FFgate (1)",
            description=(descript_prefix + "FFgate"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([d_model, d_ff]))
    fusion_id += 1
    # Up
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, d_model],
            input_b_shape=[d_model, d_ff],
            einsum_expr="BLM;MH->BLH",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: h_up (2) = y_norm (2) * W_FFup (1)",
            description=(descript_prefix + "FFup"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([d_model, d_ff]))
    fusion_id += 1
    # Gate * Up
    ops.append(
        create_elementwise_binary_op(
            input_shape=[batch_size, seqlen, d_ff],
            op_name="Mul",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="h_gate_up (2) = h_gate (2) * h_up (1)",
            description=(descript_prefix + "FFgate_up"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    fusion_id += 1
    # Down
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, d_ff],
            input_b_shape=[d_ff, d_model],
            einsum_expr="BLH;HM->BLM",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="MatMul: ff_down (2) = h_gate_up (2) * W_FFdown (1)",
            description=(descript_prefix + "FFoutput"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    ops[-1].stats.weight_size_bytes = util.get_size_bytes_from_dtype(dtype) * int(np.prod([d_model, d_ff]))

    return ops


def create_ffn_bwd(
    batch_size: int,
    input_seqlen: int,
    output_seqlen: int,
    decode_width: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    config: ModelConfig,
    ffn_type: str = "default",
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    is_decode: bool = False,
    description_prefix: str = "",
    tensor_parallelism_axes: Sequence[int] = [1],
    ici_bw_GBps: float = 900.0
) -> list[Operator]:
    ops: list[Operator] = []
    fusion_id = fusion_id_start
    tensor_parallelism_degree = int(np.prod(tensor_parallelism_axes))

    d_ff = ceil(d_ff / tensor_parallelism_degree)

    count = num_layers
    seqlen = input_seqlen
    descript_prefix = description_prefix + "Bwd-FFN-encoder-"

    # MLP
    if ffn_type == "default":
        # ops += create_ffn_matmul_default_bwd(
            # batch_size, seqlen, d_model, d_ff, count, dtype, fusion_id, is_decode, description_prefix
        # )
        raise ValueError(f"Not yet supported ffn_type: {ffn_type}")
    elif ffn_type == "llama":
        ops += create_ffn_matmul_llama_bwd(
            batch_size,
            seqlen,
            d_model,
            d_ff,
            count,
            tensor_parallelism_axes,
            ici_bw_GBps,
            config,
            dtype,
            fusion_id,
            description_prefix,
        )
    else:
        raise ValueError(f"Unsupported ffn_type: {ffn_type}")

    # All reduce
    if tensor_parallelism_degree > 1:
        ar_op = create_all_reduce_op(
            input_shape= [batch_size, seqlen, d_model],
            parallelism_axes=tensor_parallelism_axes,
            axes_bandwidth=[ici_bw_GBps] * len(tensor_parallelism_axes),
            config=config,
            fusion_id_start=fusion_id,
            count=count,
            name=f"AllReduce-MLP-{fusion_id}",
            description=f"All reduce results of MLP gradient calculation"
        )
        ops.append(ar_op)
    fusion_id += 1

    # TOD residual connection might actually be needed -> let's see
    # fusion_id = ops[-1]["Fusion index"] + 1

    return ops


def create_ffn_deepseek_moe_gate(
    batch_size: int,
    seqlen: int,
    config: ModelConfig,
    count: int,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description_prefix: str = "",
) -> list[Operator]:
    assert isinstance(config, DeepSeekConfig), \
        f"Config must be an instance of DeepSeekConfig. Got {type(config)} instead."

    ops: list[Operator] = []
    fusion_id = fusion_id_start
    description_prefix += "FFN_MoE_gate-"

    # scores = linear(x, self.weight)
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, config.d_model],
            input_b_shape=[config.num_routed_experts, config.d_model],
            einsum_expr="BSM;RM->BSR",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            name="scores = linear(x, self.weight)",
            description=(description_prefix + "scores"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    fusion_id += 1

    # scores = scores.softmax(dim=-1, dtype=torch.float32)
    ops.append(
        create_unary_op(
            input_shape=[batch_size, seqlen, config.num_routed_experts],
            op_name="Softmax",
            dtype="DT_FLOAT32",
            count=count,
            fusion_id=fusion_id,
            name="scores = softmax(scores)",
            description=(description_prefix + "scores_softmax"),
        )
    )
    fusion_id += 1

    # TODO: add topK selection operator;
    #       ignored for now as it is not the performance bottleneck.

    return ops


def create_ffn_deepseek_moe(
    batch_size: int,
    seqlen: int,
    config: ModelConfig,
    count: int,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    is_decode: bool = False,
    description_prefix: str = "",
    tensor_parallelism_axes: Sequence[int] = [1],
    expert_parallelism_axes: Sequence[int] = [1],
) -> list[Operator]:
    '''
    @tensor_parallelism_axes: Axes for expert tensor parallelism.
    @expert_parallelism_axes: Axes for expert parallelism.
    '''
    assert isinstance(config, DeepSeekConfig), \
        f"Config must be an instance of DeepSeekConfig. Got {type(config)} instead."

    ops: list[Operator] = []
    fusion_id = fusion_id_start
    tensor_parallelism_degree = int(np.prod(tensor_parallelism_axes))
    expert_parallelism_degree = int(np.prod(expert_parallelism_axes))

    d_ff = ceil(config.moe_d_ff / tensor_parallelism_degree)

    # Gating function
    ops += create_ffn_deepseek_moe_gate(
        batch_size=batch_size,
        seqlen=seqlen,
        config=config,
        count=count,
        dtype=dtype,
        fusion_id_start=fusion_id,
        description_prefix=description_prefix,
    )
    fusion_id = ops[-1].fusion_id + 1

    # All to All token dispatch
    bisection_bw = None
    if expert_parallelism_degree > 1:
        bisection_bw, ici_topology = get_bisection_bw_per_chip_GBps(config)
        ops.append(
            create_all_to_all_op(
                input=Tensor.from_shape(
                    name="moe_dispath_tokens",
                    shape=[
                        config.global_batch_size,
                        seqlen,
                        config.num_limited_groups,
                        config.d_model,
                    ],
                    dtype="DT_FLOAT8",
                ),
                config=config,
                bisection_bw=bisection_bw,
                num_parallelism=expert_parallelism_degree,
                dtype="DT_FLOAT8",
                fusion_id=fusion_id,
                name="MoE_Dispatch_AllToAll",
                description=(description_prefix + "MoE_Dispatch_AllToAll"),
                count=count,
            )
        )
        fusion_id += 1

    # MoE expert computation
    for expert_id in range(config.num_activated_routed_experts_per_token):
        ops += create_ffn_matmul_llama(
            batch_size=batch_size,
            # TODO: currently assumes worst case where all tokens are routed to
            # the same experts on the same device.
            # May want to account for token distribution in the future.
            seqlen=seqlen,
            d_model=config.d_model,
            d_ff=d_ff,
            count=count,
            dtype=dtype,
            fusion_id_start=fusion_id,
            is_decode=is_decode,
            description_prefix=(description_prefix + f"FFN_routed_expert{expert_id}-"),
        )
        fusion_id = ops[-1].fusion_id + 1

    # All to All token combine
    if expert_parallelism_degree > 1:
        assert bisection_bw is not None, \
            "Bisection bandwidth should be set when expert parallelism degree > 1."
        ops.append(
            create_all_to_all_op(
                input=Tensor.from_shape(
                    name="moe_combine_tokens",
                    shape=[
                        config.global_batch_size,
                        seqlen,
                        config.num_limited_groups,
                        config.d_model,
                    ],
                    dtype="DT_FLOAT8",
                ),
                config=config,
                bisection_bw=bisection_bw,
                num_parallelism=expert_parallelism_degree,
                dtype="DT_FLOAT8",
                fusion_id=fusion_id,
                name="MoE_Dispatch_AllToAll",
                description=(description_prefix + "MoE_Dispatch_AllToAll"),
                count=count,
            )
        )
        fusion_id += 1

    # shared expert computation
    ops += create_ffn_matmul_llama(
        batch_size=batch_size,
        seqlen=seqlen,
        d_model=config.d_model,
        d_ff=(d_ff * config.num_shared_experts),
        count=count,
        dtype=dtype,
        fusion_id_start=fusion_id,
        is_decode=is_decode,
        description_prefix=(description_prefix + "FFN_shared_expert-"),
    )
    fusion_id = ops[-1].fusion_id + 1

    # token combine weights
    ops.append(
        create_einsum_op(
            input_a_shape=[batch_size, seqlen, config.num_activated_routed_experts_per_token, config.d_model],
            input_b_shape=[batch_size, seqlen, config.num_activated_routed_experts_per_token],
            einsum_expr="BSEM;BSE->BSM",
            dtype=dtype,
            name="MoE combine weights",
            count=count,
            fusion_id=fusion_id,
            description=(description_prefix + "MoE_combine_weights"),
        )
    )
    fusion_id += 1

    # combine with shared expert
    ops.append(
        create_elementwise_binary_op(
            input_shape=[batch_size, seqlen, config.d_model],
            op_name="Add",
            dtype=dtype,
            name="MoE combine output",
            description=(description_prefix + "MoE_combine_output"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    fusion_id += 1

    return ops


def create_ffn(
    batch_size: int,
    input_seqlen: int,
    output_seqlen: int,
    decode_width: int,
    d_model: int,
    d_ff: int,
    num_layers: int,
    config: ModelConfig,
    ffn_type: str = "default",
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    is_decode: bool = False,
    description_prefix: str = "",
    tensor_parallelism_axes: Sequence[int] = [1],
    ici_bw_GBps: float = 900.0,  # unused
    expert_parallelism_axes: Sequence[int] = [1],  # for MoE models only
) -> list[Operator]:

    '''
    d_ff and d_model should be specified according to the model specifications
    without regard for tensor parallelism.
    '''

    ops: list[Operator] = []
    fusion_id = fusion_id_start
    tensor_parallelism_degree = int(np.prod(tensor_parallelism_axes))

    d_ff = ceil(d_ff / tensor_parallelism_degree)

    if is_decode:  # decode
        count = num_layers * output_seqlen
        seqlen = decode_width
        descript_prefix = description_prefix + "FFN-serving-decoder-"
    else:  # prefill
        count = num_layers
        seqlen = input_seqlen
        descript_prefix = description_prefix + "Fwd-FFN-encoder-"

    if tensor_parallelism_degree > 1:
        ag_op = create_all_gather_op(
            input_shape=[batch_size, d_model],
            parallelism_axes=tensor_parallelism_axes,
            axes_bandwidth=[config.ici_bw_GBps] * len(tensor_parallelism_axes),
            gather_dim=1,
            config=config,
            fusion_id_start=fusion_id,
            count=count,
            name=f"AllGatherMLP-{fusion_id}",
            description=f"All gather input of MLP before ffn"
        )
        ops.append(ag_op)
        fusion_id += 1

    # MLP
    if ffn_type == "default":
        ops += create_ffn_matmul_default(
            batch_size, seqlen, d_model, d_ff, count, dtype, fusion_id, is_decode, description_prefix
        )
    elif ffn_type == "llama":
        ops += create_ffn_matmul_llama(
            batch_size, seqlen, d_model, d_ff, count, dtype, fusion_id, is_decode, description_prefix
        )
    elif ffn_type == "deepseek_moe":
        ops += create_ffn_deepseek_moe(
            batch_size, seqlen, config, count, dtype, fusion_id, is_decode, description_prefix,
            tensor_parallelism_axes, expert_parallelism_axes,
        )
    else:
        raise ValueError(f"Unsupported ffn_type: {ffn_type}")
    fusion_id = ops[-1].fusion_id + 1

    if tensor_parallelism_degree > 1:
        rs_op = create_reduce_scatter_op(
            input_shape=[batch_size, seqlen, d_model],
            parallelism_axes= tensor_parallelism_axes,
            axes_bandwidth=[config.ici_bw_GBps] * len(tensor_parallelism_axes),
            reduction_dim=2,
            config=config,
            fusion_id_start=fusion_id,
            count=count,
            name=f"ReduceScatterMLP-{fusion_id}",
            description=f"Reduce and scatter results of MLP before residual"
        )
        batch_size, seqlen, d_model = rs_op.output_tensors[0].shape
        ops.append(rs_op)
        fusion_id += 1

    # residual connection
    ops.append(
        create_elementwise_binary_op(
            input_shape=[batch_size, seqlen, d_model],
            op_name="Add",
            name=(
                "attnPlusFFn = Add(([B/d]beamS[M/mh] + [B/d]beamS[M/mh]))"
                if is_decode else
                "attnPlusFFn = Add(([B/d][L/l][M/mh] + [B/d][L/l][M/mh]))"
            ),
            description=(descript_prefix + "AttnPlusFFn"),
            count=count,
            fusion_id=fusion_id,
        )
    )
    fusion_id += 1

    return ops


def create_conv2d(
    batch_size: int,
    input_channel: int,
    input_spatial_shape: Sequence[int],
    output_channel: int,
    kernel_spatial_shape: Sequence[int],
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    num_layers: int = 1,
    dtype: str = "DT_BFLOAT16",
    fusion_id_start: int = 0,
    description_prefix: str = "",
) -> list[Operator]:
    '''
    Create ops for a conv2d layer.
    @param batch_size: Batch size.
    @param input_channel: Number of input channels.
    @param input_spatial_shape: Spatial shape of input (H, W).
    @param output_channel: Number of output channels.
    @param kernel_spatial_shape: Spatial shape of kernel (H, W).
    @param stride: Stride size.
    @param padding: Padding size.
    @param dilation: Dilation size.
    @param num_layers: Number of layers.
    @param dtype: Data type.
    @param fusion_id_start: Starting fusion index.
    '''
    input_shape = [batch_size, input_channel, *input_spatial_shape]
    kernel_shape = [input_channel, output_channel, *kernel_spatial_shape]

    def compute_output_spatial_dim_size(in_dim, padding, dilation, kernel, stride) -> int:
        return floor((in_dim + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)
    ## see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    output_spatial_shape = [
        compute_output_spatial_dim_size(input_spatial_shape[0], padding, dilation, kernel_spatial_shape[0], stride),
        compute_output_spatial_dim_size(input_spatial_shape[1], padding, dilation, kernel_spatial_shape[1], stride),
    ]
    output_shape = [batch_size, output_channel, *output_spatial_shape]
    ops = [
        create_conv2d_op(
            input_shape=input_shape,
            kernel_shape=kernel_shape,
            output_shape=output_shape,
            einsum_expr="bf01;io01->bf01",
            window="{" + f"size={kernel_spatial_shape[0]}x{kernel_spatial_shape[1]} stride={stride}x{stride} pad=0_{padding}x0_{padding}" + "}",
            dtype=dtype,
            memory_placement=[0, 0, 0],
            fusion_id=fusion_id_start,
            count=num_layers,
            description=(description_prefix + "Conv2d"),
        )
    ]

    return ops
