from copy import deepcopy
from typing import Any

from absl import flags, logging
import numpy as np

from neusim.npusim.frontend.Operator import Operator, OpType
from neusim.npusim.backend import npusim_lib
from neusim.npusim.backend import util as npusim_util
from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.frontend import power_analysis_lib as power_lib


def analyze_operator_component_util(
    op: Operator, config: ChipConfig
) -> Operator:
    '''
    Fill out op.stats.flops_util and op.stats.hbm_bw_util.
    '''
    if op.op_type == OpType.MXU:
        op.stats.flops_util = op.stats.tflops_per_sec / config.peak_tflops_per_sec
    else:
        op.stats.flops_util = op.stats.tflops_per_sec / config.peak_VU_tflops_per_sec

    op.stats.hbm_bw_util = op.stats.hbm_bw_GBps / config.hbm_bw_GBps

    return op


def calculate_vmem_time_ns(
    op: Operator,
    mxu_time_ns: int,
    vpu_time_ns: int,
    memory_time_ns: int,
    config: ChipConfig
) -> float:
    '''
    Calculate the vmem time in nanoseconds for the given operator.
    '''

    if op.op_type == OpType.MXU:
        # For MXU ops
        sa_bandwidth = npusim_util.calculate_sa_bandwidth_GBps(
            sa_input_width = config.sa_dim,
            sa_output_width = config.sa_dim,
            data_type_size_bytes = 2,
            num_sa = config.num_sa,
            freq_GHz = config.freq_GHz,
        )
        # formula T_sram_utilization =  (T_sa * sa_bandwidth + T_hbm * HBM_size_GB) / vmem_bw_GBps
        t_mxu_vmem = (mxu_time_ns * sa_bandwidth + memory_time_ns * config.hbm_bw_GBps) / config.vmem_bw_GBps

        return t_mxu_vmem
    elif op.op_type == OpType.VPU:
        # For VPU ops
        vpu_bandwidth = npusim_util.calculate_vpu_bandwidth_GBps(
            n_lanes = 128,
            n_sublanes = 8,
            n_ports = config.num_vu_ports,
            freq_GHz = config.freq_GHz,
        )
        # formula T_sram_utilization =  (T_vpu * vpu_bandwidth + T_hbm * HBM_size_GB) / vmem_bw_GBps
        t_vpu_vmem = (vpu_time_ns * vpu_bandwidth + memory_time_ns * config.hbm_bw_GBps) / config.vmem_bw_GBps

        return t_vpu_vmem
    return 0


def fill_operators_execution_info(
    ops: list[Operator],
    config: ChipConfig,
    analyze_energy: bool = True,
) -> list[Operator]:
    '''
    Fill in the execution info (exe time, flops, bytes accessed, etc.) for each op.
    '''
    converted_ops = []

    # hlo_module = mem_util.construct_hlo_module_from_node_costs(node_costs)
    hlo_module = npusim_util.construct_hlo_module_from_node_costs(ops)

    for op in ops:
        I, converted_op = npusim_lib.parse_tensor_shapes_for_node_cost(op, hlo_module)

        mxu_time, vpu_time = npusim_lib.compute_node_cost_compute_time(
            I, converted_op, config # num_sa, num_vu, freq_GHz
        )
        bytes_accessed = npusim_lib.compute_bytes_accessed_from_vmem_size(
            I, converted_op, config # vmem_size_mb, hbm_bw_GBps, freq_GHz
        )
        compute_time = max(mxu_time, vpu_time)
        memory_time = max(
            int(np.ceil(bytes_accessed / (config.hbm_bw_GBps * 1024 * 1024 * 1024 / 1e9))),
            config.hbm_latency_ns,
        )
        if bytes_accessed == 0:
            memory_time = 0

        vmem_time = calculate_vmem_time_ns(converted_op, mxu_time, vpu_time, memory_time, config)
        print(f"Calculated vmem_time_ns: {vmem_time} for op: {converted_op.name}")

        ici_time = converted_op.stats.ici_time_ns
        exe_time = max(compute_time, ici_time, memory_time, vmem_time)
        print(f"Op: {converted_op.name}, mxu_time: {mxu_time}, vpu_time: {vpu_time}, compute_time: {compute_time}, memory_time: {memory_time}, ici_time: {ici_time}, vmem_time: {vmem_time}, exe_time: {exe_time}")
        converted_op.stats.sa_time_ns = mxu_time
        converted_op.stats.vu_time_ns = vpu_time
        # converted_op.stats.compute_time_ns = compute_time
        converted_op.stats.memory_time_ns = memory_time
        converted_op.stats.vmem_time_ns = int(vmem_time)
        converted_op.stats.execution_time_ns = int(exe_time)
        converted_op.stats.memory_traffic_bytes = bytes_accessed
        if compute_time == exe_time:
            converted_op.stats.bounded_by = "Compute"
        elif memory_time == exe_time and compute_time < exe_time:
            converted_op.stats.bounded_by = "Memory"
        elif vmem_time == exe_time and compute_time < exe_time:
            converted_op.stats.bounded_by = "Compute"
        else:
            converted_op.stats.bounded_by = "ICI/NVLink"
        # fill_additional_fields(converted_op)

        analyze_operator_component_util(converted_op, config)

        converted_ops.append(converted_op)

    if analyze_energy:
        for op in converted_ops:
            power_lib.analyze_operator_energy(
                op, config
            )

    return converted_ops
