from math import ceil
import os
import unittest
from unittest.mock import MagicMock, patch
import neusim.npusim.backend.util as util_under_test
import neusim.npusim.frontend.Operator as Operator
import neusim.xla_hlo_parser.xla_hlo_structures as hlo_struct


class TestNPUSimBackendUtil(unittest.TestCase):
    def test_get_size_bytes_from_dtype(self):
        test_cases = [
            ("float32", 4),
            ("float16", 2),
            ("fp32", 4),
            ("fp16", 2),
            ("bfloat16", 2),
            ("bf16", 2),
            ("int8", 1),
            ("int16", 2),
            ("int32", 4),
            ("int64", 8),
            ("float64", 8),
            ("BOOL", 1),
            ("DT_INT", 4),
            ("DT_FLOAT", 4),
        ]
        for dtype, expected_size in test_cases:
            self.assertEqual(util_under_test.get_size_bytes_from_dtype(dtype), expected_size)

        # test unsupported dtype
        with self.assertRaises(ValueError):
            util_under_test.get_size_bytes_from_dtype("asdf")

    def test_get_factors(self):
        test_cases = [
            (12, [1, 2, 3, 4, 6, 12]),
            (15, [1, 3, 5, 15]),
            (28, [1, 2, 4, 7, 14, 28]),
            (1, [1]),
            (37, [1, 37]),  # prime number
        ]
        for n, expected_factors in test_cases:
            self.assertEqual(util_under_test.get_factors(n), expected_factors)

    def test_construct_hlo_instruction_from_node_cost(self):
        # Mock node_cost as an Operator object
        node_cost = MagicMock(spec=Operator.Operator)
        # BatchMatMul requires rank 3 inputs: [Batch, M, K] and [Batch, K, N] -> [Batch, M, N]
        node_cost.output_tensor_shape_str = "[f32:(1,16,64)]"
        node_cost.input_tensor_shape_str = "f32:[1,16,128],f32:[1,128,64]"
        node_cost.name = "test_op"
        node_cost.config_str = "BatchMatMul(window={size=3x3 stride=1x1}, dim_labels=bmn->bn)"
        
        # Test basic construction
        instruction = util_under_test.construct_hlo_instruction_from_node_cost(node_cost)
        
        self.assertIsInstance(instruction, hlo_struct.HLOInstruction)
        self.assertEqual(instruction.result.name, "test_op")
        self.assertEqual(instruction.result.type.shape, [1, 16, 64])
        self.assertEqual(len(instruction.operands), 2)
        self.assertEqual(instruction.operands[0].type.shape, [1, 16, 128])
        self.assertEqual(instruction.operands[1].type.shape, [1, 128, 64])
        self.assertEqual(instruction.opcode, "convolution")
        self.assertEqual(instruction.metadata["op_type"], "Einsum")

        # Test with dict input (should convert to Operator)
        node_cost_dict = {
            "output_tensor_shape_str": "[f32:(1,16,64)]",
            "input_tensor_shape_str": "f32:[1,16,128],f32:[1,128,64]",
            "name": "test_op_dict",
            "config_str": "TestOp()",
            "Fusion index": 0, "Description": "", "Config": "TestOp()", "Name": "test_op_dict", "OpType": "Other",
            "Count": 1, "Bounded-by": "", "Execution time": 0, "Compute time": 0, "Memory time": 0, 
            "ICI/NVLink time": 0, "ICI/NVLink outbound traffic": 0, "ICI/NVLink inbound traffic": 0,
            "Aggregated DCN time": 0, "PCIe time": 0, "MXU time": 0, "VPU time": 0, "Transpose time": 0,
            "Permute time": 0, "Bytes accessed": 0, "Input Tensor Shapes": "f32:[1,16,128],f32:[1,128,64]",
            "Output Tensor Shapes": "f32:(1,16,64)]", "FLOP Count": 0, "Op Name": "test_op_dict", "Op Code": "TestOp", "Weight Size": 0,
            "parsed_op_type": "Other"
        }
        with patch('neusim.npusim.frontend.Operator.Operator.from_csv_dict') as mock_from_csv:
            mock_op = MagicMock(spec=Operator.Operator)
            mock_op.output_tensor_shape_str = "[f32:(1,16,64)]"
            mock_op.input_tensor_shape_str = "f32:[1,16,128],f32:[1,128,64]"
            mock_op.name = "test_op_dict"
            mock_op.config_str = "TestOp()"
            mock_from_csv.return_value = mock_op
            
            instruction = util_under_test.construct_hlo_instruction_from_node_cost(node_cost_dict)
            self.assertEqual(instruction.result.name, "test_op_dict")

        # Test XlaEinsum
        node_cost_einsum = MagicMock(spec=Operator.Operator)
        node_cost_einsum.output_tensor_shape_str = "[f32:(1,1,16)]"
        node_cost_einsum.input_tensor_shape_str = "f32:[1,1,128],f32:[128,16]"
        node_cost_einsum.name = "einsum_op"
        node_cost_einsum.config_str = "XlaEinsum(eq=bmk;kn->bmn, window={size=3x3})" 
        
        instruction_einsum = util_under_test.construct_hlo_instruction_from_node_cost(node_cost_einsum)
        self.assertEqual(instruction_einsum.opcode, "convolution")
        self.assertEqual(instruction_einsum.metadata["op_type"], "Einsum")
        # Check if axes were parsed
        self.assertTrue(hasattr(instruction_einsum, "input_axes"))
        self.assertTrue(hasattr(instruction_einsum, "output_axes"))
        
        # Test Conv2D
        node_cost_conv = MagicMock(spec=Operator.Operator)
        node_cost_conv.output_tensor_shape_str = "[f32:(1,32,32,16)]"
        node_cost_conv.input_tensor_shape_str = "f32:[1,32,32,16]"
        node_cost_conv.name = "conv_op"
        node_cost_conv.config_str = "Conv2D(eq=b01f_01io->b01f, window={size=3x3 stride=1x1},)"
        
        instruction_conv = util_under_test.construct_hlo_instruction_from_node_cost(node_cost_conv)
        self.assertEqual(instruction_conv.opcode, "convolution")
        self.assertIn("dim_labels", instruction_conv.metadata)
        self.assertIn("window", instruction_conv.metadata)

    def test_construct_hlo_module_from_node_costs(self):
        node_costs = []
        for i in range(2):
            mock_op = MagicMock(spec=Operator.Operator)
            mock_op.output_tensor_shape_str = "[f32:(1,128)]"
            mock_op.input_tensor_shape_str = "f32:[1,128]" 
            mock_op.name = f"op_{i}"
            mock_op.config_str = "Op()"
            node_costs.append(mock_op)
            
        module = util_under_test.construct_hlo_module_from_node_costs(node_costs, "test_module")
        
        self.assertIsInstance(module, hlo_struct.HLOModule)
        self.assertEqual(module.name, "test_module")
        self.assertEqual(len(module.ENTRY.instructions), 2)
        self.assertEqual(module.ENTRY.instructions[0].result.name, "op_0")
        self.assertEqual(module.ENTRY.instructions[1].result.name, "op_1")

    def test_get_total_execution_time_ns_from_ops(self):
        mock_op1 = MagicMock()
        mock_op1.stats.execution_time_ns = 100
        mock_op2 = MagicMock()
        mock_op2.stats.execution_time_ns = 200
        
        ops = [mock_op1, mock_op2]
        total_time = util_under_test.get_total_execution_time_ns_from_ops(ops)
        self.assertEqual(total_time, 300)

    def test_calculate_bandwidths(self):
        # SA Bandwidth
        # (128 + 128) * 2 * 8 * 1.75 = 256 * 16 * 1.75 = 4096 * 1.75 = 7168
        sa_bw = util_under_test.calculate_sa_bandwidth_GBps(
            sa_input_width=128, sa_output_width=128,
            data_type_size_bytes=2, num_sa=8, freq_GHz=1.75
        )
        self.assertAlmostEqual(sa_bw, 7168.0)
        
        # VPU Bandwidth
        # 128 * 8 * 2 * 1.75 = 3584
        vpu_bw = util_under_test.calculate_vpu_bandwidth_GBps(
            n_lanes=128, n_sublanes=8, n_ports=2, freq_GHz=1.75
        )
        self.assertAlmostEqual(vpu_bw, 3584.0)

    def test_parse_input_tensor_shapes(self):
        input_str = "DT_FLOAT:[1,128],DT_INT:[128,64]"
        shapes, dtypes = util_under_test.parse_input_tensor_shapes(input_str)
        self.assertEqual(shapes, [[1, 128], [128, 64]])
        self.assertEqual(dtypes, ["DT_FLOAT", "DT_INT"])

        # Test single tensor
        input_str_single = "DT_FLOAT:[1,128]"
        shapes, dtypes = util_under_test.parse_input_tensor_shapes(input_str_single)
        self.assertEqual(shapes, [[1, 128]])
        self.assertEqual(dtypes, ["DT_FLOAT"])

    def test_parse_output_tensor_shapes(self):
        output_str = "[DT_FLOAT:(1,128)]"
        shapes, dtypes = util_under_test.parse_output_tensor_shapes(output_str)
        self.assertEqual(shapes, [[1, 128]])
        self.assertEqual(dtypes, ["DT_FLOAT"])

        output_str_multi = "[DT_FLOAT:(1,128),DT_INT:(128,64)]"
        shapes, dtypes = util_under_test.parse_output_tensor_shapes(output_str_multi)
        self.assertEqual(shapes, [[1, 128], [128, 64]])
        self.assertEqual(dtypes, ["DT_FLOAT", "DT_INT"])
