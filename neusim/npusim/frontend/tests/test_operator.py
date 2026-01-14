import unittest
from neusim.npusim.frontend.Operator import (
    Operator, EinsumOperator, Conv2DOperator, FlashAttentionOperator,
    OperatorStatistics, EinsumStatistics, FlashAttentionStatistics,
    Axis, Tensor, DVFSPolicy, OpcodeType, OpType,
    from_csv_dict, to_csv_dict
)

class TestOperator(unittest.TestCase):
    def test_enums(self):
        # DVFSPolicy
        self.assertEqual(DVFSPolicy.from_str("Ideal"), DVFSPolicy.IDEAL)
        self.assertEqual(DVFSPolicy.from_str(None), DVFSPolicy.NONE)
        # Verify ValueError is raised for unknown strings, as per implicit behavior of Enum(value)
        with self.assertRaises(ValueError):
            DVFSPolicy.from_str("Unknown")

        # OpcodeType
        self.assertEqual(OpcodeType.from_opcode("Conv2D"), OpcodeType.CONV2D)
        self.assertEqual(OpcodeType.from_opcode("Einsum"), OpcodeType.EINSUM)
        self.assertEqual(OpcodeType.from_opcode("UnknownOp"), OpcodeType.OTHER)

        # OpType
        self.assertEqual(OpType.from_string("MXU"), OpType.MXU)
        self.assertEqual(OpType.from_string("Unknown"), OpType.OTHER)

    def test_axis(self):
        axis = Axis(name="ax", size=1024, parallelism=[2, 4], tile_size=64)
        self.assertEqual(axis.num_shards, 8)
        self.assertEqual(axis.shard_size, 128) # 1024 / 8
        self.assertEqual(axis.num_tiles, 2)    # 128 / 64

        # Default
        axis_def = Axis(size=100)
        self.assertEqual(axis_def.num_shards, 1)
        self.assertEqual(axis_def.tile_size, 100)

    def test_operator_statistics(self):
        stats = OperatorStatistics()
        stats.execution_time_ns = 1000
        stats.sa_time_ns = 500
        stats.vu_time_ns = 600
        
        self.assertEqual(stats.compute_time_ns, 600)
        
        stats.memory_traffic_bytes = 1024**3 # 1 GB
        # 1 GB / 1000 ns = 1 GB / 1e-6 s = 1e6 GB/s ? No.
        # 1 GB / 1 us.
        # hbm_bw_GBps = bytes / 1024^3 / time(ns) * 1e9
        # = 1 * 1e9 / 1000 = 1e6 GBps.
        self.assertAlmostEqual(stats.hbm_bw_GBps, 1000000.0)

        stats.static_energy_sa_J = 1.0
        stats.dynamic_energy_sa_J = 0.5
        self.assertEqual(stats.static_energy_J, 1.0) # others 0
        self.assertEqual(stats.dynamic_energy_J, 0.5)
        self.assertEqual(stats.total_energy_J, 1.5)
        # Power = Energy / Time = 1.5 / 1000ns * 1e9 = 1.5 * 1e6 W
        self.assertAlmostEqual(stats.total_power_W, 1.5e6)

    def test_operator_csv(self):
        op = Operator(name="test_op", opcode="Add")
        op.stats.execution_time_ns = 100
        op.stats.count = 5
        op.op_type = OpType.VPU
        
        csv_dict = op.to_csv_dict()
        self.assertEqual(csv_dict["Name"], "test_op")
        self.assertEqual(csv_dict["Op Code"], "Add")
        self.assertEqual(csv_dict["Execution time"], 100)
        self.assertEqual(csv_dict["OpType"], "VPU")
        
        # Round trip
        new_op = from_csv_dict(csv_dict)
        self.assertEqual(new_op.name, "test_op")
        self.assertEqual(new_op.stats.execution_time_ns, 100)
        self.assertEqual(new_op.op_type, OpType.VPU)

    def test_einsum_operator_csv(self):
        op = EinsumOperator(name="matmul", opcode="Einsum")
        op.stats.dim_labels_str = "mk,kn->mn"
        op.stats.parsed_op_type = "Einsum" # Helper to ensure correct type identification
        
        csv_dict = op.to_csv_dict()
        self.assertEqual(csv_dict["dim_labels"], "mk,kn->mn")
        
        # Round trip via factory
        # Factory relies on "parsed_op_type" or "Op Code"
        csv_dict["Op Code"] = "Einsum" 
        new_op = from_csv_dict(csv_dict)
        self.assertIsInstance(new_op, EinsumOperator)
        self.assertEqual(new_op.stats.dim_labels_str, "mk,kn->mn")

    def test_conv2d_operator_csv(self):
        op = Conv2DOperator(name="conv", opcode="Conv2D")
        op.stats.num_sa_ops = 100
        op.stats.parsed_op_type = "Conv2D"

        csv_dict = op.to_csv_dict()
        self.assertEqual(csv_dict["num_mxu_ops"], 100)
        
        csv_dict["Op Code"] = "Conv2D"
        new_op = from_csv_dict(csv_dict)
        self.assertIsInstance(new_op, Conv2DOperator)
        self.assertEqual(new_op.stats.num_sa_ops, 100) 

    def test_flash_attention_operator_csv(self):
        op = FlashAttentionOperator(name="fa", opcode="FlashAttention")
        op.stats.vu_softmax_time_ns = 50
        op.stats.parsed_op_type = "FlashAttention"

        csv_dict = op.to_csv_dict()
        self.assertEqual(csv_dict["vu_softmax_time_ns"], 50)
        
        csv_dict["Op Code"] = "FlashAttention"
        new_op = from_csv_dict(csv_dict)
        self.assertIsInstance(new_op, FlashAttentionOperator)
        self.assertEqual(new_op.stats.vu_softmax_time_ns, 50)

    def test_dvfs_csv_parsing(self):
        # Test that DVFS fields are parsed correctly
        op_dict = Operator().to_csv_dict()
        op_dict["DVFS SA Policy"] = "Ideal"
        op_dict["DVFS SA Voltage (V)"] = 0.8
        
        op = from_csv_dict(op_dict)
        self.assertEqual(op.dvfs_sa.policy, DVFSPolicy.IDEAL)
        self.assertEqual(op.dvfs_sa.voltage_V, 0.8)
