import unittest
from neusim.npusim.frontend.Operator import (
    Operator, EinsumOperator, Conv2DOperator, FlashAttentionOperator,
    OperatorStatistics, EinsumStatistics, FlashAttentionStatistics,
    Axis, Tensor, ComponentDVFSConfig, DVFSConfig, DVFSPolicy, OpcodeType, OpType,
    from_csv_dict, to_csv_dict
)

class TestOperator(unittest.TestCase):
    def test_enums(self):
        # DVFSPolicy
        self.assertEqual(DVFSPolicy.from_str("Ideal"), DVFSPolicy.IDEAL)
        self.assertEqual(DVFSPolicy.from_str(None), DVFSPolicy.NONE)
        self.assertEqual(DVFSPolicy.from_str(""), DVFSPolicy.NONE)
        self.assertEqual(DVFSPolicy.from_str("DVFSCNoPareto"), DVFSPolicy.DVFS_C_NO_PARETO)
        self.assertEqual(DVFSPolicy.from_str("CustomAll"), DVFSPolicy.CUSTOM_ALL)
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


    def test_dvfs_config_hash_covers_region_and_domain_mode(self):
        base = DVFSConfig()
        region = DVFSConfig(frequency_adjustment_interval_ns=2_000_000)
        domain = DVFSConfig(custom_compute_domain_mode="dom3")
        self.assertNotEqual(hash(base), hash(region))
        self.assertNotEqual(hash(base), hash(domain))

    def test_split_energy_aggregates_and_legacy_setters(self):
        stats = OperatorStatistics(
            static_energy_hbm_mc_J=1.0,
            static_energy_hbm_die_J=2.0,
            static_energy_hbm_io_J=3.0,
            static_energy_ici_mc_J=4.0,
            static_energy_ici_phy_J=5.0,
        )
        self.assertEqual(stats.static_energy_hbm_J, 6.0)
        self.assertEqual(stats.static_energy_hbm_phy_J, 5.0)
        self.assertEqual(stats.static_energy_ici_J, 9.0)

        stats.dynamic_energy_hbm_J = 7.0
        stats.dynamic_energy_ici_J = 8.0
        self.assertEqual(stats.dynamic_energy_hbm_mc_J, 7.0)
        self.assertEqual(stats.dynamic_energy_hbm_die_J, 0.0)
        self.assertEqual(stats.dynamic_energy_hbm_io_J, 0.0)
        self.assertEqual(stats.dynamic_energy_ici_mc_J, 8.0)
        self.assertEqual(stats.dynamic_energy_ici_phy_J, 0.0)

    def test_legacy_dvfs_assignment_and_csv_fallback(self):
        op = Operator()
        hbm = ComponentDVFSConfig(
            policy=DVFSPolicy.CUSTOM, voltage_V=0.55, frequency_GHz=1.1
        )
        ici = ComponentDVFSConfig(
            policy=DVFSPolicy.IDEAL, voltage_V=0.6, frequency_GHz=1.2
        )
        op.dvfs_hbm = hbm
        op.dvfs_ici = ici
        self.assertIs(op.dvfs_hbm_mc, hbm)
        self.assertIs(op.dvfs_ici_mc, ici)

        written = op.to_csv_dict()
        self.assertEqual(written["DVFS HBM Policy"], "Custom")
        self.assertEqual(written["DVFS HBM MC Policy"], "Custom")
        self.assertEqual(written["DVFS ICI Policy"], "Ideal")
        self.assertEqual(written["DVFS ICI MC Policy"], "Ideal")

        legacy = dict(written)
        for prefix in ("HBM MC", "HBM DIE", "HBM IO", "ICI MC", "ICI PHY"):
            for suffix in (
                "Policy",
                "Voltage (V)",
                "Frequency (GHz)",
                "Scaling Time (ns)",
                "Power Efficiency (%)",
            ):
                legacy.pop(f"DVFS {prefix} {suffix}", None)
        restored = from_csv_dict(legacy)
        self.assertEqual(restored.dvfs_hbm_mc.policy, DVFSPolicy.CUSTOM)
        self.assertEqual(restored.dvfs_hbm_mc.voltage_V, 0.55)
        self.assertEqual(restored.dvfs_ici_mc.policy, DVFSPolicy.IDEAL)
        self.assertEqual(restored.dvfs_ici_mc.frequency_GHz, 1.2)
        self.assertEqual(restored.dvfs_hbm_die.policy, DVFSPolicy.NONE)
        self.assertEqual(restored.dvfs_hbm_io.policy, DVFSPolicy.NONE)
        self.assertEqual(restored.dvfs_ici_phy.policy, DVFSPolicy.NONE)
        self.assertIsNone(restored.dvfs_hbm_die.frequency_GHz)
        self.assertIsNone(restored.dvfs_hbm_io.frequency_GHz)
        self.assertIsNone(restored.dvfs_ici_phy.frequency_GHz)
