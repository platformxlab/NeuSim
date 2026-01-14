
import unittest
import os
import shutil
from neusim.npusim.frontend.query_results_helper_lib import (
    set_results_path,
    is_model_llm,
    is_model_llm_moe,
    is_model_dlrm,
    is_model_sd,
    get_pstr_from_pconfig,
    get_pconfig_from_pstr,
    get_stats_filepath,
    get_op_stats_filepath,
    get_stats,
    get_all_stats,
    get_all_op_stats,
    get_optimal_stats_for_max_num_chips,
    get_latency_metric_name_and_min_max,
    get_throughput_metric_name_and_min_max,
    get_energy_eff_metric_name_and_min_max,
    get_carbon_eff_metric_name_and_min_max,
    get_component_data_from_file,
    get_total_execution_time_from_file,
    get_num_chips,
    get_min_num_chips,
    get_min_num_chips,
    get_slo_stat,
    get_pareto_frontier,
)

class TestQueryResultsHelper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set the results path to the local assets directory for testing
        cls.test_assets_path = os.path.join(os.path.dirname(__file__), "assets/raw")
        set_results_path(cls.test_assets_path)

    def test_is_model_helpers(self):
        self.assertTrue(is_model_llm("llama3-8b"))
        self.assertTrue(is_model_llm("deepseekv3-671b"))
        self.assertFalse(is_model_llm("dlrm-s"))
        
        self.assertTrue(is_model_llm_moe("deepseekv2-236b"))
        self.assertFalse(is_model_llm_moe("llama3-8b"))

        self.assertTrue(is_model_dlrm("dlrm-s"))
        self.assertFalse(is_model_dlrm("llama3-8b"))

        self.assertTrue(is_model_sd("gligen"))
        self.assertTrue(is_model_sd("dit-xl"))
        self.assertFalse(is_model_sd("llama3-8b"))

    def test_get_pstr_from_pconfig(self):
        pconfig = {
            "dp": 1, "tp": 1, "pp": 1,
            "dp_dcn": 1, "tp_dcn": 1, "pp_dcn": 1,
            "bs": 1
        }
        pstr = get_pstr_from_pconfig(model="llama3-8b", **pconfig)
        self.assertEqual(pstr, "dp1-tp1-pp1-dpdcn1-tpdcn1-ppdcn1-bs1")

        # Test MoE config validation (missing ep/ep_dcn)
        with self.assertRaises(AssertionError):
             get_pstr_from_pconfig(model="deepseekv3-671b", **pconfig)

        pconfig_moe = pconfig.copy()
        pconfig_moe.update({"ep": 1, "ep_dcn": 1})
        pstr_moe = get_pstr_from_pconfig(model="deepseekv3-671b", **pconfig_moe)
        self.assertEqual(pstr_moe, "dp1-tp1-pp1-ep1-dpdcn1-tpdcn1-ppdcn1-epdcn1-bs1")

    def test_get_pconfig_from_pstr(self):
        pstr = "dp1-tp1-pp1-dpdcn1-tpdcn1-ppdcn1-bs1"
        config = get_pconfig_from_pstr(pstr)
        # Expected: (dp, tp, pp, ep, dp_dcn, tp_dcn, pp_dcn, ep_dcn, bs)
        # For dense, ep=1, ep_dcn=1
        self.assertEqual(config, (1, 1, 1, 1, 1, 1, 1, 1, 1))

        pstr_moe = "dp1-tp1-pp1-ep2-dpdcn1-tpdcn1-ppdcn1-epdcn1-bs1"
        config_moe = get_pconfig_from_pstr(pstr_moe)
        self.assertEqual(config_moe, (1, 1, 1, 2, 1, 1, 1, 1, 1))

    def test_get_stats_filepath(self):
        path = get_stats_filepath(
            model="llama3-8b", version="5p", workload="inference",
            dp=1, tp=1, pp=1, dp_dcn=1, tp_dcn=1, pp_dcn=1, batch_size=1,
            prefill_or_decode="decode"
        )
        expected_suffix = "llama3-8b/dp1-tp1-pp1-dpdcn1-tpdcn1-ppdcn1-bs1/inference-v5p_decode.json"
        self.assertTrue(path.endswith(expected_suffix))

    def test_get_all_stats(self):
        # Test LLM
        stats = get_all_stats(
            model="llama3-8b", version="5p", workload="inference",
            prefill_or_decode="decode", batch_size=1
        )
        # The key is (dp, tp, pp, ep, dp_dcn, pp_dcn, bs)
        # Note: ep defaults to 1 if not MoE
        key = (1, 1, 1, 1, 1, 1, 1)
        self.assertIn(key, stats)
        # Check a value from the loaded JSON (assuming sample content)
        self.assertIsInstance(stats[key], dict)

        # Test DLRM
        stats_dlrm = get_all_stats(
            model="dlrm-s", version="5p", workload="inference",
            batch_size=1
        )
        key_dlrm = (1, 1, 1, 1, 1, 1, 1)
        self.assertIn(key_dlrm, stats_dlrm)

    def test_get_all_op_stats(self):
        # Test getting CSV stats
        # read_json_with_csv=False returns list[dict] (rows of csv)
        op_stats = get_all_op_stats(
            model="llama3-8b", version="5p", workload="inference",
            prefill_or_decode="decode", batch_size=1
        )
        key = (1, 1, 1, 1, 1, 1, 1)
        self.assertIn(key, op_stats)
        self.assertIsInstance(op_stats[key], list)
        if len(op_stats[key]) > 0:
            self.assertIsInstance(op_stats[key][0], dict)

    def test_get_optimal_stats_for_max_num_chips(self):
        # Since we only have one data point for llama3-8b in assets, it should validly return that one.
        opt_stats = get_optimal_stats_for_max_num_chips(
            model="llama3-8b", version="5p", max_num_chips=1024,
            workload="inference", prefill_or_decode="decode",
            batch_size=1, perf_metric="TPOT_ms_request"
        )
        self.assertIsInstance(opt_stats, dict)
        # Verify it didn't return empty or fail

    def test_metric_name_helpers(self):
        # Latency
        name, min_max = get_latency_metric_name_and_min_max("llama3-8b", "inference", "prefill")
        self.assertEqual(name, "TTFT_sec")
        self.assertEqual(min_max, "min")
        
        name, min_max = get_latency_metric_name_and_min_max("dlrm-s", "inference")
        self.assertEqual(name, "latency_ns")

        # Throughput
        name, min_max = get_throughput_metric_name_and_min_max("llama3-8b", "inference")
        self.assertEqual(name, "throughput_tokens_per_sec")
        self.assertEqual(min_max, "max")

        # Energy/Carbon
        name, min_max = get_energy_eff_metric_name_and_min_max("llama3-8b", "inference")
        self.assertEqual(name, "avg_power_efficiency_tkn_per_joule")
        
        name, min_max = get_carbon_eff_metric_name_and_min_max("llama3-8b", "inference")
        self.assertEqual(name, "avg_total_carbon_efficiency_tkn_per_kgCO2e")
        self.assertEqual(min_max, "max")

    def test_training_workload(self):
        # Test metric names for training
        name, min_max = get_latency_metric_name_and_min_max("llama3-8b", "training")
        self.assertEqual(name, "total_execution_time_ns")
        
        name, min_max = get_energy_eff_metric_name_and_min_max("llama3-8b", "training")
        self.assertEqual(name, "avg_power_efficiency_iteration_per_joule")

        name, min_max = get_throughput_metric_name_and_min_max("llama3-8b", "training")
        self.assertEqual(name, "total_execution_time_ns")
        self.assertEqual(min_max, "min")

        # Test retrieving training stats
        stats = get_all_stats(
            model="llama3-8b", version="5p", workload="training",
            batch_size=1
        )
        self.assertIn((1, 1, 1, 1, 1, 1, 1), stats)
        # Real value is 663101427 ns from training-v5p.json
        self.assertEqual(stats[(1, 1, 1, 1, 1, 1, 1)]["total_execution_time_ns"], 663101427)

    def test_sd_model(self):
        # Test metric names for SD model (gligen)
        name, min_max = get_latency_metric_name_and_min_max("gligen", "inference")
        self.assertEqual(name, "latency_step_sec")

        # Test retrieving SD stats
        stats = get_all_stats(
            model="gligen", version="5p", workload="inference",
            batch_size=1
        )
        self.assertIn((1, 1, 1, 1, 1, 1, 1), stats)
        self.assertAlmostEqual(stats[(1, 1, 1, 1, 1, 1, 1)]["latency_step_sec"], 0.00010261685)

    def test_get_min_num_chips(self):
        # Minimum chips for standard inference asset (1 chip)
        min_chips = get_min_num_chips(
            model="llama3-8b", version="5p", workload="inference",
             prefill_or_decode="decode", batch_size=1
        )
        self.assertEqual(min_chips, 1)

    def test_get_component_data(self):
        # Use an existing csv asset
        csv_path = get_op_stats_filepath(
            model="llama3-8b", version="5p", workload="inference", prefill_or_decode="decode",
            dp=1, tp=1, pp=1, dp_dcn=1, tp_dcn=1, pp_dcn=1, batch_size=1,
            results_path=self.test_assets_path
        )
        
        # From inference-v5p_decode.csv (calculated sum)
        exec_time = get_component_data_from_file(csv_path, "Execution time")
        self.assertEqual(exec_time, 2649178112)
        
        # From inference-v5p_decode.csv (calculated sum)
        compute_time = get_component_data_from_file(csv_path, "Compute time")
        self.assertEqual(compute_time, 0)

    def test_get_total_execution_time(self):
         csv_path = get_op_stats_filepath(
            model="llama3-8b", version="5p", workload="inference", prefill_or_decode="decode",
            dp=1, tp=1, pp=1, dp_dcn=1, tp_dcn=1, pp_dcn=1, batch_size=1,
            results_path=self.test_assets_path
        )
         total_time = get_total_execution_time_from_file(csv_path)
         self.assertIsInstance(total_time, int)
         self.assertGreater(total_time, 0)

    def test_get_pareto_frontier(self):
        # Create dummy stats
        s1 = {"cost": 10, "lat": 100}
        s2 = {"cost": 20, "lat": 50}  # faster, more expensive
        s3 = {"cost": 30, "lat": 150} # dominated by s1 and s2 (worse cost than s2/s1, worse lat than s2/s1)
        
        cmp_cost = lambda a, b: a["cost"] < b["cost"]
        cmp_lat = lambda a, b: a["lat"] < b["lat"]
        
        frontier = get_pareto_frontier([s1, s2, s3], [cmp_cost, cmp_lat])
        self.assertIn(s1, frontier)
        self.assertIn(s2, frontier)
        self.assertNotIn(s3, frontier)

    def test_get_num_chips_with_args(self):
        # Standard calculation
        n = get_num_chips(
            "llama3-8b", dp=2, tp=2, pp=1, dp_dcn=1, tp_dcn=1, pp_dcn=1, ep=1, ep_dcn=1
        )
        self.assertEqual(n, 4)
        
        # DLRM case (dp must eq tp, pp=1)
        n_dlrm = get_num_chips(
            "dlrm-s", dp=4, tp=4, pp=1, dp_dcn=2, tp_dcn=1, pp_dcn=1
        )
        # For DLRM: num_chips = dp * dp_dcn = 4 * 2 = 8
        self.assertEqual(n_dlrm, 8)

    def test_get_num_chips_with_pstr(self):
        # Test with pstr only
        n = get_num_chips(
            "llama3-8b", pstr="dp2-tp2-pp1-dpdcn1-tpdcn1-ppdcn1-bs1"
        )
        self.assertEqual(n, 4)

    def test_get_slo_stat(self):
        # Should return stats for the config with min chips (1)
        slo_stat = get_slo_stat(
            model="llama3-8b", workload="inference", prefill_or_decode="decode", version="5p"
        )
        self.assertEqual(slo_stat["sim_config"]["num_chips"], 1)

    def test_optimal_stats_max_metric(self):
        opt_stats = get_optimal_stats_for_max_num_chips(
            model="gligen", version="5p", max_num_chips=1024,
            workload="inference", prefill_or_decode="",
            batch_size=1, perf_metric="throughput_step_per_sec_per_request",
            min_or_max_metric="max"
        )
        self.assertIsNotNone(opt_stats)
        # Check against real expected value from gligen/dp1-tp1.../inference-v5p.json
        self.assertAlmostEqual(opt_stats["throughput_step_per_sec_per_request"], 243.62470685857147)

    def test_optimal_stats_callable_and_error(self):
        # Test custom metric function
        # Mocking all_stats
        all_stats = {
            "conf1": {"custom_score": 10},
            "conf2": {"custom_score": 20}
        }
        res = get_optimal_stats_for_max_num_chips(
            "llama3-8b", "5p", perf_metric=lambda x: x["custom_score"],
            min_or_max_metric="max", all_stats=all_stats
        )
        self.assertEqual(res["custom_score"], 20)

        # Test invalid min_or_max
        with self.assertRaises(ValueError):
             get_optimal_stats_for_max_num_chips(
                "llama3-8b", "5p", perf_metric="custom_score",
                min_or_max_metric="average", all_stats=all_stats
            )

    def test_get_component_data_bounded_by(self):
        # Test with a csv with specific bounded-by values to hit branches
        temp_csv = os.path.join(self.test_assets_path, "temp_components.csv")
        with open(temp_csv, "w") as f:
            f.write("Execution time,Compute time,Memory time,ICI/NVLink time,Bounded-by,Count\n")
            f.write("100,50,60,20,Memory,1\n") # Bounded by Memory
            f.write("100,60,50,20,Compute,1\n") # Bounded by Compute
            f.write("100,20,20,80,ICI/NVLink,1\n") # Bounded by ICI
        try:
            # Test Memory bounded logic
            # val = (Memory - max(Compute, ICI)) * Count = (60 - 50) * 1 = 10
            val_mem = get_component_data_from_file(temp_csv, "Memory time")
            self.assertEqual(val_mem, 10)

            # Test ICI bounded logic
            # val = abs(ICI - Compute) * Count = abs(80 - 20) * 1 = 60
            val_ici = get_component_data_from_file(temp_csv, "ICI/NVLink time")
            self.assertEqual(val_ici, 60)

            # Test Invalid Key
            with self.assertRaises(ValueError):
                get_component_data_from_file(temp_csv, "Invalid time")
        finally:
            if os.path.exists(temp_csv):
                 os.remove(temp_csv)

    def test_errors(self):
        # Invalid pstr (enough parts to avoid index error, but wrong count)
        with self.assertRaises(ValueError):
             get_pconfig_from_pstr("dp1-tp1-pp1-bs1") 

        with self.assertRaises(ValueError):
             get_stats_filepath("llama3-8b", "5p", "invalid_workload", 1, 1, 1, 1, 1, 1, 1)

    def test_get_all_stats_filtering(self):
        # Using version '6p' which we populated with:
        # 1. dp1-tp1...bs1 (1 chip)
        # 2. dp1-tp1...bs32 (1 chip)
        # 3. dp1-tp2...bs1 (2 chips)
        
        # Keys: (dp, tp, pp, dp_dcn, tp_dcn, pp_dcn, batch_size)
        key_bs1_1chip = (1, 1, 1, 1, 1, 1, 1)
        key_bs32_1chip = (1, 1, 1, 1, 1, 1, 32)
        key_bs1_2chip = (1, 2, 1, 1, 1, 1, 1)

        # 1. No filter (just model/ver/workload/prefill_or_decode)
        stats_all = get_all_stats("llama3-8b", "6p", "inference", prefill_or_decode="decode")
        self.assertIn(key_bs1_1chip, stats_all)
        self.assertIn(key_bs32_1chip, stats_all)
        self.assertIn(key_bs1_2chip, stats_all)

        # 2. Filter by batch_size=1
        stats_bs1 = get_all_stats("llama3-8b", "6p", "inference", batch_size=1, prefill_or_decode="decode")
        self.assertIn(key_bs1_1chip, stats_bs1)
        self.assertIn(key_bs1_2chip, stats_bs1)
        self.assertNotIn(key_bs32_1chip, stats_bs1)
        
        # 3. Filter by batch_size=32
        stats_bs32 = get_all_stats("llama3-8b", "6p", "inference", batch_size=32, prefill_or_decode="decode")
        self.assertIn(key_bs32_1chip, stats_bs32)
        self.assertNotIn(key_bs1_1chip, stats_bs32)

        # 4. Filter by max_num_chips=1
        stats_max1 = get_all_stats("llama3-8b", "6p", "inference", max_num_chips=1, prefill_or_decode="decode")
        self.assertIn(key_bs1_1chip, stats_max1)
        self.assertIn(key_bs32_1chip, stats_max1)
        self.assertNotIn(key_bs1_2chip, stats_max1) # 2 chips > 1

        # 5. Filter by max_num_chips=1 AND batch_size=1
        stats_combo = get_all_stats("llama3-8b", "6p", "inference", batch_size=1, max_num_chips=1, prefill_or_decode="decode")
        self.assertIn(key_bs1_1chip, stats_combo)
        self.assertNotIn(key_bs1_2chip, stats_combo)
        self.assertNotIn(key_bs32_1chip, stats_combo)

    def test_read_json_with_csv(self):
        # Test get_all_op_stats with read_json_with_csv=True
        # Should return tuple match
        from neusim.npusim.frontend.query_results_helper_lib import get_all_op_stats
        
        stats = get_all_op_stats(
            "llama3-8b", "5p", "inference", "decode", batch_size=1,
            read_json_with_csv=True
        )
        key = (1, 1, 1, 1, 1, 1, 1)
        self.assertIn(key, stats)
        val = stats[key]
        self.assertIsInstance(val, tuple)
        self.assertEqual(len(val), 2)
        # (json_dict, csv_list)
        self.assertIsInstance(val[0], dict)
        self.assertIsInstance(val[1], list)
