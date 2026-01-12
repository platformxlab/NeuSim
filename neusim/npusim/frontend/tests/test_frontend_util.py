import unittest
from unittest.mock import MagicMock, patch
import zipfile

import neusim.npusim.frontend.util as util_under_test
from neusim.configs.models.ModelConfig import ModelConfig
from neusim.npusim.frontend.Operator import Operator

class TestNPUSimFrontendUtil(unittest.TestCase):
    def test_get_factors(self):
        self.assertEqual(util_under_test.get_factors(1), [1])
        self.assertEqual(util_under_test.get_factors(6), [1, 2, 3, 6])
        self.assertEqual(util_under_test.get_factors(7), [1, 7])
        self.assertEqual(util_under_test.get_factors(12), [1, 2, 3, 4, 6, 12])

    def test_prime_factorize(self):
        self.assertEqual(util_under_test.prime_factorize(1), [])
        self.assertEqual(util_under_test.prime_factorize(2), [2])
        self.assertEqual(util_under_test.prime_factorize(3), [3])
        self.assertEqual(util_under_test.prime_factorize(4), [2, 2])
        self.assertEqual(util_under_test.prime_factorize(6), [2, 3])
        self.assertEqual(util_under_test.prime_factorize(12), [2, 2, 3])
        self.assertEqual(util_under_test.prime_factorize(30), [2, 3, 5])

    def test_split_parallelism_degree(self):
        # p_degree=1, n_axes=1 -> [1]
        self.assertEqual(util_under_test.split_parallelism_degree(1, 1), [1])
        # p_degree=4, n_axes=2 -> [2, 2]
        self.assertEqual(util_under_test.split_parallelism_degree(4, 2), [2, 2])
        # p_degree=8, n_axes=3 -> [2, 2, 2]
        self.assertEqual(util_under_test.split_parallelism_degree(8, 3), [2, 2, 2])
        # p_degree=6, n_axes=2 -> [2, 3] or [3, 2]
        self.assertEqual(sorted(util_under_test.split_parallelism_degree(6, 2)), [2, 3])

    def test_get_ICI_topology_from_num_chips(self):
        config = MagicMock(spec=ModelConfig)
        
        # 2D case
        config.ICI_topology = "MESH_2D"
        config.num_chips = 4
        # expected [2, 2]
        self.assertEqual(sorted(util_under_test.get_ICI_topology_from_num_chips(config)), [2, 2])

        # 3D case
        config.ICI_topology = "TORUS_3D"
        config.num_chips = 8
        self.assertEqual(sorted(util_under_test.get_ICI_topology_from_num_chips(config)), [2, 2, 2])

    def test_get_bisection_bw_per_chip_GBps(self):
        config = MagicMock(spec=ModelConfig)
        config.ici_bw_GBps = 100.0
        
        with patch('neusim.npusim.frontend.util.get_ICI_topology_from_num_chips') as mock_topo:
            # 1D case (forced by return value)
            mock_topo.return_value = [4]
            config.num_chips = 4
            bw, topo = util_under_test.get_bisection_bw_per_chip_GBps(config)
            self.assertEqual(bw, 25.0)
            self.assertEqual(topo, [4])
            
            # 2D case
            mock_topo.return_value = [4, 4]
            config.num_chips = 16
            bw, topo = util_under_test.get_bisection_bw_per_chip_GBps(config)
            self.assertEqual(bw, 25.0)
            
            # 3D case
            mock_topo.return_value = [4, 4, 4]
            config.num_chips = 64
            bw, topo = util_under_test.get_bisection_bw_per_chip_GBps(config)
            self.assertEqual(bw, 50.0)

    def test_compute_component_slack_for_op(self):
        op = MagicMock()
        op.stats.execution_time_ns = 100
        op.stats.sa_time_ns = 80
        op.stats.vu_time_ns = 50
        op.stats.memory_time_ns = 100
        op.stats.ici_time_ns = 0
        op.stats.vmem_time_ns = 120 
        
        extras, ratios = util_under_test.compute_component_slack_for_op(op)
        
        self.assertEqual(extras["sa"], 20)
        self.assertEqual(ratios["sa"], 0.25) # 20/80
        self.assertEqual(extras["vu"], 50)
        self.assertEqual(ratios["vu"], 1.0) # 50/50
        self.assertEqual(extras["hbm"], 0)
        self.assertEqual(ratios["hbm"], 0.0)
        self.assertEqual(extras["ici"], 0)
        self.assertEqual(ratios["ici"], 0.0)
        self.assertEqual(extras["vmem"], 0)
        self.assertEqual(ratios["vmem"], 0.0)

    def test_open_zip(self):
        with patch('zipfile.ZipFile') as mock_zip:
            # Test existing zip
            util_under_test.open_zip("test.zip")
            mock_zip.assert_called_with(file="test.zip", mode="r", compression=zipfile.ZIP_DEFLATED)
            
            # Test non-existing file (adds .zip)
            with patch('os.path.exists', return_value=False):
                util_under_test.open_zip("new_file", mode="w")
                mock_zip.assert_called_with(file="new_file.zip", mode="w", compression=zipfile.ZIP_DEFLATED)
