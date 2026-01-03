from math import ceil
import os
import unittest
import neusim.npusim.backend.util as util_under_test


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
