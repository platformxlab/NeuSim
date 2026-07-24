import unittest

import neusim.fleetsim.util as util_test


class TestFleetSimUtil(unittest.TestCase):
    def test_pad_to(self):
        self.assertEqual(
            util_test.pad_to(10, 64),
            64,
        )
        self.assertEqual(
            util_test.pad_to(64, 64),
            64,
        )
        self.assertEqual(
            util_test.pad_to(65, 64),
            128,
        )
        self.assertEqual(
            util_test.pad_to(128, 64),
            128,
        )
        self.assertEqual(
            util_test.pad_to(129, 35),
            140,
        )

    def test_pad_seqlen(self):
        self.assertEqual(
            util_test.pad_seqlen(
                10,
                [4],
                [],
            ),
            12,
        )
        self.assertEqual(
            util_test.pad_seqlen(
                10,
                [4, 8],
                [16],
            ),
            12,
        )
        self.assertEqual(
            util_test.pad_seqlen(
                33,
                [4, 8, 16],
                [32, 64],
            ),
            40,
        )
        self.assertEqual(
            util_test.pad_seqlen(
                65,
                [4, 8, 16],
                [32, 64],
            ),
            80,
        )
