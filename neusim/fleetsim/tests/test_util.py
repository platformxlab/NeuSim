import unittest

import neusim.fleetsim.util as util_test


class TestFleetSimUtil(unittest.TestCase):
    def test_num_chips_to_shape_2D(self):
        self.assertEqual(util_test.num_chips_to_shape_2D(1), [1, 1])
        self.assertEqual(util_test.num_chips_to_shape_2D(4), [2, 2])
        self.assertEqual(util_test.num_chips_to_shape_2D(6), [2, 3])
        self.assertEqual(util_test.num_chips_to_shape_2D(12), [3, 4])
        self.assertEqual(util_test.num_chips_to_shape_2D(16), [4, 4])
        self.assertEqual(util_test.num_chips_to_shape_2D(20), [4, 5])
        self.assertEqual(util_test.num_chips_to_shape_2D(36), [6, 6])
        self.assertEqual(util_test.num_chips_to_shape_2D(100), [10, 10])
        self.assertEqual(util_test.num_chips_to_shape_2D(4096), [64, 64])

    def test_num_chips_to_shape_3D(self):
        self.assertEqual(util_test.num_chips_to_shape_3D(1), [1, 1, 1])
        self.assertEqual(util_test.num_chips_to_shape_3D(4), [1, 2, 2])
        self.assertEqual(util_test.num_chips_to_shape_3D(8), [2, 2, 2])
        self.assertEqual(util_test.num_chips_to_shape_3D(12), [2, 2, 3])
        self.assertEqual(util_test.num_chips_to_shape_3D(16), [2, 2, 4])
        self.assertEqual(util_test.num_chips_to_shape_3D(27), [3, 3, 3])
        self.assertEqual(util_test.num_chips_to_shape_3D(64), [4, 4, 4])
        self.assertEqual(util_test.num_chips_to_shape_3D(100), [4, 5, 5])
        self.assertEqual(util_test.num_chips_to_shape_3D(4096), [16, 16, 16])

    def test_shape_to_num_chips(self):
        self.assertEqual(util_test.shape_to_num_chips([1, 1]), 1)
        self.assertEqual(util_test.shape_to_num_chips([2, 2]), 4)
        self.assertEqual(util_test.shape_to_num_chips([2, 3]), 6)
        self.assertEqual(util_test.shape_to_num_chips([3, 4]), 12)
        self.assertEqual(util_test.shape_to_num_chips([4, 4]), 16)
        self.assertEqual(util_test.shape_to_num_chips([4, 5]), 20)
        self.assertEqual(util_test.shape_to_num_chips([6, 6]), 36)
        self.assertEqual(util_test.shape_to_num_chips([10, 10]), 100)
        self.assertEqual(util_test.shape_to_num_chips([64, 64]), 4096)

        self.assertEqual(util_test.shape_to_num_chips([1, 1, 1]), 1)
        self.assertEqual(util_test.shape_to_num_chips([1, 2, 2]), 4)
        self.assertEqual(util_test.shape_to_num_chips([2, 2, 2]), 8)
        self.assertEqual(util_test.shape_to_num_chips([2, 2, 3]), 12)
        self.assertEqual(util_test.shape_to_num_chips([2, 2, 4]), 16)
        self.assertEqual(util_test.shape_to_num_chips([3, 3, 3]), 27)
        self.assertEqual(util_test.shape_to_num_chips([4, 4, 4]), 64)
        self.assertEqual(util_test.shape_to_num_chips([4, 5, 5]), 100)
        self.assertEqual(util_test.shape_to_num_chips([16, 16, 16]), 4096)

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

    def test_sample_from_list(self):
        lst = list(range(100))

        sampled = util_test.sample_from_list(lst, 10)
        self.assertEqual(len(sampled), 10)
        self.assertEqual(sampled, [0, 11, 22, 33, 44, 55, 66, 77, 88, 99])

        sampled = util_test.sample_from_list(lst, 5)
        self.assertEqual(len(sampled), 5)
        self.assertEqual(sampled, [0, 24, 49, 74, 99])

        sampled = util_test.sample_from_list(lst, 13)
        self.assertEqual(len(sampled), 13)
        self.assertEqual(sampled, [0, 8, 16, 24, 33, 41, 49, 57, 66, 74, 82, 90, 99])


class TestListMap(unittest.TestCase):
    def test_create_listmap(self):
        ListMap = util_test.ListMap
        m = ListMap()

        some_unhashable_objs = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        values = [10, 20, 30]
        m = ListMap(list(zip(some_unhashable_objs, values, strict=True)))
        self.assertEqual(len(m), 3)

    def test_setitem_and_getitem(self):
        ListMap = util_test.ListMap
        m = ListMap()

        some_unhashable_objs = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        values = [10, 20, 30]
        m = ListMap(list(zip(some_unhashable_objs, values, strict=False)))

        self.assertEqual(m[[1, 2]], 10)
        self.assertEqual(m[[3, 4]], 20)
        self.assertEqual(m[[5, 6]], 30)

        m[[7, 8]] = 40
        self.assertEqual(m[[7, 8]], 40)

        m[[1, 2]] = 15
        self.assertEqual(m[[1, 2]], 15)

    def test_delitem(self):
        ListMap = util_test.ListMap
        m = ListMap()

        some_unhashable_objs = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        values = [10, 20, 30]
        m = ListMap(list(zip(some_unhashable_objs, values, strict=False)))

        del m[[3, 4]]
        self.assertFalse([3, 4] in m)

    def test_contains(self):
        ListMap = util_test.ListMap
        m = ListMap()

        some_unhashable_objs = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        values = [10, 20, 30]
        m = ListMap(list(zip(some_unhashable_objs, values, strict=False)))

        self.assertTrue([1, 2] in m)
        self.assertTrue([3, 4] in m)
        self.assertTrue([5, 6] in m)
        self.assertFalse([7, 8] in m)

    def test_len(self):
        ListMap = util_test.ListMap
        m = ListMap()

        some_unhashable_objs = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        values = [10, 20, 30]
        m = ListMap(list(zip(some_unhashable_objs, values, strict=False)))

        self.assertEqual(len(m), 3)
        m[[7, 8]] = 40
        self.assertEqual(len(m), 4)
        del m[[3, 4]]
        self.assertEqual(len(m), 3)

    def test_items_keys_values(self):
        ListMap = util_test.ListMap
        m = ListMap()

        some_unhashable_objs = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        values = [10, 20, 30]
        m = ListMap(list(zip(some_unhashable_objs, values, strict=False)))

        self.assertEqual(m.items(), list(zip(some_unhashable_objs, values, strict=False)))
        self.assertEqual(m.keys(), some_unhashable_objs)
        self.assertEqual(m.values(), values)
