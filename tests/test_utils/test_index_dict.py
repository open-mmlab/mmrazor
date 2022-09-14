# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmrazor.utils.index_dict import IndexDict


class TestIndexDict(unittest.TestCase):

    def test_dict(self):
        dict = IndexDict()
        dict[(4, 5)] = 2
        dict[(1, 3)] = 1

        self.assertSequenceEqual(list(dict.keys()), [(1, 3), (4, 5)])
        with self.assertRaisesRegex(AssertionError, 'overlap'):
            dict[2, 3] = 3
