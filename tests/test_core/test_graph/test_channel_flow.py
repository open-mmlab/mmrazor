# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmrazor.structures.graph.channel_flow import ChannelElem, ChannelTensor


class TestChannelTensor(unittest.TestCase):

    def test_union(self):
        tensor1 = ChannelTensor(8)
        tensor2 = ChannelTensor(8)
        tensor3 = ChannelTensor(8)
        tensor4 = ChannelTensor(8)

        ChannelTensor.union_two(tensor1, tensor2)
        ChannelTensor.union_two(tensor3, tensor4)
        self.assertUionedTensor(tensor1, tensor2)
        self.assertUionedTensor(tensor3, tensor4)

        ChannelTensor.union_two(tensor1, tensor4)

        self.assertUionedTensor(tensor1, tensor2)
        self.assertUionedTensor(tensor2, tensor3)
        self.assertUionedTensor(tensor3, tensor4)
        self.assertUionedTensor(tensor1, tensor4)

    def test_cat(self):
        tensor1 = ChannelTensor(8)
        tensor2 = ChannelTensor(8)
        tensor3 = ChannelTensor(16)

        tensor_cat = ChannelTensor.cat([tensor1, tensor2])
        self.assertEqual(len(tensor_cat), 16)
        ChannelTensor.union_two(tensor_cat, tensor3)

        tensor31 = tensor3[:8]
        tensor32 = tensor3[8:]
        self.assertUionedTensor(tensor1, tensor31)
        self.assertUionedTensor(tensor2, tensor32)

    def test_add_cat(self):
        """8+8 && 4+12 -> 4+4+8."""
        tensor1 = ChannelTensor(8)
        tensor2 = ChannelTensor(8)
        tensor_cat1 = ChannelTensor.cat([tensor1, tensor2])

        tensor3 = ChannelTensor(4)
        tensor4 = ChannelTensor(12)
        tensor_cat2 = ChannelTensor.cat([tensor3, tensor4])

        ChannelTensor.union_two(tensor_cat1, tensor_cat2)
        self.assertUionedTensor(tensor_cat1, tensor_cat2)

        self.assertUionedTensor(tensor_cat1[0:4], tensor3[0:4])
        self.assertUionedTensor(tensor_cat1[4:8], tensor4[0:4])
        self.assertUionedTensor(tensor_cat1[8:16], tensor4[4:12])

        self.assertUionedTensor(tensor_cat2[0:4], tensor1[0:4])
        self.assertUionedTensor(tensor_cat2[4:8], tensor1[4:8])
        self.assertUionedTensor(tensor_cat2[8:], tensor2)

    def assertUionedTensor(self, tensor1: ChannelTensor,
                           tensor2: ChannelTensor):
        assert len(tensor1) == len(tensor2)
        for e1, e2 in zip(tensor1, tensor2):
            self.assertEqual(e1.root, e2.root)


class TestChannelElem(unittest.TestCase):

    def test_union(self):
        tensor = ChannelTensor(10)
        elem1 = tensor[1]
        elem2 = tensor[2]
        ChannelElem.union_two(elem1, elem2)
        self.assertEqual(elem1.root, elem2.root)

        elem3 = tensor[3]
        ChannelElem.union_two(elem2, elem3)
        self.assertEqual(elem1.root, elem3.root)
