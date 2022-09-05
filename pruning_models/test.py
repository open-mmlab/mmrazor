import unittest

import torch

from .vgg import VGGPruning

DATA = [{'model': VGGPruning}]


class TestModel(unittest.TestCase):

    def test_forward(self):
        for data in DATA:
            with self.subTest(data=data['model']):
                x = torch.rand([2, 3, 32, 32])
                model = data['model']()
                y = model(x)[0]
                self.assertEqual(list(y.shape), [2, 512])
