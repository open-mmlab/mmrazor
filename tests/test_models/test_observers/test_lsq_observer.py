# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch

from mmrazor import digit_version
from mmrazor.models import LSQObserver, LSQPerChannelObserver


class TestLSQObserver(TestCase):

    def setUp(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        self.lsq = LSQObserver.with_args(
            dtype=torch.quint8,
            qscheme=torch.per_tensor_symmetric,
            reduce_range=False,
            quant_min=0,
            quant_max=255)

    def test_forward(self):
        lsq_observer = self.lsq()
        torch.manual_seed(42)
        X = torch.rand(20, 10, dtype=torch.float32)
        Y = lsq_observer(X)
        # Output of observer is identical to input
        self.assertTrue(torch.equal(Y, X))

        X = torch.rand(0, dtype=torch.float32)
        Y = lsq_observer(X)
        # Output of observer is identical to input
        self.assertTrue(torch.equal(Y, X))

    def test_calculate_qparams(self):
        lsq_observer = self.lsq()
        X = torch.ones(10, dtype=torch.float32)
        _ = lsq_observer(X)
        scale, zero_point = lsq_observer.calculate_qparams()
        # tensor_norm = 1, quant_max = 255
        self.assertEqual(scale, 2 * torch.tensor([1.]) / (255**0.5))
        self.assertEqual(zero_point, 127)


class TestLSQPerChannelObserver(TestCase):

    def setUp(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        self.lsq = LSQPerChannelObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            quant_min=-127,
            quant_max=127)

    def test_forward(self):
        lsq_observer = self.lsq()
        torch.manual_seed(42)
        X = torch.rand(2, 10, dtype=torch.float32)
        Y = lsq_observer(X)
        # Output of observer is identical to input
        self.assertTrue(torch.equal(Y, X))

        X = torch.rand(0, dtype=torch.float32)
        Y = lsq_observer(X)
        # Output of observer is identical to input
        self.assertTrue(torch.equal(Y, X))

    def test_calculate_qparams(self):
        lsq_observer = self.lsq()
        X = torch.ones(2, 10, dtype=torch.float32)
        X[0] -= 1
        _ = lsq_observer(X)
        scale, zero_point = lsq_observer.calculate_qparams()
        self.assertEqual(scale[0], 2 * torch.tensor([0.]) / (127**0.5))
        self.assertEqual(scale[1], 2 * torch.tensor([1.]) / (127**0.5))
