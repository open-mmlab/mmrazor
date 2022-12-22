# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest
from unittest import TestCase

import torch
import torch.nn as nn
from torch.ao.quantization import QConfig
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
from torchvision.models import resnet18

from mmrazor.models.observers import MinMaxFloorObserver


class ToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # TODO


class TestMinMaxObserver(TestCase):
    """TODO.

    Args:
        TestCase (_type_): _description_
    """

    def test_init(self):
        pass

    def test_prepare(self):
        pass

    def test_convert(self):
        pass

    def test_states(self):
        pass

    def test_forward(self):
        pass


class TestLSQObserver(TestMinMaxObserver):
    pass


class TestMinMaxFloorObserver(TestMinMaxObserver):

    def setUp(self) -> None:
        self.model_fp = resnet18()
        self.w_qscheme = dict(
            dtype=torch.qint8, qscheme=torch.per_tensor_affine)
        self.a_qscheme = dict(
            dtype=torch.quint8, qscheme=torch.per_tensor_affine)

    def test_init(self) -> None:
        with self.assertRaises(NotImplementedError):
            _ = MinMaxFloorObserver(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_symmetric,
                reduce_range=True)

    def test_prepare(self) -> None:
        flag = False
        model_to_quantize = copy.deepcopy(self.model_fp)
        model_to_quantize.eval()
        qconfig_dict = {
            '':
            QConfig(
                activation=MinMaxFloorObserver.with_args(**self.a_qscheme),
                weight=MinMaxFloorObserver.with_args(**self.w_qscheme))
        }
        prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
        for m in prepared_model.modules():
            if isinstance(m, MinMaxFloorObserver):
                flag = True
                break
        self.assertTrue(flag)

    def test_convert(self) -> None:
        flag = True
        model_to_quantize = copy.deepcopy(self.model_fp)
        model_to_quantize.eval()
        qconfig_dict = {
            '':
            QConfig(
                activation=MinMaxFloorObserver.with_args(**self.a_qscheme),
                weight=MinMaxFloorObserver.with_args(**self.w_qscheme))
        }
        prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
        prepared_model(torch.randn(1, 3, 224, 224))
        quantized_model = convert_fx(prepared_model)
        for m in quantized_model.modules():
            if isinstance(m, MinMaxFloorObserver):
                flag = False
                break
        self.assertTrue(flag)

    def test_states(self) -> None:
        test_input = torch.Tensor([6., -8.])
        observer = MinMaxFloorObserver(**self.w_qscheme)
        self.assertEqual(
            [observer.min_val, observer.max_val],
            [torch.tensor(float('inf')),
             torch.tensor(float('-inf'))])
        observer.forward(test_input)
        # per_tensor_affine
        scale, zero_point = observer.calculate_qparams()
        self.assertEqual(zero_point.item(), 18)

    def test_forward(self) -> None:
        test_input = torch.Tensor([1., -1.])
        observer = MinMaxFloorObserver(**self.w_qscheme)
        test_output = observer.forward(test_input)
        self.assertIs(test_input, test_output)


if __name__ == '__main__':
    unittest.main()
