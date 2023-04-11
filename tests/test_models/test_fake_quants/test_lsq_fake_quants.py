# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from torch.nn.parameter import Parameter

from mmrazor import digit_version
from mmrazor.models import LearnableFakeQuantize

try:
    from torch.ao.quantization import (MovingAverageMinMaxObserver,
                                       MovingAveragePerChannelMinMaxObserver)
except ImportError:
    from mmrazor.utils import get_placeholder
    MovingAverageMinMaxObserver = get_placeholder('torch>=1.13')
    MovingAveragePerChannelMinMaxObserver = get_placeholder('torch>=1.13')


class TestLearnableFakeQuantize(TestCase):

    def setUp(self):
        if digit_version(torch.__version__) < digit_version('1.13.0'):
            self.skipTest('version of torch < 1.13.0')
        self.zero_point_trainable_fakequant = LearnableFakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255,
            dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=True,
            zero_point_trainable=True)

        self.zero_point_untrainable_fakequant = \
            LearnableFakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine,
                reduce_range=True,
                zero_point_trainable=False)

        self.zero_point_untrainable_per_channel_fakequant = \
            LearnableFakeQuantize.with_args(
                observer=MovingAveragePerChannelMinMaxObserver,
                quant_min=0,
                quant_max=255,
                dtype=torch.quint8,
                qscheme=torch.per_channel_affine,
                reduce_range=True,
                zero_point_trainable=False)

    def test_repr(self):
        fq_module = self.zero_point_untrainable_fakequant()
        repr_str = f'static_enabled={torch.tensor([1], dtype=torch.uint8)}, '
        repr_str += f'fake_quant_enabled=' \
                    f'{torch.tensor([1], dtype=torch.uint8)}, '
        repr_str += 'quant_min=0, '
        repr_str += 'quant_max=127, '
        repr_str += f'dtype={torch.quint8}, '
        repr_str += f'qscheme={torch.per_tensor_affine}, '
        repr_str += f'scale={Parameter(torch.tensor([1.0]))}, '
        repr_str += f'zero_point={torch.tensor([0.])}, '
        repr_str += 'zero_point_trainable=False'
        self.assertEqual(fq_module.extra_repr(), repr_str)

        fq_module = self.zero_point_trainable_fakequant()
        repr_str = f'static_enabled={torch.tensor([1], dtype=torch.uint8)}, '
        repr_str += f'fake_quant_enabled=' \
                    f'{torch.tensor([1], dtype=torch.uint8)}, '
        repr_str += 'quant_min=0, '
        repr_str += 'quant_max=127, '
        repr_str += f'dtype={torch.quint8}, '
        repr_str += f'qscheme={torch.per_tensor_affine}, '
        repr_str += f'scale={Parameter(torch.tensor([1.0]))}, '
        repr_str += f'zero_point={Parameter(torch.tensor([0.]))}, '
        repr_str += 'zero_point_trainable=True'
        self.assertEqual(fq_module.extra_repr(), repr_str)

    def test_calculate_qparams(self):
        fq_module = self.zero_point_untrainable_fakequant()
        scale, zero_point = fq_module.calculate_qparams()
        self.assertEqual(scale, 1.)
        self.assertEqual(zero_point, 0.)

        fq_module = self.zero_point_trainable_fakequant()
        scale, zero_point = fq_module.calculate_qparams()
        self.assertEqual(scale, 1.)
        self.assertEqual(zero_point, 0.)

    def test_forward(self):
        fq_module = self.zero_point_untrainable_fakequant()
        torch.manual_seed(42)
        X = torch.rand(20, 10, dtype=torch.float32)
        # Output of fake quant is not identical to input
        Y = fq_module(X)
        self.assertFalse(torch.equal(Y, X))
        # self.assertNotEqual(Y, X)
        fq_module.toggle_fake_quant(False)
        X = torch.rand(20, 10, dtype=torch.float32)
        Y = fq_module(X)
        # Fake quant is disabled,output is identical to input
        self.assertTrue(torch.equal(Y, X))

        # Explicit copy at this point in time, because FakeQuant keeps internal
        # state in mutable buffers.
        scale = fq_module.scale.clone().detach()
        zero_point = fq_module.zero_point.clone().detach()

        fq_module.toggle_observer_update(False)
        fq_module.toggle_fake_quant(True)
        X = 10.0 * torch.rand(20, 10, dtype=torch.float32) - 5.0
        Y = fq_module(X)
        self.assertFalse(torch.equal(Y, X))
        # Observer is disabled, scale and zero-point do not change
        self.assertEqual(fq_module.scale, scale)
        self.assertEqual(fq_module.zero_point, zero_point)

        fq_module.toggle_observer_update(True)
        Y = fq_module(X)
        self.assertFalse(torch.equal(Y, X))
        # Observer is enabled, scale and zero-point are different
        self.assertNotEqual(fq_module.scale, scale)
        self.assertNotEqual(fq_module.zero_point, zero_point)

        fq_module = self.zero_point_trainable_fakequant()
        torch.manual_seed(42)
        X = torch.rand(20, 10, dtype=torch.float32)
        # Output of fake quant is not identical to input
        Y = fq_module(X)
        self.assertFalse(torch.equal(Y, X))
        # self.assertNotEqual(Y, X)
        fq_module.toggle_fake_quant(False)
        X = torch.rand(20, 10, dtype=torch.float32)
        Y = fq_module(X)
        # Fake quant is disabled,output is identical to input
        self.assertTrue(torch.equal(Y, X))

        # Explicit copy at this point in time, because FakeQuant keeps internal
        # state in mutable buffers.
        scale = fq_module.scale.clone().detach()
        zero_point = fq_module.zero_point.clone().detach()

        fq_module.toggle_observer_update(False)
        fq_module.toggle_fake_quant(True)
        X = 10.0 * torch.rand(20, 10, dtype=torch.float32) - 5.0
        Y = fq_module(X)
        self.assertFalse(torch.equal(Y, X))
        # Observer is disabled, scale and zero-point do not change
        self.assertEqual(fq_module.scale, scale)
        self.assertEqual(fq_module.zero_point, zero_point)

        fq_module.toggle_observer_update(True)
        Y = fq_module(X)
        self.assertFalse(torch.equal(Y, X))
        # Observer is enabled, scale and zero-point are different
        self.assertNotEqual(fq_module.scale, scale)
        self.assertNotEqual(fq_module.zero_point, zero_point)

    def test_state(self):
        fq_module = self.zero_point_untrainable_fakequant()

        fq_module.enable_param_learning()
        self.assertEqual(fq_module.learning_enabled[0], 1)
        self.assertEqual(fq_module.scale.requires_grad, 1)
        self.assertEqual(fq_module.zero_point.requires_grad, 0)
        self.assertEqual(fq_module.fake_quant_enabled[0], 1)
        self.assertEqual(fq_module.static_enabled[0], 0)

        fq_module.enable_static_estimate()
        self.assertEqual(fq_module.learning_enabled[0], 0)
        self.assertEqual(fq_module.scale.requires_grad, 0)
        self.assertEqual(fq_module.zero_point.requires_grad, 0)
        self.assertEqual(fq_module.fake_quant_enabled[0], 1)
        self.assertEqual(fq_module.static_enabled[0], 1)

        fq_module.enable_val()
        self.assertEqual(fq_module.learning_enabled[0], 0)
        self.assertEqual(fq_module.scale.requires_grad, 0)
        self.assertEqual(fq_module.zero_point.requires_grad, 0)
        self.assertEqual(fq_module.fake_quant_enabled[0], 1)
        self.assertEqual(fq_module.static_enabled[0], 0)

        fq_module.enable_static_observation()
        self.assertEqual(fq_module.learning_enabled[0], 0)
        self.assertEqual(fq_module.scale.requires_grad, 0)
        self.assertEqual(fq_module.zero_point.requires_grad, 0)
        self.assertEqual(fq_module.fake_quant_enabled[0], 0)
        self.assertEqual(fq_module.static_enabled[0], 1)

        fq_module = self.zero_point_trainable_fakequant()

        fq_module.enable_param_learning()
        self.assertEqual(fq_module.learning_enabled[0], 1)
        self.assertEqual(fq_module.scale.requires_grad, 1)
        self.assertEqual(fq_module.zero_point.requires_grad, 1)
        self.assertEqual(fq_module.fake_quant_enabled[0], 1)
        self.assertEqual(fq_module.static_enabled[0], 0)

    def test_load_state_dict(self):
        fq_module = self.zero_point_untrainable_per_channel_fakequant()
        state_dict = fq_module.state_dict()
        X = torch.rand(32, 16, 3, 3, dtype=torch.float32)
        # After forwarding, the shape of `scale` and `zero_point` in
        # `fq_module` will be in shape (32, ), while the shape of those in
        # `state_dict` are in shape (1, ).
        _ = fq_module(X)
        fq_module.load_state_dict(state_dict)
