# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmrazor.models.architectures.dynamic_ops import DynamicInputResizer
from mmrazor.models.architectures.ops import InputResizer
from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.registry import MODELS

_INPUT_MUTABLE = dict(
    input_resizer=dict(type='DynamicInputResizer'),
    mutable_shape=dict(
        type='OneShotMutableValue',
        value_list=[[192, 192], [224, 224], [256, 256], [288, 288]],
        default_value=[224, 224]))


class TestInputResizer(TestCase):

    def setUp(self):
        input_resizer_cfg_ = _INPUT_MUTABLE['input_resizer']
        self.dynamic_input_resizer = MODELS.build(input_resizer_cfg_)

        if not isinstance(self.dynamic_input_resizer, DynamicInputResizer):
            raise TypeError('input_resizer should be a `dict` or '
                            '`DynamicInputResizer` instance, but got '
                            f'{type(self.dynamic_input_resizer)}')

        self.mutable_shape = OneShotMutableValue(
            value_list=[[192, 192], [224, 224], [256, 256], [288, 288]],
            default_value=[224, 224])

        self.dynamic_input_resizer.register_mutable_attr(
            'shape', self.mutable_shape)

        self.assertTrue(
            self.dynamic_input_resizer.get_mutable_attr('shape').current_choice
            == [224, 224])

    def test_convert(self):
        static_m = InputResizer()

        dynamic_m = DynamicInputResizer.convert_from(static_m)

        self.assertIsNotNone(dynamic_m)

    def test_to_static_op(self):
        input = torch.randn(1, 3, 224, 224)

        mutable_shape = OneShotMutableValue(
            value_list=[192, 224, 256], default_value=224)
        mutable_shape.current_choice = 192

        with pytest.raises(RuntimeError):
            self.dynamic_input_resizer.to_static_op()

        mutable_shape.fix_chosen(mutable_shape.dump_chosen().chosen)
        self.dynamic_input_resizer.register_mutable_attr(
            'shape', mutable_shape)
        static_op = self.dynamic_input_resizer.to_static_op()
        x = static_op(input)
        static_m = InputResizer()
        output = static_m(input, mutable_shape.current_choice)
        self.assertTrue(torch.equal(x, output))

        mutable_shape = OneShotMutableValue(
            value_list=[[192, 192], [224, 224], [256, 256], [288, 288]],
            default_value=[224, 224])
        mutable_shape.current_choice = [192, 192]
        mutable_shape.fix_chosen(mutable_shape.dump_chosen().chosen)
        self.dynamic_input_resizer.register_mutable_attr(
            'shape', mutable_shape)

        static_op = self.dynamic_input_resizer.to_static_op()
        self.assertIsNotNone(static_op)
        x = self.dynamic_input_resizer(input)
        assert torch.equal(
            self.dynamic_input_resizer(input),
            static_op(input, mutable_shape.current_choice))
