# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmrazor.models.architectures.dynamic_op import DynamicInputResizer
from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.structures.subnet import export_fix_subnet, load_fix_subnet


class TestDynamicOP(TestCase):

    def test_dynamic_input_resizer(self) -> None:
        d_ir = DynamicInputResizer(size=(320, 320))

        x = torch.rand(3, 3, 224, 224)
        out_before_mutate = d_ir(x)
        assert tuple(out_before_mutate.shape) == (3, 3, 320, 320)

        mutable_shape = OneShotMutableValue(
            value_list=[(192, 192), (224, 224), (288, 288), (320, 320)])
        d_ir.mutate_shape(mutable_shape)

        with pytest.raises(RuntimeError):
            d_ir.to_static_op()

        d_ir.mutable_shape.current_choice = (320, 320)

        out = d_ir(x)
        assert torch.equal(out_before_mutate, out)

        d_ir.mutable_shape.current_choice = (288, 288)
        out1 = d_ir(x)

        fix_mutables = export_fix_subnet(d_ir)
        with pytest.raises(RuntimeError):
            load_fix_subnet(d_ir, fix_mutables)

        s_ir = d_ir.to_static_op()
        assert s_ir._size == (288, 288)
        out2 = s_ir(x)

        assert torch.equal(out1, out2)
