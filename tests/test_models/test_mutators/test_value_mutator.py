# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmrazor.models.mutables import MutableValue
from mmrazor.models.mutators import DynamicValueMutator
from ...data.models import DynamicAttention


class TestValueMutator(unittest.TestCase):

    def test_models_with_predefined_dynamic_op(self):
        for Model in [
                DynamicAttention,
        ]:
            with self.subTest(model=Model):
                model = Model()
                value_mutator = DynamicValueMutator()
                value_mutator.prepare_from_supernet(model)
                value_choices = value_mutator.sample_choices()
                value_mutator.set_choices(value_choices)

                mutable_value_space = []
                for mutable_value, module in model.named_modules():
                    if isinstance(module, MutableValue):
                        mutable_value_space.append(mutable_value)
                assert len(
                    value_mutator.search_groups) == len(mutable_value_space)

                x = torch.rand([2, 3, 224, 224])
                y = model(x)
                self.assertEqual(list(y.shape), [2, 624])
