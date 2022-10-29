# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

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

                print(value_choices)
                # self.assertGreater(len(mutator.mutable_units), 0)

                x = torch.rand([2, 3, 224, 224])
                y = model(x)
                print(list(y.shape))
                # self.assertEqual(list(y.shape), [2, 1000])
