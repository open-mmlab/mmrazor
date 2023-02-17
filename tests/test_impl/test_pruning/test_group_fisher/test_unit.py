# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmrazor.implementations.pruning.group_fisher import \
    GroupFisherChannelMutator
from ....data.models import MMClsResNet18


class TestGroupFisherChannelUnit(unittest.TestCase):

    def test_init(self):
        model = MMClsResNet18()
        mutator = GroupFisherChannelMutator(
            parse_cfg=dict(
                type='ChannelAnalyzer',
                demo_input=(1, 3, 224, 224),
                tracer_type='BackwardTracer'))
        mutator.prepare_from_supernet(model)

        x = torch.rand([1, 3, 224, 224])
        mutator.start_record_info()
        for i in range(2):
            model.train()
            loss = model(x).sum()
            loss.backward()
        mutator.end_record_info()

        for unit in mutator.mutable_units:
            for module in unit.input_related_dynamic_ops:
                self.assertEqual(len(module.recorded_input), 2)
                self.assertEqual(len(module.recorded_grad), 2)
                self.assertIsInstance(module.recorded_grad[0], torch.Tensor)

        unit = mutator.mutable_units[0]
        fisher = unit._fisher_of_a_module(next(unit.input_related_dynamic_ops))
        self.assertEqual(list(fisher.shape), [1, unit.num_channels])

        fisher = unit.current_batch_fisher
        self.assertEqual(list(fisher.shape), [unit.num_channels])

        fisher = unit._get_normalized_fisher_info(fisher, unit.delta_type)
        unit.update_fisher_info()
