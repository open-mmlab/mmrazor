# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys
import unittest
from typing import Union

import torch

# from mmrazor.models.mutables import MutableChannelGroup
from mmrazor.models.mutables.mutable_channel import \
    SequentialMutableChannelGroup
from mmrazor.models.mutators.channel_mutator import BaseChannelMutator
from mmrazor.registry import MODELS
from ...test_core.test_graph.test_graph import TestGraph

sys.setrecursionlimit(2000)


@MODELS.register_module()
class RandomChannelGroup(SequentialMutableChannelGroup):

    def generate_mask(self, choice: Union[int, float]) -> torch.Tensor:
        if isinstance(choice, float):
            choice = max(1, int(self.num_channels * choice))
        assert 0 < choice <= self.num_channels
        rand_imp = torch.rand([self.num_channels])
        ind = rand_imp.topk(choice)[1]
        mask = torch.zeros([self.num_channels])
        mask.scatter_(-1, ind, 1)
        return mask


DATA_GROUPSS = [SequentialMutableChannelGroup, RandomChannelGroup]


class TestChannelMutator(unittest.TestCase):

    def _test_a_mutator(self, mutator: BaseChannelMutator, model):
        mutator.set_choices(mutator.sample_choices())
        self.assertGreater(len(mutator.mutable_groups), 0)
        x = torch.rand([2, 3, 224, 224])
        y = model(x)
        self.assertEqual(list(y.shape), [2, 1000])

    def test_sample_subnet(self):
        data_models = TestGraph.backward_tracer_passed_models()

        for i, data in enumerate(data_models):
            with self.subTest(i=i, data=data):
                model = data()

                mutator = BaseChannelMutator()
                mutator.prepare_from_supernet(model)

                self.assertGreaterEqual(len(mutator.mutable_groups), 1)

                self._test_a_mutator(mutator, model)

    def test_generic_support(self):
        data_models = TestGraph.backward_tracer_passed_models()

        for data_model in data_models[:1]:
            for group_type in DATA_GROUPSS:
                with self.subTest(model=data_model, unit=group_type):

                    model = data_model()

                    mutator = BaseChannelMutator(channel_group_cfg=group_type)
                    mutator.prepare_from_supernet(model)
                    mutator.groups

                    self._test_a_mutator(mutator, model)

    def test_init_groups_from_cfg(self):
        ARCHITECTURE_CFG = dict(
            type='mmcls.ImageClassifier',
            backbone=dict(type='mmcls.MobileNetV2', widen_factor=1.5),
            neck=dict(type='mmcls.GlobalAveragePooling'),
            head=dict(
                type='mmcls.LinearClsHead',
                num_classes=1000,
                in_channels=1920,
                loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
                topk=(1, 5)))
        model = MODELS.build(ARCHITECTURE_CFG)

        # generate config
        model1 = copy.deepcopy(model)
        mutator = BaseChannelMutator()
        mutator.prepare_from_supernet(model1)
        config = mutator.config_template(
            with_channels=True, with_group_init_args=True)

        # test passing config
        model2 = copy.deepcopy(model)
        config2 = copy.deepcopy(config)
        config2['tracer_cfg'] = None
        mutator2 = MODELS.build(config2)
        mutator2.prepare_from_supernet(model2)
        self.assertEqual(
            len(mutator.mutable_groups), len(mutator2.mutable_groups))
        self._test_a_mutator(mutator2, model2)

    def test_mix_config_tracer(self):
        model = TestGraph.backward_tracer_passed_models()[0]()

        model0 = copy.deepcopy(model)
        mutator0 = BaseChannelMutator()
        mutator0.prepare_from_supernet(model0)
        config = mutator0.config_template(with_group_init_args=True)

        model1 = copy.deepcopy(model)
        mutator1 = MODELS.build(config)
        mutator1.prepare_from_supernet(model1)
        config1 = mutator1.config_template(with_group_init_args=True)

        self.assertDictEqual(config1, config)
        self._test_a_mutator(mutator1, model1)
