# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest
from typing import Union

import torch

# from mmrazor.models.mutables import MutableChannelUnit
from mmrazor.models.mutables.mutable_channel import (
    L1MutableChannelUnit, SequentialMutableChannelUnit)
from mmrazor.models.mutators.channel_mutator import ChannelMutator
from mmrazor.models.task_modules import ChannelAnalyzer
from mmrazor.registry import MODELS
from ...data.models import DynamicAttention, DynamicLinearModel, DynamicMMBlock
from ...data.tracer_passed_models import backward_passed_library


@MODELS.register_module()
class RandomChannelUnit(SequentialMutableChannelUnit):

    def generate_mask(self, choice: Union[int, float]) -> torch.Tensor:
        if isinstance(choice, float):
            choice = max(1, int(self.num_channels * choice))
        assert 0 < choice <= self.num_channels
        rand_imp = torch.rand([self.num_channels])
        ind = rand_imp.topk(choice)[1]
        mask = torch.zeros([self.num_channels])
        mask.scatter_(-1, ind, 1)
        return mask


DATA_UNITS = [
    SequentialMutableChannelUnit, RandomChannelUnit, L1MutableChannelUnit
]


class TestChannelMutator(unittest.TestCase):

    def _test_a_mutator(self, mutator: ChannelMutator, model):
        self.assertGreater(len(mutator.mutable_units), 0)
        x = torch.rand([2, 3, 224, 224])
        y = model(x)
        self.assertEqual(list(y.shape), [2, 1000])

    def test_init(self):
        model = backward_passed_library.include_models()[0]()
        mutator = ChannelMutator(parse_cfg=ChannelAnalyzer())
        mutator.prepare_from_supernet(model)
        self.assertGreaterEqual(len(mutator.mutable_units), 1)
        self._test_a_mutator(mutator, model)

    def test_sample_subnet(self):
        data_models = backward_passed_library.include_models()[:2]

        for i, data in enumerate(data_models):
            with self.subTest(i=i, data=data):
                model = data()

                mutator = ChannelMutator()
                mutator.prepare_from_supernet(model)

                self.assertGreaterEqual(len(mutator.mutable_units), 1)

                self._test_a_mutator(mutator, model)

    def test_generic_support(self):
        data_models = backward_passed_library.include_models()

        for data_model in data_models[:1]:
            for unit_type in DATA_UNITS:
                with self.subTest(model=data_model, unit=unit_type):

                    model = data_model()

                    mutator = ChannelMutator(channel_unit_cfg=unit_type)
                    mutator.prepare_from_supernet(model)
                    mutator.units

                    self._test_a_mutator(mutator, model)

    def test_init_units_from_cfg(self):
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
        mutator = ChannelMutator()
        mutator.prepare_from_supernet(model1)
        config = mutator.config_template(
            with_channels=True, with_unit_init_args=True)

        # test passing config
        model2 = copy.deepcopy(model)
        config2 = copy.deepcopy(config)
        config2['parse_cfg'] = {'type': 'Config'}
        mutator2 = MODELS.build(config2)
        mutator2.prepare_from_supernet(model2)
        self.assertEqual(
            len(mutator.mutable_units), len(mutator2.mutable_units))
        self._test_a_mutator(mutator2, model2)

    def test_mix_config_tracer(self):
        model = backward_passed_library.include_models()[0]()

        model0 = copy.deepcopy(model)
        mutator0 = ChannelMutator()
        mutator0.prepare_from_supernet(model0)
        config = mutator0.config_template(with_unit_init_args=True)

        model1 = copy.deepcopy(model)
        mutator1 = MODELS.build(config)
        mutator1.prepare_from_supernet(model1)
        config1 = mutator1.config_template(with_unit_init_args=True)

        self.assertDictEqual(config1, config)
        self._test_a_mutator(mutator1, model1)

    def test_models_with_predefined_dynamic_op(self):
        for Model in [
                DynamicLinearModel,
        ]:
            with self.subTest(model=Model):
                model = Model()
                mutator = ChannelMutator(
                    channel_unit_cfg={
                        'type': 'OneShotMutableChannelUnit',
                        'default_args': {}
                    },
                    parse_cfg={'type': 'Predefined'})
                mutator.prepare_from_supernet(model)
                self._test_a_mutator(mutator, model)

    def test_models_with_predefined_dynamic_op_without_pruning(self):
        for Model in [
                DynamicAttention,
        ]:
            with self.subTest(model=Model):
                model = Model()
                mutator = ChannelMutator(
                    channel_unit_cfg={
                        'type': 'OneShotMutableChannelUnit',
                        'default_args': {
                            'unit_predefined': True
                        }
                    },
                    parse_cfg={'type': 'Predefined'})
                mutator.prepare_from_supernet(model)
                self.assertGreater(len(mutator.mutable_units), 0)
                x = torch.rand([2, 3, 224, 224])
                y = model(x)
                self.assertEqual(list(y.shape), [2, 624])

    def test_related_shortcut_layer(self):
        for Model in [
                DynamicMMBlock,
        ]:
            with self.subTest(model=Model):
                model = Model()
                mutator = ChannelMutator(
                    channel_unit_cfg={
                        'type': 'OneShotMutableChannelUnit',
                        'default_args': {
                            'unit_predefined': True
                        }
                    },
                    parse_cfg={'type': 'Predefined'})
                mutator.prepare_from_supernet(model)
                self.assertGreater(len(mutator.mutable_units), 0)
                x = torch.rand([2, 3, 224, 224])
                y = model(x)
                self.assertEqual(list(y[-1].shape), [2, 1984, 1, 1])
