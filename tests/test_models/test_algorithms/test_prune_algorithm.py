# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import unittest

import torch
from mmcls.structures import ClsDataSample
from mmengine import MessageHub
from mmengine.model import BaseModel

from mmrazor.models.algorithms.pruning.ite_prune_algorithm import (
    ItePruneAlgorithm, ItePruneConfigManager)
from mmrazor.registry import MODELS


# @TASK_UTILS.register_module()
class ImageClassifierPseudoLoss:
    """Calculate the pseudo loss to trace the topology of a `ImageClassifier`
    in MMClassification with `BackwardTracer`."""

    def __call__(self, model) -> torch.Tensor:
        pseudo_img = torch.rand(2, 3, 32, 32)
        pseudo_output = model(pseudo_img)
        return pseudo_output.sum()


MODEL_CFG = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

MUTATOR_CONFIG_NUM = dict(
    type='ChannelMutator',
    channel_unit_cfg={
        'type': 'SequentialMutableChannelUnit',
        'default_args': {
            'choice_mode': 'number'
        }
    })
MUTATOR_CONFIG_FLOAT = dict(
    type='ChannelMutator',
    channel_unit_cfg={
        'type': 'SequentialMutableChannelUnit',
        'default_args': {
            'choice_mode': 'ratio'
        }
    })

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')


class TestItePruneAlgorithm(unittest.TestCase):

    def _set_epoch_ite(self, epoch, ite, max_epoch):
        iter_per_epoch = 10
        message_hub = MessageHub.get_current_instance()
        message_hub.update_info('epoch', epoch)
        message_hub.update_info('max_epochs', max_epoch)
        message_hub.update_info('max_iters', max_epoch * 10)
        message_hub.update_info('iter', ite + iter_per_epoch * epoch)

    def fake_cifar_data(self):
        imgs = torch.randn(16, 3, 32, 32).to(DEVICE)
        data_samples = [
            ClsDataSample().set_gt_label(torch.randint(0, 10,
                                                       (16, ))).to(DEVICE)
        ]

        return {'inputs': imgs, 'data_samples': data_samples}

    def test_ite_prune_config_manager(self):
        float_origin, float_target = 1.0, 0.5
        int_origin, int_target = 10, 5
        for origin, target, manager in [
            (float_origin, float_target,
             ItePruneConfigManager({'a': float_target}, {'a': float_origin}, 2,
                                   5)),
            (int_origin, int_target,
             ItePruneConfigManager({'a': int_target}, {'a': int_origin}, 2, 5))
        ]:
            times = 1
            for e in range(1, 20):
                for ite in range(1, 5):
                    self._set_epoch_ite(e, ite, 5)
                    if (e, ite) in [(0, 0), (2, 0), (4, 0), (6, 0), (8, 0)]:
                        self.assertTrue(manager.is_prune_time(e, ite))
                        self.assertEqual(
                            manager.prune_at(e)['a'],
                            origin - (origin - target) * times / 5)
                        times += 1
                    else:
                        self.assertFalse(manager.is_prune_time(e, ite))

    def test_iterative_prune_int(self):

        data = self.fake_cifar_data()

        model = MODELS.build(MODEL_CFG)
        mutator = MODELS.build(MUTATOR_CONFIG_FLOAT)
        mutator.prepare_from_supernet(model)
        prune_target = mutator.sample_choices()

        epoch = 10
        epoch_step = 2
        times = 3

        algorithm = ItePruneAlgorithm(
            MODEL_CFG,
            target_pruning_ratio=prune_target,
            mutator_cfg=MUTATOR_CONFIG_FLOAT,
            step_epoch=epoch_step,
            prune_times=times).to(DEVICE)

        for e in range(epoch):
            for ite in range(5):
                self._set_epoch_ite(e, ite, 5)

                algorithm.forward(
                    data['inputs'], data['data_samples'], mode='loss')

        current_choices = algorithm.mutator.current_choices
        for key in current_choices:
            self.assertAlmostEqual(
                current_choices[key], prune_target[key], delta=0.1)

    def test_load_pretrained(self):
        epoch_step = 2
        times = 3
        data = self.fake_cifar_data()

        # prepare checkpoint
        model_cfg = copy.deepcopy(MODEL_CFG)
        model: BaseModel = MODELS.build(model_cfg)
        checkpoint_path = os.path.dirname(__file__) + '/checkpoint'
        torch.save(model.state_dict(), checkpoint_path)

        # build algorithm
        model_cfg['init_cfg'] = {
            'type': 'Pretrained',
            'checkpoint': checkpoint_path
        }
        algorithm = ItePruneAlgorithm(
            model_cfg,
            mutator_cfg=MUTATOR_CONFIG_NUM,
            target_pruning_ratio={},
            step_epoch=epoch_step,
            prune_times=times,
        ).to(DEVICE)
        algorithm.init_weights()
        algorithm.forward(data['inputs'], data['data_samples'], mode='loss')

        # delete checkpoint
        os.remove(checkpoint_path)

# from mmrazor.models.algorithms import IteAlgorithm
# from mmengine.model import BaseModel
from mmrazor.models.mutators import ChannelMutator, SlimmableChannelMutator
import torch.nn as nn

class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)

class Model_2(BaseModel):
    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 8, 3, 2, 1),
            nn.Conv2d(8, 16, 3, 2, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 1000)

    def forward(self, x):
        x_ = self.pool(self.feature(x))
        return self.head(x_.flatten(1))

class TestItePruneAlgorithm_2(unittest.TestCase):
    def test_1(self):
        model = Model()
        algorithm = ItePruneAlgorithm(model,
                                    mutator_cfg=dict(
                                        type='ChannelMutator',
                                        channel_unit_cfg=dict(type='L1MutableChannelUnit')),)
        print(algorithm)
        # IteAlgorithm(
        #   (data_preprocessor): BaseDataPreprocessor()
        #   (architecture): Model(
        #     (data_preprocessor): BaseDataPreprocessor()
        #     (conv): DynamicConv2d(
        #       3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        #       (mutable_attrs): ModuleDict(
        #         (in_channels): MutableChannelContainer(name=, num_channels=3, activated_channels: 3
        #         (out_channels): MutableChannelContainer(name=, num_channels=8, activated_channels: 8
        #       )
        #     )
        #   )
        #   (mutator): BaseChannelMutator()
        # )
    def test_2(self):
        model = Model_2()
        mutator = ChannelMutator()
        mutator.prepare_from_supernet(model) # 解析模型中的动态OP
        print(mutator.sample_choices())
        # {
        #     'feature.0_(0, 8)_out_1_in_1': 0.5,
        #     'feature.1_(0, 16)_out_1_in_1': 0.5625
        # }

        # DynamicConv2d / DynamicLinear
        # model = Model_2()
        # mutator = SlimmableChannelMutator()
        # mutator.prepare_from_supernet(model)
        # print(mutator.sample_choices())


        # 需要在build model的时候
        # channel_mutator :: prepare_from_supernet
        # sequential_mutable_channel_unit :: prepare_for_pruning
        # self._register_channel_container(model, MutableChannelContainer)
        # self._register_mutable_channel(self.mutable_channel)

        # _prepare_from_predefined_model 预定义？
