# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import unittest

import torch
from mmcls.structures import ClsDataSample
from mmengine import MessageHub
from mmengine.model import BaseModel

from mmrazor.models.algorithms.pruning.ite_prune_algorithm import \
    ItePruneAlgorithm
from mmrazor.models.mutators import BaseChannelMutator
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

MUTATOR_CONFIG = dict(
    type='BaseChannelMutator',
    channl_group_cfg={'type': 'L1ChannelGroup'},
    tracer_cfg=dict(
        type='BackwardTracer', loss_calculator=ImageClassifierPseudoLoss()),
)

DEVICE = torch.device('cuda:0')


class TestItePruneAlgorithm(unittest.TestCase):

    def fake_cifar_data(self):
        imgs = torch.randn(16, 3, 32, 32).to(DEVICE)
        data_samples = [
            ClsDataSample().set_gt_label(torch.randint(0, 10,
                                                       (16, ))).to(DEVICE)
        ]

        return {'inputs': imgs, 'data_samples': data_samples}

    def test_iterative_prune(self):
        data = self.fake_cifar_data()
        message_hub = MessageHub.get_current_instance()

        model = MODELS.build(MODEL_CFG)
        mutator_cfg = copy.deepcopy(MUTATOR_CONFIG)
        mutator_cfg.pop('type')
        mutator = BaseChannelMutator(**mutator_cfg)
        mutator.prepare_from_supernet(model)
        prune_target = mutator.sample_choices()

        epoch = 10
        epoch_step = 2
        times = 3

        algorithm = ItePruneAlgorithm(
            MODEL_CFG,
            target_pruning_ratio=prune_target,
            mutator_cfg=MUTATOR_CONFIG,
            step_epoch=epoch_step,
            prune_times=times).to(DEVICE)

        for e in range(epoch):
            for ite in range(5):
                message_hub.update_info('epoch', e)
                message_hub.update_info('iter', ite)

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
            mutator_cfg=MUTATOR_CONFIG,
            target_pruning_ratio={},
            step_epoch=epoch_step,
            prune_times=times,
        ).to(DEVICE)
        algorithm.init_weights()
        algorithm.forward(data['inputs'], data['data_samples'], mode='loss')

        # delete checkpoint
        os.remove(checkpoint_path)
