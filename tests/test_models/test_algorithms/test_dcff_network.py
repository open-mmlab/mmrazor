# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import unittest

import torch
from mmcls.structures import ClsDataSample
from mmengine import MessageHub
from mmengine.model import BaseModel

from mmrazor.models.algorithms.pruning.dcff import DCFF
from mmrazor.models.algorithms.pruning.ite_prune_algorithm import \
    ItePruneConfigManager
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
    type='DCFFChannelMutator',
    channel_unit_cfg={
        'type': 'DCFFChannelUnit',
        'default_args': {
            'choice_mode': 'number'
        }
    })
MUTATOR_CONFIG_FLOAT = dict(
    type='DCFFChannelMutator',
    channel_unit_cfg={
        'type': 'DCFFChannelUnit',
        'default_args': {
            'choice_mode': 'ratio'
        }
    })

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')


class TestDCFFAlgorithm(unittest.TestCase):

    def _set_epoch_ite(self, epoch, ite, max_epoch):
        iter_per_epoch = 10
        message_hub = MessageHub.get_current_instance()
        message_hub.update_info('epoch', epoch)
        message_hub.update_info('max_epochs', max_epoch)
        message_hub.update_info('max_iters', max_epoch * iter_per_epoch)
        message_hub.update_info('iter', ite + iter_per_epoch * epoch)

    def fake_cifar_data(self):
        imgs = torch.randn(16, 3, 32, 32).to(DEVICE)
        data_samples = [
            ClsDataSample().set_gt_label(torch.randint(0, 10,
                                                       (16, ))).to(DEVICE)
        ]

        return {'inputs': imgs, 'data_samples': data_samples}

    def test_ite_prune_config_manager(self):
        iter_per_epoch = 10
        float_origin, float_target = 1.0, 0.5
        int_origin, int_target = 10, 5
        for origin, target, manager in [
            (float_origin, float_target,
             ItePruneConfigManager({'a': float_target}, {'a': float_origin},
                                   2 * iter_per_epoch, 5)),
            (int_origin, int_target,
             ItePruneConfigManager({'a': int_target}, {'a': int_origin},
                                   2 * iter_per_epoch, 5))
        ]:
            times = 1
            for e in range(1, 10):
                for ite in range(iter_per_epoch):
                    self._set_epoch_ite(e, ite, 10)
                    if (e, ite) in [(0, 0), (2, 0), (4, 0), (6, 0), (8, 0)]:
                        self.assertTrue(
                            manager.is_prune_time(e * iter_per_epoch + ite))
                        times += 1
                        self.assertEqual(
                            manager.prune_at(e * iter_per_epoch + ite)['a'],
                            origin - (origin - target) * times / 5)
                    else:
                        self.assertFalse(
                            manager.is_prune_time(e * iter_per_epoch + ite))

    def test_iterative_prune_int(self):

        data = self.fake_cifar_data()

        model = MODELS.build(MODEL_CFG)
        mutator = MODELS.build(MUTATOR_CONFIG_FLOAT)
        mutator.prepare_from_supernet(model)
        mutator.set_choices(mutator.sample_choices())
        prune_target = mutator.choice_template

        iter_per_epoch = 10
        epoch = 10
        epoch_step = 2
        times = 5

        algorithm = DCFF(
            MODEL_CFG,
            target_pruning_ratio=prune_target,
            mutator_cfg=MUTATOR_CONFIG_FLOAT,
            step_freq=epoch_step).to(DEVICE)

        for e in range(epoch):
            for ite in range(10):
                self._set_epoch_ite(e, ite, epoch)

                algorithm.forward(
                    data['inputs'], data['data_samples'], mode='loss')
                self.assertEqual(times, algorithm.prune_times)
                self.assertEqual(epoch_step * iter_per_epoch,
                                 algorithm.step_freq)

        current_choices = algorithm.mutator.current_choices
        group_prune_target = algorithm.group_target_pruning_ratio(
            prune_target, mutator.search_groups)
        for key in current_choices:
            self.assertAlmostEqual(
                current_choices[key], group_prune_target[key], delta=0.1)

    def test_load_pretrained(self):
        iter_per_epoch = 10
        epoch_step = 20
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
        algorithm = DCFF(
            model_cfg,
            mutator_cfg=MUTATOR_CONFIG_FLOAT,
            target_pruning_ratio=None,
            step_freq=epoch_step).to(DEVICE)
        algorithm.init_weights()
        self._set_epoch_ite(10, 5, 200)
        algorithm.forward(data['inputs'], data['data_samples'], mode='loss')
        self.assertEqual(algorithm.step_freq, epoch_step * iter_per_epoch)

        # delete checkpoint
        os.remove(checkpoint_path)

    def test_group_target_ratio(self):

        model = MODELS.build(MODEL_CFG)
        mutator = MODELS.build(MUTATOR_CONFIG_FLOAT)
        mutator.prepare_from_supernet(model)
        mutator.set_choices(mutator.sample_choices())
        prune_target = mutator.choice_template

        custom_groups = [[
            'backbone.layer1.0.conv1_(0, 64)_64',
            'backbone.layer1.1.conv1_(0, 64)_64'
        ]]
        mutator_cfg = copy.deepcopy(MUTATOR_CONFIG_FLOAT)
        mutator_cfg['custom_groups'] = custom_groups

        iter_per_epoch = 10
        epoch_step = 2
        epoch = 6
        data = self.fake_cifar_data()

        prune_target['backbone.layer1.0.conv1_(0, 64)_64'] = 0.1
        prune_target['backbone.layer1.1.conv1_(0, 64)_64'] = 0.1

        algorithm = DCFF(
            MODEL_CFG,
            target_pruning_ratio=prune_target,
            mutator_cfg=mutator_cfg,
            step_freq=epoch_step).to(DEVICE)

        algorithm.init_weights()
        self._set_epoch_ite(1, 2, epoch)
        algorithm.forward(data['inputs'], data['data_samples'], mode='loss')
        self.assertEqual(algorithm.step_freq, epoch_step * iter_per_epoch)

        prune_target['backbone.layer1.0.conv1_(0, 64)_64'] = 0.1
        prune_target['backbone.layer1.1.conv1_(0, 64)_64'] = 0.2

        with self.assertRaises(ValueError):

            algorithm = DCFF(
                MODEL_CFG,
                target_pruning_ratio=prune_target,
                mutator_cfg=mutator_cfg,
                step_freq=epoch_step).to(DEVICE)

            algorithm.init_weights()
            self._set_epoch_ite(1, 2, epoch)
            algorithm.forward(
                data['inputs'], data['data_samples'], mode='loss')
            self.assertEqual(algorithm.step_freq, epoch_step * iter_per_epoch)
