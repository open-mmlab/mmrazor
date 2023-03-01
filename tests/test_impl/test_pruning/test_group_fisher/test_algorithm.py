# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmcls.structures import ClsDataSample
from mmengine import MessageHub

from mmrazor.implementations.pruning.group_fisher.algorithm import \
    GroupFisherAlgorithm
from mmrazor.implementations.pruning.group_fisher.ops import GroupFisherConv2d
from ....data.models import MMClsResNet18

if torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')


class TestGroupFisherPruneAlgorithm(TestCase):

    def fake_cifar_data(self):
        imgs = torch.randn(16, 3, 32, 32).to(DEVICE)
        data_samples = [
            ClsDataSample().set_gt_label(torch.randint(0, 10,
                                                       (16, ))).to(DEVICE)
        ]

        return {'inputs': imgs, 'data_samples': data_samples}

    def test_group_fisher_prune(self):
        data = self.fake_cifar_data()

        MUTATOR_CONFIG = dict(
            type='GroupFisherChannelMutator',
            parse_cfg=dict(
                type='ChannelAnalyzer', tracer_type='BackwardTracer'),
            channel_unit_cfg=dict(type='GroupFisherChannelUnit'))

        epoch = 2
        interval = 1

        algorithm = GroupFisherAlgorithm(
            MMClsResNet18(), mutator=MUTATOR_CONFIG,
            interval=interval).to(DEVICE)
        mutator = algorithm.mutator

        for e in range(epoch):
            for ite in range(10):
                self._set_epoch_ite(e, ite, epoch)
                algorithm.forward(
                    data['inputs'], data['data_samples'], mode='loss')
                self.gen_fake_grad(mutator)
        self.assertEqual(interval, algorithm.interval)

    def gen_fake_grad(self, mutator):
        for unit in mutator.mutable_units:
            for channel in unit.input_related:
                module = channel.module
                if isinstance(module, GroupFisherConv2d):
                    module.recorded_grad = module.recorded_input

    def _set_epoch_ite(self, epoch, ite, max_epoch):
        iter_per_epoch = 10
        message_hub = MessageHub.get_current_instance()
        message_hub.update_info('epoch', epoch)
        message_hub.update_info('max_epochs', max_epoch)
        message_hub.update_info('max_iters', max_epoch * 10)
        message_hub.update_info('iter', ite + iter_per_epoch * epoch)
