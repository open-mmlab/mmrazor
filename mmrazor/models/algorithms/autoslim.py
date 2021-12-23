# Copyright (c) OpenMMLab. All rights reserved.
import copy

import mmcv
import torch
import torch.nn as nn
from mmcv.cnn import get_model_complexity_info
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.builder import ALGORITHMS, build_pruner
from mmrazor.models.utils import add_prefix
from .base import BaseAlgorithm


@ALGORITHMS.register_module()
class AutoSlim(BaseAlgorithm):
    """AutoSlim: A one-shot architecture search for channel numbers.

    Please refer to the `paper <https://arxiv.org/abs/1903.11728>` for
    details.

    Args:
        num_sample_training (int): In each iteration we train the model at
            smallest width, largest width and (`num_sample_training` − 2)
            random widths. It should be no less than 2. Defaults to 4
        input_shape (tuple): Input shape used for calculation the flops of the
            supernet.
        bn_training_mode (bool): Whether set bn to training mode when model is
            set to eval mode. Note that in slimmable networks, accumulating
            different numbers of channels results in different feature means
            and variances, which further leads to inaccurate statistics of
            shared BN. Set ``bn_training_mode`` to True to use the feature
            means and variances in a batch.
    """

    def __init__(self,
                 num_sample_training=4,
                 input_shape=(3, 224, 224),
                 bn_training_mode=False,
                 **kwargs):

        super(AutoSlim, self).__init__(**kwargs)
        assert num_sample_training >= 2, \
            'num_sample_training should be no less than 2'
        self.num_sample_training = num_sample_training

        # set bn to training mode when model is set to eval mode
        self.bn_training_mode = bn_training_mode

        if input_shape is not None:
            self.input_shape = input_shape
            self._init_flops()

    def _init_pruner(self, pruner):
        """Build registered pruners and make preparations.

        Args:
            pruner (dict): The registered pruner to be used
                in the algorithm.
        """
        if pruner is None:
            self.pruner = None
            return

        # judge whether our StructurePruner can prune the architecture
        try:
            pseudo_pruner = build_pruner(pruner)
            pseudo_architecture = copy.deepcopy(self.architecture)
            pseudo_pruner.prepare_from_supernet(pseudo_architecture)
            subnet_dict = pseudo_pruner.sample_subnet()
            pseudo_pruner.set_subnet(subnet_dict)
            subnet_dict = pseudo_pruner.export_subnet()

            pseudo_pruner.deploy_subnet(pseudo_architecture, subnet_dict)
            pseudo_img = torch.randn(1, 3, 224, 224)
            pseudo_architecture.forward_dummy(pseudo_img)
        except RuntimeError:
            raise NotImplementedError('Our current StructurePruner does not '
                                      'support pruning this architecture. '
                                      'StructurePruner is not perfect enough '
                                      'to handle all the corner cases. We will'
                                      ' appreciate it if you create a issue.')

        self.pruner = build_pruner(pruner)

        if self.retraining:
            if isinstance(self.channel_cfg, dict):
                self.pruner.deploy_subnet(self.architecture, self.channel_cfg)
                self.deployed = True
            elif isinstance(self.channel_cfg, (list, tuple)):

                self.pruner.convert_switchable_bn(self.architecture,
                                                  len(self.channel_cfg))
                self.pruner.prepare_from_supernet(self.architecture)
            else:
                raise NotImplementedError
        else:
            self.pruner.prepare_from_supernet(self.architecture)

    def _init_flops(self):
        """Get flops information of the supernet."""
        flops_model = copy.deepcopy(self.architecture)
        flops_model.eval()
        if hasattr(flops_model, 'forward_dummy'):
            flops_model.forward = flops_model.forward_dummy
        else:
            raise NotImplementedError(
                'FLOPs counter is currently not currently supported with {}'.
                format(flops_model.__class__.__name__))

        flops, params = get_model_complexity_info(
            flops_model, self.input_shape, print_per_layer_stat=False)
        flops_lookup = dict()
        for name, module in flops_model.named_modules():
            flops = getattr(module, '__flops__', 0)
            flops_lookup[name] = flops
        del (flops_model)

        for name, module in self.architecture.named_modules():
            module.__flops__ = flops_lookup[name]

    def get_subnet_flops(self):
        """A hacky way to get flops information of a subnet."""
        flops = 0
        last_out_mask_ratio = None
        for name, module in self.architecture.named_modules():
            if type(module) in [
                    nn.Conv2d, mmcv.cnn.bricks.Conv2d, nn.Linear,
                    mmcv.cnn.bricks.Linear
            ]:
                in_mask_ratio = float(module.in_mask.sum() /
                                      module.in_mask.numel())
                out_mask_ratio = float(module.out_mask.sum() /
                                       module.out_mask.numel())
                flops += module.__flops__ * in_mask_ratio * out_mask_ratio
                last_out_mask_ratio = out_mask_ratio
            elif type(module) == nn.BatchNorm2d:
                out_mask_ratio = float(module.out_mask.sum() /
                                       module.out_mask.numel())
                flops += module.__flops__ * out_mask_ratio
                last_out_mask_ratio = out_mask_ratio
            elif type(module) in [
                    nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6
            ]:

                assert last_out_mask_ratio, 'An activate module can not be ' \
                                            'the first module of a network.'
                flops += module.__flops__ * last_out_mask_ratio

        return round(flops)

    def train_step(self, data, optimizer):
        """Train step function.

        This function implements the standard training iteration for
        autoslim pretraining and retraining.

        Args:
            data (dict): Input data from dataloader.
            optimizer (:obj:`torch.optim.Optimizer`): The optimizer to
                accumulate gradient
        """
        optimizer.zero_grad()

        losses = dict()
        if not self.retraining:
            assert self.pruner is not None

            self.pruner.set_max_channel()
            if self.distiller is not None:
                max_model_losses = self.distiller.exec_teacher_forward(
                    self.architecture, data)
            else:
                max_model_losses = self(**data)
            losses.update(add_prefix(max_model_losses, 'max_model'))
            max_model_loss, _ = self._parse_losses(max_model_losses)
            max_model_loss.backward()

            self.pruner.set_min_channel()
            if self.distiller is not None:
                self.distiller.exec_student_forward(self.architecture, data)
                min_model_losses = self.distiller.compute_distill_loss(data)
            else:
                min_model_losses = self(**data)
            losses.update(add_prefix(min_model_losses, 'min_model'))
            min_model_loss, _ = self._parse_losses(min_model_losses)
            min_model_loss.backward()

            for i in range(self.num_sample_training - 2):
                subnet_dict = self.pruner.sample_subnet()
                self.pruner.set_subnet(subnet_dict)
                if self.distiller is not None:
                    self.distiller.exec_student_forward(
                        self.architecture, data)
                    model_losses = self.distiller.compute_distill_loss(data)
                    losses.update(
                        add_prefix(model_losses,
                                   'prune_model{}_distiller'.format(i + 1)))
                else:
                    model_losses = self(**data)
                    losses.update(
                        add_prefix(model_losses,
                                   'prune_model{}'.format(i + 1)))
                model_loss, _ = self._parse_losses(model_losses)
                model_loss.backward()
        else:
            if self.deployed:
                # Only one subnet retrains. The supernet has already deploy
                model_losses = self(**data)
                losses.update(add_prefix(model_losses, 'prune_model'))
                model_loss, _ = self._parse_losses(model_losses)
                model_loss.backward()
            else:
                # More than one subnet retraining together
                assert isinstance(self.channel_cfg, (list, tuple))
                for i, subnet in enumerate(self.channel_cfg):
                    self.pruner.switch_subnet(subnet, i)
                    model_losses = self(**data)
                    losses.update(
                        add_prefix(model_losses,
                                   'prune_model_{}'.format(i + 1)))
                    model_loss, _ = self._parse_losses(model_losses)
                    model_loss.backward()

        # TODO: clip grad norm
        optimizer.step()

        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))

        return outputs

    def train(self, mode=True):
        """Overwrite the train method in ``nn.Module`` to set ``nn.BatchNorm``
        to training mode when model is set to eval mode when
        ``self.bn_training_mode`` is ``True``.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                mode (``False``). Default: ``True``.
        """
        super(AutoSlim, self).train(mode)
        if not mode and self.bn_training_mode:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.training = True
