# Copyright (c) OpenMMLab. All rights reserved.
import copy
from functools import partial

from mmcv.cnn import get_model_complexity_info
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.builder import ALGORITHMS
from .base import BaseAlgorithm


@ALGORITHMS.register_module()
class SPOS(BaseAlgorithm):
    """Implementation of `SPOS <https://arxiv.org/abs/1904.00420>`_"""

    def __init__(self,
                 input_shape=(3, 224, 224),
                 bn_training_mode=False,
                 **kwargs):

        super(SPOS, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.bn_training_mode = bn_training_mode
        if not self.retraining:
            self._init_flops()
        self.apply(partial(self.mutator.reset_in_subnet, in_subnet=True))

    def _init_flops(self):
        """Get flops of all modules in supernet in order to easily get each
        subnet's flops."""
        flops_model = copy.deepcopy(self.architecture)
        flops_model.eval()
        if hasattr(flops_model, 'forward_dummy'):
            flops_model.forward = flops_model.forward_dummy
        else:
            raise NotImplementedError(
                'FLOPs counter is currently not currently supported with {}'.
                format(flops_model.__class__.__name__))

        flops, params = get_model_complexity_info(flops_model,
                                                  self.input_shape)
        flops_lookup = dict()
        for name, module in flops_model.named_modules():
            flops = getattr(module, '__flops__', 0)
            flops_lookup[name] = flops
        del (flops_model)

        for name, module in self.architecture.named_modules():
            module.__flops__ = flops_lookup[name]

    def get_subnet_flops(self):
        """Get subnet's flops based on the complexity information of
        supernet."""
        flops = 0
        for name, module in self.architecture.named_modules():
            if module.__in_subnet__:
                flops += getattr(module, '__flops__', 0)
        return flops

    def train_step(self, data, optimizer):
        """The iteration step during training.

        In retraining stage, to train subnet like common model. In pre-training
        stage, First to sample a subnet from supernet, then to train the
        subnet.
        """
        if self.retraining:
            outputs = super(SPOS, self).train_step(data, optimizer)
        else:
            subnet_dict = self.mutator.sample_subnet()
            self.mutator.set_subnet(subnet_dict)
            outputs = super(SPOS, self).train_step(data, optimizer)
        return outputs

    def train(self, mode=True):
        """Overwrite the train method in `nn.Module` to set `nn.BatchNorm` to
        training mode when model is set to eval mode when
        `self.bn_training_mode` is `True`.

        Args:
            mode (bool): whether to set training mode (`True`) or evaluation
                mode (`False`). Default: `True`.
        """
        super(SPOS, self).train(mode)
        if not mode and self.bn_training_mode:
            for module in self.modules():
                if isinstance(module, _BatchNorm):
                    module.training = True
