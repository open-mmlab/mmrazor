# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.model.utils import _BatchNormXd

from mmrazor.registry import TASK_UTILS


class BaseQuantizer(BaseModule):
    """tmp."""

    def __init__(self, tracer, is_qat):
        super().__init__()
        self.tracer = TASK_UTILS.build(tracer)
        self.is_qat = is_qat

    def sync_module_training_mode(self, model, mode=True):
        """Synchronize the training modes.

        Note that modes of conv and bn must be the same during ``_fuse_fx``.
        """
        for module in model.modules():
            module.training = mode
        return

    @staticmethod
    def convert_batchnorm2d(model):
        """Helper function to convert all :attr:`_BatchNormXd` layers and
        :class:`torch.nn.SyncBatchNorm` layers in the model to
        :class:`torch.nn.BatchNorm2d` layers.
        """
        # todo: Convert all `_BatchNormXd` and `SyncBatchNorm`
        #  layers to `BatchNorm2d` layers but they may be :attr:`BatchNorm*D`
        #  layers
        module_checklist = [nn.modules.batchnorm.SyncBatchNorm, _BatchNormXd]

        def traverse(module: nn.Module):
            for child_name, child in module.named_children():
                if isinstance(child, tuple(module_checklist)):
                    bn = nn.BatchNorm2d(child.num_features, child.eps,
                                        child.momentum, child.affine,
                                        child.track_running_stats)
                    setattr(module, child_name, bn)
                else:
                    traverse(child)

        traverse(model)

    @abstractmethod
    def prepare(self, model, graph_module):
        """tmp."""
        pass

    def swap_ff_with_fxff(self, model):
        """Swap FloatFunctional with FXFloatFunctional."""
        modules_to_swap = []
        for name, module in model.named_children():
            if isinstance(module, torch.ao.nn.quantized.FloatFunctional):
                modules_to_swap.append(name)
            else:
                self.swap_ff_with_fxff(module)

        for name in modules_to_swap:
            del model._modules[name]
            model._modules[name] = torch.ao.nn.quantized.FXFloatFunctional()
