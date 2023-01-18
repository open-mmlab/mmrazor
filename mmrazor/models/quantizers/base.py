# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmengine.model.utils import _BatchNormXd

from mmrazor.registry import TASK_UTILS


class BaseQuantizer(BaseModule):
    """Base class for quantizers. Its role for several subclass is as follows:
    1. Provide tracer for tracing model for all subclass.
    2. Define some common abstract methods, such as `prepare`.
    3. Provide some common functional interfaces, such as `swap_ff_with_fxff`.

    Args:
        tracer (Dict): It can be used to trace the float model to generate the
            corresponding graph, which contributes to prepare for quantizing
            the float model with code-free.
    """

    def __init__(self, tracer: Dict):
        super().__init__()
        self.tracer = TASK_UTILS.build(tracer)

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
    def prepare(self, model):
        """Prepare for quantizing model, which usually includes as follows:

        1. Swap floatfunctional with FXFloatFunctional;
        2. Trace model to generate `GraphModule`;
        2. Fuse some OPs combination, such as conv + bn, conv + relu and so on;
        3. Swap some conv or linear module with QAT Modules which contain
        weight fakequant nodes;
        4. Insert required fakequant nodes for activation.
        5. (Optional) Delete some redundant fakequant nodes according to the
        special requirement of the backend for deployment.
        """
        pass

    def swap_ff_with_fxff(self, model: torch.nn.Module):
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
