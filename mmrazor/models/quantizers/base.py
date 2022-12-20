# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

import torch
from mmengine.model import BaseModule

from mmrazor.registry import TASK_UTILS


class BaseQuantizer(BaseModule):

    def __init__(self, tracer):
        super().__init__()
        self.tracer = TASK_UTILS.build(tracer)

    @abstractmethod
    def prepare(self):
        pass

    def swap_ff_with_fxff(self, model):
        r""" Swap FloatFunctional with FXFloatFunctional
        """
        modules_to_swap = []
        for name, module in model.named_children():
            if isinstance(module, torch.ao.nn.quantized.FloatFunctional):
                modules_to_swap.append(name)
            else:
                self.swap_ff_with_fxff(module)

        for name in modules_to_swap:
            del model._modules[name]
            model._modules[name] = torch.ao.nn.quantized.FXFloatFunctional()
