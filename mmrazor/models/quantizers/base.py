# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

import torch
from mmengine.model import BaseModule

from mmrazor.models.task_modules.tracer.fx import (
    del_fakequant_after_function, del_fakequant_after_method,
    del_fakequant_after_module, del_fakequant_after_op,
    del_fakequant_before_function, del_fakequant_before_method,
    del_fakequant_before_module, del_fakequant_before_op)
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

    def del_fakequant(self, prepared):
        prepared = del_fakequant_before_module(
            prepared, self.module_del_prev_fakequant, inplace=True)
        prepared = del_fakequant_after_module(
            prepared, self.module_del_next_fakequant, inplace=True)
        prepared = del_fakequant_before_method(
            prepared, self.method_del_prev_fakequant, inplace=True)
        prepared = del_fakequant_after_method(
            prepared, self.method_del_next_fakequant, inplace=True)
        prepared = del_fakequant_before_function(
            prepared, self.function_del_prev_fakequant, inplace=True)
        prepared = del_fakequant_after_function(
            prepared, self.function_del_next_fakequant, inplace=True)
        prepared = del_fakequant_before_op(
            prepared, self.op_del_prev_fakequant, inplace=True)
        prepared = del_fakequant_after_op(
            prepared, self.op_del_next_fakequant, inplace=True)
        return prepared

    @property
    def module_del_prev_fakequant(self):
        return tuple()

    @property
    def module_del_next_fakequant(self):
        return tuple()

    @property
    def function_del_prev_fakequant(self):
        return tuple()

    @property
    def function_del_next_fakequant(self):
        return tuple()

    @property
    def method_del_prev_fakequant(self):
        return tuple()

    @property
    def method_del_next_fakequant(self):
        return tuple()

    @property
    def op_del_prev_fakequant(self):
        return tuple()

    @property
    def op_del_next_fakequant(self):
        return tuple()
