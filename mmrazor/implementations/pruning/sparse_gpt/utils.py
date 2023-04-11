# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Protocol, Type

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops import DynamicMixin
from mmrazor.models.utils import get_module_device


class ModuleProtocol(Protocol):
    weight: torch.Tensor

    def forward(self, x):
        pass

    def register_forward_hook(self, hook):
        pass

    def register_backward_hook(self, hook):
        pass

    def register_forward_pre_hook(self, hook):
        pass

    def register_buffer(self, name, tensor):
        pass


def replace_with_dynamic_ops(model: nn.Module,
                             dynamicop_map: Dict[Type[nn.Module],
                                                 Type[DynamicMixin]]):
    """Replace torch modules with dynamic-ops."""

    def replace_op(model: nn.Module, name: str, module: nn.Module):
        names = name.split('.')
        for sub_name in names[:-1]:
            model = getattr(model, sub_name)

        setattr(model, names[-1], module)

    for name, module in model.named_modules():
        if type(module) in dynamicop_map:
            new_module = dynamicop_map[type(module)].convert_from(module).to(
                get_module_device(module))
            replace_op(model, name, new_module)
