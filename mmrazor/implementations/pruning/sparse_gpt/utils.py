# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Protocol, Type

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops import DynamicMixin
from mmrazor.models.utils import get_module_device
from mmrazor.utils import print_log


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


def register_efficient_forward_hook(module: nn.Module,
                                    device=torch.device('cuda:0')):

    def forward_pre_hook(module: nn.Module, input):
        module.to(device)

    def forward_hook(module: nn.Module, input, output):
        module.to('cpu')
        torch.cuda.empty_cache()

    h1 = module.register_forward_pre_hook(forward_pre_hook)
    h2 = module.register_forward_hook(forward_hook)
    return [h1, h2]


def enable_efficient_forward(model: nn.Module,
                             device=torch.device('cuda:0'),
                             wrap_modules=[]):
    handles = []
    blocks = []
    for name, module in model.named_children():
        if type(module) in wrap_modules or len(module._parameters) != 0 or len(
                module._buffers) != 0:
            handles_ = register_efficient_forward_hook(module, device)
            blocks_ = [name]
        else:
            handles_, blocks_ = enable_efficient_forward(
                module, device, wrap_modules)
        handles += handles_
        blocks += blocks_
    return handles, blocks


class memory_efficient_forward:

    def __init__(self,
                 model: nn.Module,
                 device=torch.device('cuda:0'),
                 wrap_modules=[]) -> None:
        self.model = model
        self.device = device
        self.wrap_modules = wrap_modules

    def __enter__(self, ):
        handles, blocks = enable_efficient_forward(self.model, self.device,
                                                   self.wrap_modules)
        print_log(f'enable memory efficient forward for {blocks}')
        self.handlers = handles

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for h in self.handlers:
            h.remove()
