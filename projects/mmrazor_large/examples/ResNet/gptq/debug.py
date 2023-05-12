from typing import Protocol

import torch

from mmrazor.implementations.quantization.gptq import GPTQMixIn


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


m = ModuleProtocol
print(hasattr(m, 'weight'))
a = GPTQMixIn()
print(hasattr(a, 'weight'))
