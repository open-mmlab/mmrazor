import torch
import torch.nn as nn


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


class efficient_forward:

    def __init__(self, model: nn.Module,
                 device=torch.device('cuda:0')) -> None:
        self.model = model
        self.device = device

    def __enter__(self, ):
        handles = []
        for module in self.model.modules():
            if len(module._parameters) != 0 or len(module._buffers) != 0:
                handles += register_efficient_forward_hook(module, self.device)
        self.handlers = handles

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for h in self.handlers:
            h.remove()
