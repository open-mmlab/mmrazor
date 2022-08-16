# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Union

import torch
import torch.nn as nn


def get_module_device(module: nn.Module) -> torch.device:
    """Get the device of a module.

    Args:
        module (nn.Module): A module contains the parameters.
    """
    try:
        next(module.parameters())
    except StopIteration as e:
        raise ValueError('The input module should contain parameters.') from e

    if next(module.parameters()).is_cuda:
        return next(module.parameters()).get_device()

    return torch.device('cpu')


def set_requires_grad(nets: Union[nn.Module, List[nn.Module]],
                      requires_grad: bool = False) -> None:
    """Set requires_grad for all the networks.

    Args:
        nets (nn.Module | list[nn.Module]): A list of networks or a single
            network.
        requires_grad (bool): Whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
