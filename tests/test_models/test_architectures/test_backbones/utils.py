# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from torch import Tensor
from torch.nn import Conv2d, Module

from mmrazor.registry import MODELS


@MODELS.register_module()
class MockMutable(Module):

    def __init__(self, choices: List[str], module_kwargs: Dict) -> None:
        super().__init__()

        self.choices = choices
        self.module_kwargs = module_kwargs
        self.conv = Conv2d(**module_kwargs, kernel_size=3, padding=3 // 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)
