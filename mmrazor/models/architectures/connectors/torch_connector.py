# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from .base_connector import BaseConnector

FUNCTION_LIST = [
    'adaptive_avg_pool2d',
    'adaptive_max_pool2d',
    'avg_pool2d',
    'dropout',
    'dropout2d',
    'max_pool2d',
    'normalize',
    'relu',
    'softmax',
    'interpolate',
]


@MODELS.register_module()
class TorchFunctionalConnector(BaseConnector):
    """TorchFunctionalConnector: Call function in torch.nn.functional
    to process input data

    usage:
        tensor1 = torch.rand(3,3,16,16)
        pool_connector = TorchFunctionalConnector(
                            function_name='avg_pool2d',
                            func_args=dict(kernel_size=4),
                        )
        tensor2 = pool_connector.forward_train(tensor1)
        tensor2.size()
        # torch.Size([3, 3, 4, 4])

        which is equal to torch.nn.functional.avg_pool2d(kernel_size=4)
    Args:
        function_name (str, optional): function. Defaults to None.
        func_args (dict): args parsed to function. Defaults to {}.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self,
                 function_name: Optional[str] = None,
                 func_args: Dict = {},
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)
        assert function_name is not None, 'Arg `function_name` cannot be None'
        if function_name not in FUNCTION_LIST:
            raise ValueError(
                ' Arg `function_name` are not available, See this list',
                FUNCTION_LIST)
        self.func = getattr(F, function_name)
        self.func_args = func_args

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Frorward function for training.

        Args:
            x (torch.Tensor): Input features.
        """
        x = self.func(x, **self.func_args)
        return x


MODULE_LIST = [
    'AdaptiveAvgPool2d',
    'AdaptiveMaxPool2d',
    'AvgPool2d',
    'BatchNorm2d',
    'Conv2d',
    'Dropout',
    'Dropout2d',
    'Linear',
    'MaxPool2d',
    'ReLU',
    'Softmax',
]


@MODELS.register_module()
class TorchNNConnector(BaseConnector):
    """TorchNNConnector: create nn.module in torch.nn to process input data

    usage:
        tensor1 = torch.rand(3,3,16,16)
        pool_connector = TorchNNConnector(
                            module_name='AvgPool2d',
                            module_args=dict(kernel_size=4),
                        )
        tensor2 = pool_connector.forward_train(tensor1)
        tensor2.size()
        # torch.Size([3, 3, 4, 4])

        which is equal to torch.nn.AvgPool2d(kernel_size=4)
    Args:
        module_name (str, optional):
            module name. Defaults to None.
            possible_values:['AvgPool2d',
                            'Dropout2d',
                            'AdaptiveAvgPool2d',
                            'AdaptiveMaxPool2d',
                            'ReLU',
                            'Softmax',
                            'BatchNorm2d',
                            'Linear',]
        module_args (dict):
            args parsed to nn.Module().__init__(). Defaults to {}.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self,
                 module_name: Optional[str] = None,
                 module_args: Dict = {},
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg)
        assert module_name is not None, 'Arg `module_name` cannot be None'
        if module_name not in MODULE_LIST:
            raise ValueError(
                ' Arg `module_name` are not available, See this list',
                MODULE_LIST)
        self.func = getattr(nn, module_name)(**module_args)

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        """Frorward function for training.

        Args:
            x (torch.Tensor): Input features.
        """
        x = self.func(x)
        return x
