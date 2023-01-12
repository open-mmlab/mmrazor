# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Dict

import torch
from mmengine.model import BaseModule

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

    @abstractmethod
    def prepare(self, model, graph_module):
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
