# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from typing import List

import torch

from mmrazor.registry import MODELS

try:
    import torch.ao.quantization.observer as torch_observer_src
    from torch.ao.quantization.observer import PerChannelMinMaxObserver
except ImportError:
    from mmrazor.utils import get_package_placeholder
    torch_observer_src = get_package_placeholder('torch>=1.13')
    PerChannelMinMaxObserver = get_package_placeholder('torch>=1.13')


@torch.jit.export
def reset_min_max_vals(self):
    """Resets the min/max values.

    `min_val` and `max_val` are always be on cpu in the pytorch version of this
    method.
    """
    min_val = torch.rand(0, )
    max_val = torch.rand(0, )
    self.min_val.resize_(min_val.shape).copy_(min_val)
    self.max_val.resize_(max_val.shape).copy_(max_val)


PerChannelMinMaxObserver.reset_min_max_vals = reset_min_max_vals


# TORCH_observers = register_torch_observers()
# TORCH_observers including:
# FixedQParamsObserver
# HistogramObserver
# MinMaxObserver
# MovingAverageMinMaxObserver
# MovingAveragePerChannelMinMaxObserver
# NoopObserver
# ObserverBase
# PerChannelMinMaxObserver
# PlaceholderObserver
# RecordingObserver
# ReuseInputObserver
# UniformQuantizationObserverBase
def register_torch_observers() -> List[str]:
    """Register observers in ``torch.ao.quantization.observer`` to the
    ``MODELS`` registry.

    Returns:
        List[str]: A list of registered observers' name.
    """
    torch_observers = []
    for module_name in dir(torch_observer_src):
        if module_name.startswith('__') or module_name.startswith('_') or \
                                            module_name.startswith('default'):
            continue
        _observer = getattr(torch_observer_src, module_name)
        if inspect.isclass(_observer) and issubclass(
                _observer, torch_observer_src.ObserverBase):
            if MODELS.get(module_name) is None:
                MODELS.register_module(module=_observer)
                torch_observers.append(module_name)
    return torch_observers
