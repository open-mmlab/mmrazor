import torch.ao.quantization.observer as torch_observer_src
import inspect
from typing import List
from mmrazor.registry import MODELS

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
        if inspect.isclass(_observer) and issubclass(_observer,
                                            torch_observer_src.ObserverBase):
            if MODELS.get(module_name) is None:
                MODELS.register_module(module=_observer)
                torch_observers.append(module_name)
    return torch_observers

TORCH_observers = register_torch_observers()
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
