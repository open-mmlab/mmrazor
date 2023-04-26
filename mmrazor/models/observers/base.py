# Copyright (c) OpenMMLab. All rights reserved.
try:
    from torch.ao.quantization.observer import UniformQuantizationObserverBase
except ImportError:
    from mmrazor.utils import get_placeholder
    UniformQuantizationObserverBase = get_placeholder('torch>=1.13')

BaseObserver = UniformQuantizationObserverBase
