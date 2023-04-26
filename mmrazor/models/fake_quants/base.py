# Copyright (c) OpenMMLab. All rights reserved.
try:
    from torch.ao.quantization import FakeQuantize
except ImportError:
    from mmrazor.utils import get_placeholder
    FakeQuantize = get_placeholder('torch>=1.13')

BaseFakeQuantize = FakeQuantize
