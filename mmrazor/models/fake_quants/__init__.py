# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseFakeQuantize
from .torch_fake_quants import register_torch_fake_quants

__all__ = ['BaseFakeQuantize', 'register_torch_fake_quants']
