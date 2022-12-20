# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseObserver
from .torch_observers import register_torch_observers

__all__ = ['BaseObserver', 'register_torch_observers']
