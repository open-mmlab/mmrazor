# Copyright (c) OpenMMLab. All rights reserved.
try:
    import mmdet
except (ImportError, ModuleNotFoundError):
    mmdet = None

if mmdet is not None:
    from .train import train_detector

    __all__ = ['train_detector']
