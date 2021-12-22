# Copyright (c) OpenMMLab. All rights reserved.
try:
    import mmdet
except (ImportError, ModuleNotFoundError):
    mmdet = None

if mmdet is not None:
    from .train import set_random_seed, train_detector

    __all__ = ['set_random_seed', 'train_detector']
