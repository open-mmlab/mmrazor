# Copyright (c) OpenMMLab. All rights reserved.
try:
    import mmseg
except (ImportError, ModuleNotFoundError):
    mmseg = None

if mmseg:
    from .train import set_random_seed, train_segmentor

    __all__ = ['set_random_seed', 'train_segmentor']
