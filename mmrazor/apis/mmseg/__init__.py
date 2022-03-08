# Copyright (c) OpenMMLab. All rights reserved.
try:
    import mmseg
except (ImportError, ModuleNotFoundError):
    mmseg = None

if mmseg:
    from .inference import init_mmseg_model
    from .train import set_random_seed, train_segmentor

    __all__ = ['set_random_seed', 'train_segmentor', 'init_mmseg_model']
