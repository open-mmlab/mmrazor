# Copyright (c) OpenMMLab. All rights reserved.
try:
    import mmseg
except (ImportError, ModuleNotFoundError):
    mmseg = None

if mmseg:
    from .inference import init_mmseg_model
    from .train import set_random_seed, train_mmseg_model

    __all__ = ['train_mmseg_model', 'init_mmseg_model', 'set_random_seed']
