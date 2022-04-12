# Copyright (c) OpenMMLab. All rights reserved.
try:
    import mmdet3d
except (ImportError, ModuleNotFoundError):
    mmdet3d = None

if mmdet3d is not None:
    from .inference import init_mmdet3d_model
    from .train import set_random_seed, train_mmdet3d_model

    __all__ = ['train_mmdet3d_model', 'init_mmdet3d_model', 'set_random_seed']
