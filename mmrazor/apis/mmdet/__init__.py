# Copyright (c) OpenMMLab. All rights reserved.
try:
    import mmdet
except (ImportError, ModuleNotFoundError):
    mmdet = None

if mmdet is not None:
    from .inference import init_mmdet_model
    from .train import train_mmdet_model

    __all__ = ['train_mmdet_model', 'init_mmdet_model']
