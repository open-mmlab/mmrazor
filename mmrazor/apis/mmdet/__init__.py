# Copyright (c) OpenMMLab. All rights reserved.
try:
    import mmdet
except (ImportError, ModuleNotFoundError):
    mmdet = None

if mmdet is not None:
    from .inference import init_mmdet_model

    __all__ = ['init_mmdet_model']
