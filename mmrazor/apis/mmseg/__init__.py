# Copyright (c) OpenMMLab. All rights reserved.
try:
    import mmseg
except (ImportError, ModuleNotFoundError):
    mmseg = None

if mmseg:
    from .inference import init_mmseg_model

    __all__ = ['init_mmseg_model']
