# Copyright (c) OpenMMLab. All rights reserved.
from .inference import init_mmcls_model
from .train import set_random_seed, train_mmcls_model

__all__ = ['train_mmcls_model', 'init_mmcls_model', 'set_random_seed']
