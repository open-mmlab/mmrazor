# Copyright (c) OpenMMLab. All rights reserved.
from .epoch_based_runner import MultiLoaderEpochBasedRunner
from .iter_based_runner import MultiLoaderIterBasedRunner

__all__ = ['MultiLoaderEpochBasedRunner', 'MultiLoaderIterBasedRunner']
