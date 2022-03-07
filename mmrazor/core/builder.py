# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

SEARCHERS = Registry('search')


def build_searcher(cfg, default_args=None):
    """Build searcher."""
    return build_from_cfg(cfg, SEARCHERS, default_args=default_args)
