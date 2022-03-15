# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import Registry, build_from_cfg

SEARCHERS = Registry('search')
RECORDERS = Registry('recorders')
DELIVERS = Registry('delivers')


def build_searcher(cfg, default_args=None):
    """Build searcher."""
    return build_from_cfg(cfg, SEARCHERS, default_args=default_args)


def build_recorder(cfg, default_args=None):
    """Build recorder."""
    return build_from_cfg(cfg, RECORDERS, default_args=default_args)

def build_deliver(cfg, default_args=None):
    """Build deliver."""
    return build_from_cfg(cfg, DELIVERS, default_args=default_args)
