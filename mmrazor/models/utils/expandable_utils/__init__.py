# Copyright (c) OpenMMLab. All rights reserved.
"""This module is used to expand the channels of a supernet.

We only expose some tool functions, rather than all DynamicOps and
MutableChannelUnits, as They uses a few hacky operations.
"""
from .tools import (expand_expandable_dynamic_model, expand_static_model,
                    make_channel_divisible, to_expandable_model)

__all__ = [
    'make_channel_divisible',
    'to_expandable_model',
    'expand_expandable_dynamic_model',
    'expand_static_model',
]
