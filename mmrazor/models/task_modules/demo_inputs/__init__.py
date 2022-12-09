# Copyright (c) OpenMMLab. All rights reserved.
from .default_demo_inputs import DefaultDemoInput, defaul_demo_inputs
from .demo_inputs import (BaseDemoInput, DefaultMMClsDemoInput,
                          DefaultMMDemoInput, DefaultMMDetDemoInput,
                          DefaultMMSegDemoInput)

__all__ = [
    'defaul_demo_inputs',
    'DefaultMMClsDemoInput',
    'DefaultMMDetDemoInput',
    'DefaultMMDemoInput',
    'DefaultMMSegDemoInput',
    'BaseDemoInput',
    'DefaultDemoInput',
]
