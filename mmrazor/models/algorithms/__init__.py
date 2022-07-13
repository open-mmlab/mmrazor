# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseAlgorithm
from .distill import FpnTeacherDistill, SingleTeacherDistill
from .nas import SPOS

__all__ = [
    'SingleTeacherDistill', 'BaseAlgorithm', 'FpnTeacherDistill', 'SPOS'
]
