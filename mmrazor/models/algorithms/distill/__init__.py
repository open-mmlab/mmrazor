# Copyright (c) OpenMMLab. All rights reserved.
from .configurable import (DAFLDataFreeDistillation, DataFreeDistillation,
                           FpnTeacherDistill, SelfDistill,
                           SingleTeacherDistill)

__all__ = [
    'SingleTeacherDistill', 'FpnTeacherDistill', 'SelfDistill',
    'DataFreeDistillation', 'DAFLDataFreeDistillation'
]
