# Copyright (c) OpenMMLab. All rights reserved.
from .datafree_distillation import (DAFLDataFreeDistillation,
                                    DataFreeDistillation)
from .fpn_teacher_distill import FpnTeacherDistill
from .self_distill import SelfDistill
from .single_teacher_distill import SingleTeacherDistill

__all__ = ['SelfDistill', 'SingleTeacherDistill', 'FpnTeacherDistill',
           'DataFreeDistillation', 'DAFLDataFreeDistillation'
]
