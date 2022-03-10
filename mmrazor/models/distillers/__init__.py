# Copyright (c) OpenMMLab. All rights reserved.
from .self_distiller import SelfDistiller
from .single_teacher import SingleTeacherDistiller
from .single_teacher_v2 import SingleTeacherDistillerV2

__all__ = [
    'SelfDistiller', 'SingleTeacherDistiller', 'SingleTeacherDistillerV2'
]
