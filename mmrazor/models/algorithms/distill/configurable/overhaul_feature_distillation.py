# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from mmengine.model import BaseModel

from mmrazor.registry import MODELS
from ....distillers import OFDDistiller
from .single_teacher_distill import SingleTeacherDistill


@MODELS.register_module()
class OverhaulFeatureDistillation(SingleTeacherDistill):

    def __init__(self,
                 distiller: dict,
                 teacher: Union[BaseModel, Dict],
                 teacher_ckpt: Optional[str] = None,
                 teacher_trainable: bool = False,
                 teacher_norm_eval: bool = True,
                 student_trainable: bool = True,
                 calculate_student_loss: bool = True,
                 **kwargs) -> None:
        super().__init__(distiller, teacher, teacher_ckpt, teacher_trainable,
                         teacher_norm_eval, student_trainable,
                         calculate_student_loss, **kwargs)

        assert isinstance(self.distiller, OFDDistiller), (
            'distiller of `OverhaulFeatureDistillation` expects `OFDDistiller`'
            f', but get {type(self.distiller)}')

        self.distiller.init_ofd_connectors(self.teacher)
