# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from mmengine.model import BaseModel

from mmrazor.registry import MODELS
from ....distillers import OFDDistiller
from .single_teacher_distill import SingleTeacherDistill


@MODELS.register_module()
class OverhaulFeatureDistillation(SingleTeacherDistill):
    """`A Comprehensive Overhaul of Feature Distillation`
    https://sites.google.com/view/byeongho-heo/overhaul.

    Inherited from ``SingleTeacherDistill``.


    Args:
        distiller (dict): The config dict for built distiller. Must be a
            ``OFDDistiller``.
        teacher (dict | BaseModel): The config dict for teacher model or built
            teacher model.
        teacher_ckpt (str): The path of teacher's checkpoint. Defaults to None.
        teacher_trainable (bool): Whether the teacher is trainable. Defaults
            to False.
        teacher_norm_eval (bool): Whether to set teacher's norm layers to eval
            mode, namely, freeze running stats (mean and var). Note: Effect on
            Batch Norm and its variants only. Defaults to True.
        student_trainable (bool): Whether the student is trainable. Defaults
            to True.
        calculate_student_loss (bool): Whether to calculate student loss
            (original task loss) to update student model. Defaults to True.
    """

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
