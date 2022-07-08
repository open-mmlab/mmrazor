# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmcv.runner import load_checkpoint
from mmengine import BaseDataElement
from mmengine.model import BaseModel
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODELS
from ...base import LossResults
from .configurable_distill import ConfigurableDistill


@MODELS.register_module()
class SingleTeacherDistill(ConfigurableDistill):
    """``SingleTeacherDistill`` can be used to develop distill algorithms which
    only use one teacher.

    Args:
        teacher (dict | BaseModel): The config dict for teacher model or built
            teacher model.
        teacher_ckpt (str): The path of teacher's checkpoint. Defaults to None.
        teacher_trainable (bool): Whether the teacher is trainable. Defaults
            to False.
        teacher_norm_eval (bool): Whether to set teacher's norm layers to eval
            mode, namely, freeze running stats (mean and var). Note: Effect on
            Batch Norm and its variants only. Defaults to True.
    """

    def __init__(self,
                 teacher: Union[BaseModel, Dict],
                 teacher_ckpt: Optional[str] = None,
                 teacher_trainable: bool = False,
                 teacher_norm_eval: bool = True,
                 **kwargs):
        super().__init__(**kwargs)

        if isinstance(teacher, Dict):
            teacher = MODELS.build(teacher)

        if not isinstance(teacher, BaseModel):
            raise TypeError('teacher should be a `dict` or '
                            f'`BaseModel` instance, but got '
                            f'{type(teacher)}')

        self.teacher = teacher
        if teacher_ckpt:
            # avoid loaded parameters be overwritten
            self.teacher.init_weights()
            _ = load_checkpoint(self.teacher, teacher_ckpt)
        self.teacher_trainable = teacher_trainable
        self.teacher_norm_eval = teacher_norm_eval

        # In ``ConfigurableDistll``, the recorder manager is just constructed,
        # but not really initialized yet.
        self.student_recorders.initialize(self.student)
        self.teacher_recorders.initialize(self.teacher)

    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""

        losses = dict()

        # If the `override_data` of a delivery is False, the delivery will
        # record the origin data.
        self.distill_deliveries.override_data = False
        if self.teacher_trainable:
            with self.teacher_recorders, self.distill_deliveries:
                teacher_losses = self.teacher(
                    batch_inputs, data_samples, mode='loss')

            losses.update(add_prefix(teacher_losses, 'teacher'))
        else:
            with self.teacher_recorders, self.distill_deliveries:
                with torch.no_grad():

                    _ = self.teacher(batch_inputs, data_samples, mode='loss')

        # If the `override_data` of a delivery is True, the delivery will
        # override the origin data with the recorded data.
        self.distill_deliveries.override_data = True
        with self.student_recorders, self.distill_deliveries:
            student_losses = self.student(
                batch_inputs, data_samples, mode='loss')
        losses.update(add_prefix(student_losses, 'student'))

        # Automatically compute distill losses based on `loss_forward_mappings`
        distill_losses = self.compute_distill_losses(
            self.distill_losses, self.loss_forward_mappings,
            self.student_recorders, self.teacher_recorders)
        losses.update(add_prefix(distill_losses, 'distill'))

        return losses

    def train(self, mode=True):
        """Set distiller's forward mode."""
        super().train(mode)
        if mode and self.teacher_norm_eval:
            for m in self.teacher.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
