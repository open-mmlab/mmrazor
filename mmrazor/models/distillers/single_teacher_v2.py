# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.core import RecorderManager
from ..builder import DISTILLERS, MODELS, build_loss


@DISTILLERS.register_module()
class SingleTeacherDistillerV2(BaseModule):
    """Distiller with single teacher.

    Args:
        teacher (dict): The config dict for teacher.
        teacher_trainable (bool): Whether the teacher is trainable.
            Default: False.
        teacher_norm_eval (bool): Whether to set teacher's norm layers to eval
            mode, namely, freeze running stats (mean and var). Note: Effect on
            Batch Norm and its variants only. Default: True.
        components (dict): The details of the distillation. It usually includes
            the module names of the teacher and the student, and the losses
            used in the distillation.
    """

    def __init__(self,
                 teacher,
                 student_recorders=tuple(),
                 teacher_recorders=tuple(),
                 teacher_trainable=False,
                 teacher_norm_eval=True,
                 components=tuple(),
                 **kwargs):
        super().__init__(**kwargs)
        self.teacher_trainable = teacher_trainable
        self.teacher_norm_eval = teacher_norm_eval
        self.teacher = self.build_teacher(teacher)

        self.student_recorder_manager = RecorderManager(student_recorders)
        self.teacher_recorder_manager = RecorderManager(teacher_recorders)

        self.components = components
        self.losses = nn.ModuleDict()

        for i, component in enumerate(self.components):
            loss_name = f'loss_{i}'
            self.losses[loss_name] = build_loss(component.loss)

    def build_teacher(self, cfg):
        """Build a model from the `cfg`."""

        teacher = MODELS.build(cfg)

        return teacher

    def prepare_from_student(self, student):
        pass

    def exec_teacher_forward(self, data):
        """Execute the teacher's forward function.

        After this function, the teacher's featuremaps will be saved in
        ``teacher_outputs``.
        """

        with self.teacher_recorder_manager(self.teacher):
            if self.teacher_trainable:
                output = self.teacher(**data)
            else:
                with torch.no_grad():
                    output = self.teacher(**data)

        return output

    def exec_student_forward(self, student, data):
        """Execute the teacher's forward function.

        After this function, the student's featuremaps will be saved in
        ``student_outputs``.
        """

        with self.student_recorder_manager(student.architecture.model):
            output = student(**data)
        return output

    def train(self, mode=True):
        """Set distiller's forward mode."""
        super(SingleTeacherDistillerV2, self).train(mode)
        if mode and self.teacher_norm_eval:
            for m in self.teacher.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def compute_distill_loss(self, data=None):
        """Compute the distillation loss."""

        losses = dict()

        for i, component in enumerate(self.components):
            # Get the student's outputs.
            student_items = list()
            for item_cfg in component.student_items:
                item = self.student_recorder_manager.get_record_data(
                    **item_cfg)
                if isinstance(item, (list, tuple)) and len(item) == 1:
                    student_items.append(item[0])
                else:
                    student_items.append(item)

            # Get the teacher's outputs.
            teacher_items = list()
            for item_cfg in component.teacher_items:
                item = self.teacher_recorder_manager.get_record_data(
                    **item_cfg)
                if isinstance(item, (list, tuple)) and len(item) == 1:
                    teacher_items.append(item[0])
                else:
                    teacher_items.append(item)

            loss_name = f'loss_{i}'
            loss_module = self.losses[loss_name]
            # TODO ugly implementation.
            # Pass the gt_label to loss function.
            # Only used by WSLD.
            loss_module.current_data = data
            losses[loss_name] = loss_module(*student_items, *teacher_items)
            loss_module.current_data = None

        return losses
