# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from ..builder import DISTILLERS, build_loss
from .base import BaseDistiller


@DISTILLERS.register_module()
class SelfDistiller(BaseDistiller):
    """Transfer knowledge inside a single model.

    Args:
        components (dict): The details of the distillation. It usually includes
            the module names of the teacher and the student, and the losses
            used in the distillation.
    """

    def __init__(self, components, **kwargs):
        super().__init__(**kwargs)
        self.components = components
        self.losses = nn.ModuleDict()

        self.student_outputs = dict()
        self.teacher_outputs = dict()

        for component in self.components:
            student_module_name = component['student_module']
            teacher_module_name = component['teacher_module']
            self.student_outputs[student_module_name] = list()
            self.teacher_outputs[teacher_module_name] = list()

            for loss in component.losses:
                loss_cfg = loss.copy()
                loss_name = loss_cfg.pop('name')
                self.losses[loss_name] = build_loss(loss_cfg)

    def prepare_from_student(self, student):
        """Registers a global forward hook for each teacher module and student
        module to be used in the distillation.

        Args:
            student (:obj:`torch.nn.Module`): The student model to be used
                in the distillation.
        """
        self.module2name = {}
        for name, module in student.model.named_modules():
            self.module2name[module] = name
        self.name_modules = dict(student.model.named_modules())

        for component in self.components:
            student_module_name = component['student_module']
            teacher_module_name = component['teacher_module']

            student_module = self.name_modules[student_module_name]
            teacher_module = self.name_modules[teacher_module_name]

            student_module.register_forward_hook(
                self.student_forward_output_hook)
            teacher_module.register_forward_hook(
                self.teacher_forward_output_hook)

    def teacher_forward_output_hook(self, module, inputs, outputs):
        """Save the output.

        Args:
            module (:obj:`torch.nn.Module`): the module of register hook
            inputs (tuple): input of module
            outputs (tuple): out of module
        """
        if self.training and getattr(self, 'is_teacher', None):
            self.teacher_outputs[self.module2name[module]].append(outputs)

    def student_forward_output_hook(self, module, inputs, outputs):
        """Save the output.

        Args:
            module (:obj:`torch.nn.Module`): the module of register hook
            inputs (tuple): input of module
            outputs (tuple): out of module
        """
        if self.training and not getattr(self, 'is_teacher', None):
            self.student_outputs[self.module2name[module]].append(outputs)

    def reset_outputs(self, outputs):
        """Reset the teacher's outputs or student's outputs."""
        for key in outputs.keys():
            outputs[key] = list()

    def exec_teacher_forward(self, teacher, data):
        """Forward computation of the teacher.

        Args:
            teacher (:obj:`torch.nn.Module`): The teacher model to be used
                in the distillation.
            data (dict): The output of dataloader.
        """
        self.reset_outputs(self.teacher_outputs)
        self.is_teacher = True
        output = teacher(**data)
        self.is_teacher = False

        return output

    def exec_student_forward(self, student, data):
        """Forward computation of the student.

        Args:
            student (:obj:`torch.nn.Module`): The student model to be used
                in the distillation.
            data (dict): The output of dataloader.
        """
        assert not self.is_teacher
        self.reset_outputs(self.student_outputs)
        output = student(**data)

        return output

    def compute_distill_loss(self, data):
        """Compute the distillation loss."""

        losses = dict()

        for i, component in enumerate(self.components):
            student_module_name = component['student_module']
            student_outputs = self.student_outputs[student_module_name]

            teacher_module_name = component['teacher_module']
            teacher_outputs = self.teacher_outputs[teacher_module_name]

            for out_idx, (s_out, t_out) in enumerate(
                    zip(student_outputs, teacher_outputs)):

                for loss in component.losses:
                    loss_module = self.losses[loss.name]
                    loss_name = f'{loss.name}.{out_idx}'

                    loss_module.current_data = data
                    losses[loss_name] = loss_module(s_out, t_out)
                    loss_module.current_data = None

        return losses
