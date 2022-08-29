# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine.structures import BaseDataElement
from torch import nn

from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODELS
from ...base import BaseAlgorithm, LossResults


@MODELS.register_module()
class SelfDistill(BaseAlgorithm):
    """``SelfDistill`` can be used to develop distill algorithms without
    teacher.

    Args:
        distiller (dict): The config dict for built distiller. Distiller may
            have teacher.
        student_trainable (bool): Whether the student is trainable. Defaults
            to True.
        calculate_student_loss (bool): Whether to calculate student loss
            (original task loss) to update student model. Defaults to True.
    """

    def __init__(self,
                 distiller: dict,
                 student_trainable: bool = True,
                 calculate_student_loss: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.distiller = MODELS.build(distiller)
        # The student model will not calculate gradients and update parameters
        # in some pretraining process.
        self.student_trainable = student_trainable

        # The student loss will not be updated into ``losses`` in some
        # pretraining process.
        self.calculate_student_loss = calculate_student_loss

        # In ``ConfigurableDistller``, the recorder manager is just
        # constructed, but not really initialized yet.
        self.distiller.prepare_from_student(self.student)
        # Still prepare from self-teacher. Teacher recorders of
        # ``SelfDistiller`` hook from self.student but require detach().
        self.distiller.prepare_from_teacher(self.student)

    @property
    def student(self) -> nn.Module:
        """Alias for ``architecture``."""
        return self.architecture

    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""

        losses = dict()

        # If the `override_data` of a delivery is True, the delivery will
        # override the origin data with the recorded data.
        self.distiller.set_deliveries_override(True)
        # Original task loss will not be used during some pretraining process.
        if self.calculate_student_loss:
            # teacher_recorders hook from student
            with self.distiller.student_recorders, \
                    self.distiller.teacher_recorders, \
                    self.distiller.deliveries:
                student_losses = self.student(
                    batch_inputs, data_samples, mode='loss')
            losses.update(add_prefix(student_losses, 'student'))
        else:
            with self.distiller.student_recorders, \
                    self.distiller.teacher_recorders, \
                    self.distiller.deliveries:
                if self.student_trainable:
                    _ = self.student(batch_inputs, data_samples, mode='loss')
                else:
                    with torch.no_grad():
                        _ = self.student(
                            batch_inputs, data_samples, mode='loss')

        # Automatically compute distill losses based on `loss_forward_mappings`
        # The required data already exists in the recorders.
        distill_losses = self.distiller.compute_distill_losses()
        losses.update(add_prefix(distill_losses, 'distill'))

        return losses
