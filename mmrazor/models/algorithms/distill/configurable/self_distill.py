# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
from mmengine import BaseDataElement
from torch import nn

from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODELS
from ...base import BaseAlgorithm, LossResults


@MODELS.register_module()
class SelfDistill(BaseAlgorithm):
    """``SingleTeacherDistill`` can be used to develop distill algorithms which
    only use one teacher.

    Args:
        distiller (dict): The config dict for built distiller.
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
            with self.distiller.student_recorders, self.distiller.deliveries:
                student_losses = self.student(
                    batch_inputs, data_samples, mode='loss')
            losses.update(add_prefix(student_losses, 'student'))
        else:
            with self.distiller.student_recorders, self.distiller.deliveries:
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
