# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.runner import load_checkpoint
from mmengine.structures import BaseDataElement
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODELS
from ...base import BaseAlgorithm, LossResults


@MODELS.register_module()
class DFNDDistill(BaseAlgorithm):
    """``DFNDDistill`` algorithm for training student model in the wild dataset.
        https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Learning_Student_Networks_in_the_Wild_CVPR_2021_paper.pdf

    Args:
        distiller (dict): The config dict for built distiller.
        teacher (dict | BaseModel): The config dict for teacher model or built
            teacher model.
        val_data_preprocessor (Union[Dict, nn.Module]): Data preprocessor for
            evaluation dataset. Defaults to None.
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
        teacher_module_inplace(bool): Whether to allow teacher module inplace
            attribute True. Defaults to False.
    """

    def __init__(self,
                 distiller: dict,
                 teacher: Union[BaseModel, Dict],
                 val_data_preprocessor: Optional[Union[Dict,
                                                       nn.Module]] = None,
                 teacher_ckpt: Optional[str] = None,
                 teacher_trainable: bool = False,
                 teacher_norm_eval: bool = True,
                 student_trainable: bool = True,
                 calculate_student_loss: bool = True,
                 teacher_module_inplace: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.distiller = MODELS.build(distiller)

        if isinstance(teacher, Dict):
            teacher = MODELS.build(teacher)

        if not isinstance(teacher, BaseModel):
            raise TypeError('teacher should be a `dict` or '
                            f'`BaseModel` instance, but got '
                            f'{type(teacher)}')

        self.teacher = teacher

        # Find all nn.Modules in the model that contain the 'inplace' attribute
        # and set them to False.
        self.teacher_module_inplace = teacher_module_inplace
        if not self.teacher_module_inplace:
            self.set_module_inplace_false(teacher, 'self.teacher')

        if teacher_ckpt:
            _ = load_checkpoint(self.teacher, teacher_ckpt)
            # avoid loaded parameters be overwritten
            self.teacher._is_init = True
        self.teacher_trainable = teacher_trainable
        if not self.teacher_trainable:
            for param in self.teacher.parameters():
                param.requires_grad = False
        self.teacher_norm_eval = teacher_norm_eval

        # The student model will not calculate gradients and update parameters
        # in some pretraining process.
        self.student_trainable = student_trainable

        # The student loss will not be updated into ``losses`` in some
        # pretraining process.
        self.calculate_student_loss = calculate_student_loss

        # In ``ConfigurableDistller``, the recorder manager is just
        # constructed, but not really initialized yet.
        self.distiller.prepare_from_student(self.student)
        self.distiller.prepare_from_teacher(self.teacher)

        # may be modified by stop distillation hook
        self.distillation_stopped = False
        if val_data_preprocessor is None:
            val_data_preprocessor = dict(type='BaseDataPreprocessor')
        if isinstance(val_data_preprocessor, nn.Module):
            self.val_data_preprocessor = val_data_preprocessor
        elif isinstance(val_data_preprocessor, dict):
            self.val_data_preprocessor = MODELS.build(val_data_preprocessor)
        else:
            raise TypeError('val_data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(val_data_preprocessor)}')

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

        # If the `override_data` of a delivery is False, the delivery will
        # record the origin data.
        self.distiller.set_deliveries_override(False)
        if self.teacher_trainable:
            with self.distiller.teacher_recorders, self.distiller.deliveries:
                teacher_losses = self.teacher(
                    batch_inputs, data_samples, mode='loss')

            losses.update(add_prefix(teacher_losses, 'teacher'))
        else:
            with self.distiller.teacher_recorders, self.distiller.deliveries:
                with torch.no_grad():
                    _ = self.teacher(batch_inputs, data_samples, mode='tensor')

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
                    _ = self.student(batch_inputs, data_samples, mode='tensor')
                else:
                    with torch.no_grad():
                        _ = self.student(
                            batch_inputs, data_samples, mode='tensor')

        if not self.distillation_stopped:
            # Automatically compute distill losses based on
            # `loss_forward_mappings`.
            # The required data already exists in the recorders.
            distill_losses = self.distiller.compute_distill_losses()
            losses.update(add_prefix(distill_losses, 'distill'))

        return losses

    def train(self, mode: bool = True) -> None:
        """Set distiller's forward mode."""
        super().train(mode)
        if mode and self.teacher_norm_eval:
            for m in self.teacher.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.val_data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.val_data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.val_data_preprocessor(data, False)
        return self._run_forward(data, mode='predict')  # type: ignore
