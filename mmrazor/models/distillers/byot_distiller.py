# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from mmrazor.registry import MODELS
from ..algorithms.base import LossResults
from .configurable_distiller import ConfigurableDistiller


@MODELS.register_module()
class BYOTDistiller(ConfigurableDistiller):
    """``BYOTDistiller`` inherits ``ConfigurableDistiller`` and only modifies
    ``get_record()`` function to ``get_record_with_cidx()``.

    In ``BYOTDistiller``, ``self.teacher_recorder`` records self-teacher data
    which requires detach().
    """

    def compute_distill_losses(self) -> LossResults:
        """Compute distill losses automatically."""
        # Record all computed losses' results.
        losses = dict()
        for loss_name, forward_mappings in self.loss_forward_mappings.items():
            forward_kwargs = dict()
            for forward_key, record in forward_mappings.items():
                forward_var = self.get_record_with_cidx(**record)
                forward_kwargs[forward_key] = forward_var

            loss_module = self.distill_losses[loss_name]
            loss = loss_module(**forward_kwargs)  # type: ignore
            # add computed loss result.
            losses[loss_name] = loss

        return losses

    def get_record_with_cidx(self,
                             recorder: str,
                             from_student: bool,
                             record_idx: int = 0,
                             data_idx: Optional[int] = None,
                             connector_idx: Optional[int] = None,
                             connector: Optional[str] = None) -> List:
        """According to each item in ``record_infos``, get the corresponding
        record in ``recorder_manager``.

        Introduce ``connector_idx`` as index of connnector's output.
        """

        if from_student:
            recorder_ = self.student_recorders.get_recorder(recorder)
        else:
            recorder_ = self.teacher_recorders.get_recorder(recorder)
        record_data = recorder_.get_record_data(record_idx, data_idx)

        if connector:
            record_data = self.connectors[connector](record_data)
        # Similar with record_idx and data_idx, connector_idx index from
        # connector output tuple.
        if connector_idx is not None:
            record_data = record_data[connector_idx]
        # Detach self-teacher output Tensor from model, assert hook tensor.
        if not from_student:
            record_data = record_data.detach()

        return record_data
