# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from mmrazor.registry import MODELS
from .configurable_distiller import ConfigurableDistiller


@MODELS.register_module()
class BYOTDistiller(ConfigurableDistiller):
    """``BYOTDistiller`` inherits ``ConfigurableDistiller`` and only modifies
    ``get_record()`` function to ``get_record_with_cidx()``.

    In ``BYOTDistiller``, ``self.teacher_recorder`` records self-teacher data
    which requires detach().
    """

    def get_record(self,
                   recorder: str,
                   from_student: bool,
                   record_idx: int = 0,
                   data_idx: Optional[int] = None,
                   connector: Optional[str] = None,
                   connector_idx: Optional[int] = None) -> List:
        """According to each item in ``record_infos``, get the corresponding
        record in ``recorder_manager``.

        Detach teacher_record.
        """

        if from_student:
            recorder_ = self.student_recorders.get_recorder(recorder)
        else:
            recorder_ = self.teacher_recorders.get_recorder(recorder)
        record_data = recorder_.get_record_data(record_idx, data_idx)

        if connector:
            record_data = self.connectors[connector](record_data)
        if connector_idx is not None:
            record_data = record_data[connector_idx]
        # Detach self-teacher output Tensor from model, assert hook tensor.
        if not from_student:
            record_data = record_data.detach()

        return record_data
