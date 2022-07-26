# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from inspect import signature
from typing import Dict, List, Optional

from mmengine.model import BaseModel
from torch import nn

from mmrazor.registry import MODELS
from mmrazor.structures import DistillDeliveryManager, RecorderManager
from ..algorithms.base import LossResults
from .base_distiller import BaseDistiller


@MODELS.register_module()
class ConfigurableDistiller(BaseDistiller):
    """``ConfigurableDistiller`` is a powerful tool that can reproduce most
    distillation algorithms without modifying the code of teacher or student
    models.

    ``ConfigurableDistiller`` can get various intermediate results of the
    model in a hacky way by ``Recorder``. More details see user-docs for
    ``Recorder``.

    ``ConfigurableDistiller`` can use the teacher's intermediate results to
    override the student's intermediate results in a hacky way by ``Delivery``.
    More details see user-docs for ``Delivery``.

    Args:
        student_recorders (dict, optional): Config for multiple recorders. A
            student model may have more than one recorder. These recorders
            only record the student model's intermediate results. Defaults to
            None.
        teacher_recorders (dict, optional): Config for multiple recorders. A
            teacher model may have more than one recorder. These recorders
            only record the teacher model's intermediate results. Defaults to
            None.
        distill_deliveries (dict, optional): Config for multiple deliveries. A
            distill algorithm may have more than one delivery. Defaults to
            None.
        student_connectors (dict, optional): Config for multiple connectors. A
            student model may have more than one connector. Each connector only
            corresponds to one student model's intermediate result and one
            distill loss. Defaults to None.
        teacher_connectors (dict, optional): Config for multiple connectors. A
            teacher model may have more than one connector. Each connector only
            corresponds to one teacher model's intermediate result and one
            distill loss. Defaults to None.
        distill_losses: (Dict[str, Dict], optional): Config for multiple
            distill losses. A distill algorithm may have more than one distill
            loss. Defaults to None.
        loss_forward_mappings: (Dict[str, Dict], optional): Mapping between
            distill loss forward arguments and records.

    Note:
        If a distill loss needs to backward, the name of the loss must contain
        "loss". If it is only used as a statistical value, the name can not
        contain "loss". More details see docs for
        :func:`mmengine.model.BaseModel._parse_loss`.

    Note:
        The keys of ``loss_forward_mappings`` should be consistent with the
        keys of ``distill_losses``.

        Each item in ``loss_forward_mappings`` is a mapping between a distill
        loss and its forward arguments. The keys of the mapping are the
        signature of the loss's forward, and the values of the mapping are the
        recorded data location.

        ``from_recorder``refers to the recorder where the data is stored, and
        if ``from_student`` is True, it means the recorder is in `
        `student_recorders``; otherwise, it means the recorder is in
        ``teacher_recorders``.

    Examples:
        >>> distill_losses = dict(
        ...     loss_neck=dict(type='L2Loss', loss_weight=5))

        >>> student_recorders = dict(
        ...     feat = dict(type='ModuleOutputs', sources=['neck.gap']))

        >>> teacher_recorders = dict(
        ...     feat = dict(type='ModuleOutputs', sources=['neck.gap']))

        >>> student_connectors = dict(
        ...     loss_neck = dict(type='ReLUConnector', sources=['neck.gap']))

        >>> teacher_connectors = dict(
        ...     loss_neck = dict(type='ReLUConnector', sources=['neck.gap']))

        >>> loss_forward_mappings = dict(
        ...     loss_neck=dict(
        ...         s_feature=dict(from_recorder='feat', from_student=True),
        ...         t_feature=dict(from_recorder='feat', from_student=False)))
    """

    def __init__(self,
                 student_recorders: Optional[Dict[str, Dict]] = None,
                 teacher_recorders: Optional[Dict[str, Dict]] = None,
                 distill_deliveries: Optional[Dict[str, Dict]] = None,
                 student_connectors: Optional[Dict[str, Dict]] = None,
                 teacher_connectors: Optional[Dict[str, Dict]] = None,
                 distill_losses: Optional[Dict[str, Dict]] = None,
                 loss_forward_mappings: Optional[Dict[str, Dict]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        # The recorder manager is just constructed, but not really initialized
        # yet. Recorder manager initialization needs to input the corresponding
        # model.
        self.student_recorders = RecorderManager(student_recorders)
        self.teacher_recorders = RecorderManager(teacher_recorders)

        self.deliveries = DistillDeliveryManager(distill_deliveries)

        self.distill_losses = self.build_distill_losses(distill_losses)

        if loss_forward_mappings:
            # Check if loss_forward_mappings is in the correct format.
            self._check_loss_forward_mappings(self.distill_losses,
                                              loss_forward_mappings,
                                              self.student_recorders,
                                              self.teacher_recorders)
            self.loss_forward_mappings = loss_forward_mappings
        else:
            self.loss_forward_mappings = dict()

        self.stu_connector = self.build_connectors(student_connectors)
        self.tea_connector = self.build_connectors(teacher_connectors)

    def set_deliveries_override(self, override: bool) -> None:
        """Set the `override_data` of all deliveries."""
        self.deliveries.override_data = override

    def prepare_from_student(self, model: BaseModel) -> None:
        """Initialize student recorders."""
        self.student_recorders.initialize(model)

    def prepare_from_teacher(self, model: nn.Module) -> None:
        """Initialize teacher recorders."""
        self.teacher_recorders.initialize(model)

    def build_connectors(
        self,
        connectors: Optional[Dict[str, Dict]] = None,
    ) -> nn.ModuleDict:
        """Initialize connectors."""

        distill_connecotrs = nn.ModuleDict()
        if connectors:
            for loss_name, connector_cfg in connectors.items():
                assert loss_name in self.distill_losses
                assert loss_name not in distill_connecotrs
                connector = MODELS.build(connector_cfg)
                distill_connecotrs[loss_name] = connector

        return distill_connecotrs

    def build_distill_losses(
        self,
        losses: Optional[Dict[str, Dict]] = None,
    ) -> nn.ModuleDict:
        """build distill losses according config."""

        distill_losses = nn.ModuleDict()
        if losses:
            for loss_name, loss_cfg in losses.items():
                assert loss_name not in distill_losses
                if 'loss' not in loss_name:
                    warnings.warn(
                        f'Warning: If {loss_name} is a loss that needs to '
                        f'backward, the name of {loss_name} must contain '
                        f'"loss". If it is only used as a statistical value, '
                        'then the name must not contain "loss". More details '
                        'see docs for '
                        ':func:`mmengine.model.BaseModel._parse_loss`',
                        UserWarning)
                item_loss = MODELS.build(loss_cfg)
                distill_losses[loss_name] = item_loss

        return distill_losses

    def get_record(self,
                   recorder: str,
                   from_student: bool,
                   record_idx: int = 0,
                   data_idx: Optional[int] = None) -> List:
        """According to each item in ``record_infos``, get the corresponding
        record in ``recorder_manager``."""

        if from_student:
            recorder_ = self.student_recorders.get_recorder(recorder)
        else:
            recorder_ = self.teacher_recorders.get_recorder(recorder)

        return recorder_.get_record_data(record_idx, data_idx)

    def compute_distill_losses(self) -> LossResults:
        """Compute distill losses automatically."""
        # Record all computed losses' results.
        losses = dict()
        for loss_name, forward_mappings in self.loss_forward_mappings.items():
            forward_kwargs = dict()
            for forward_key, record in forward_mappings.items():
                forward_var = self.get_record(**record)
                if loss_name in self.stu_connector and record.from_student:
                    forward_var = self.stu_connector[loss_name](forward_var)
                if loss_name in self.tea_connector and not record.from_student:
                    forward_var = self.tea_connector[loss_name](forward_var)
                forward_kwargs[forward_key] = forward_var

            loss_module = self.distill_losses[loss_name]
            loss = loss_module(**forward_kwargs)  # type: ignore
            # add computed loss result.
            losses[loss_name] = loss

        return losses

    def _check_loss_forward_mappings(
            self, losses: nn.ModuleDict, loss_forward_mappings: Dict[str,
                                                                     Dict],
            student_recorders: RecorderManager,
            teacher_recorders: RecorderManager) -> None:
        """Check if ``loss_forward_mappings`` is in the correct format."""

        if not isinstance(loss_forward_mappings, dict):
            raise TypeError(
                'loss_forward_mappings should be a dict instance, but got'
                f'{type(loss_forward_mappings)}')

        for loss_name, forward_mappings in loss_forward_mappings.items():
            assert loss_name in losses, \
                f'"{loss_name}" is not in distill losses. The keys of ' \
                'loss_forward_kwargs must match the keys of distill_losses.'

            if not isinstance(forward_mappings, dict):
                raise TypeError(
                    'Each item of loss_forward_mappings should be a dict '
                    f'instance, but got {type(forward_mappings)}')

            loss_module = losses[loss_name]
            loss_forward_keys = signature(
                loss_module.forward).parameters.keys()
            assert len(loss_forward_keys) == len(forward_mappings.keys())

            for forward_key, record_info in forward_mappings.items():
                assert forward_key in loss_forward_keys, \
                    f'{forward_key} is not in the signature of \
                    {type(loss_module).__name__} forward, \
                    please check your config.'

                assert 'recorder' in record_info, \
                    'Each item of loss_forward_mappings should have ' \
                    '"recorder", pls check your config.'

                assert 'from_student' in record_info, \
                    'Each item of loss_forward_mappings should have ' \
                    '"from_student", pls check your config.'

                recorder: str = record_info['recorder']
                from_student: bool = record_info['from_student']

                if not isinstance(from_student, bool):
                    raise TypeError(f'from_student should be a bool instance, '
                                    f'but got {type(from_student)}')

                if from_student:
                    assert recorder in student_recorders.recorders, \
                        f'For {forward_key}, "{recorder}" must be in \
                        `student_recorders`.'

                else:
                    assert recorder in teacher_recorders.recorders, \
                        f'For {forward_key}, "{recorder}" must be in \
                        `teacher_recorders`.'
