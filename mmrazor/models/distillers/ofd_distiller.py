# Copyright (c) OpenMMLab. All rights reserved.
import math
from operator import attrgetter

import torch
import torch.nn as nn
from scipy.stats import norm

from mmrazor.registry import MODELS
from ..architectures.connectors import OFDTeacherConnector
from ..losses import OFDLoss
from .configurable_distiller import ConfigurableDistiller


@MODELS.register_module()
class OFDDistiller(ConfigurableDistiller):
    """Distiller for ``OverhaulFeatureDistillation``, inherited from
    ``ConfigurableDistiller``, add func:

    ``init_ofd_connectors`` to initialize margin.
    """

    def init_ofd_connectors(self, teacher: nn.Module) -> None:
        """Initialize OFD connectors' `margin`."""
        for loss_key, loss_forward_mapping in self.loss_forward_mappings.items(
        ):
            if isinstance(self.distill_losses[loss_key], OFDLoss):
                for _input_keys, _input_mapping in loss_forward_mapping.items(
                ):
                    recorder_mgn = self.student_recorders if _input_mapping[
                        'from_student'] else self.teacher_recorders
                    recorder = recorder_mgn.get_recorder(
                        _input_mapping['recorder'])
                    module_key = recorder.source
                    bn_module = attrgetter(module_key)(teacher)

                    assert isinstance(
                        bn_module, (nn.BatchNorm2d, nn.SyncBatchNorm)), (
                            'Overhaul distillation only support connection on '
                            'layers: [`BatchNorm2d`, `SyncBatchNorm`]')

                    if 'connector' in _input_mapping and not _input_mapping[
                            'from_student']:
                        connector = self.connectors[
                            _input_mapping['connector']]
                        assert isinstance(connector, OFDTeacherConnector), (
                            'OFD loss mapping for `t_feature` expect type '
                            '`OFDTeacherConnector`, but get '
                            f'`{type(connector)}`')
                        margin = self._get_margin_from_BN(bn_module)
                        connector.init_margin(margin)

    def _get_margin_from_BN(self, bn: nn.BatchNorm2d) -> torch.Tensor:
        """Get margin from BN layer.

        Args:
            bn (nn.BatchNorm2d): input module, must be a BN layer.

        Returns:
            torch.Tensor: margin
        """
        margin = []
        std = bn.weight.data
        mean = bn.bias.data
        for (s, m) in zip(std, mean):
            s = abs(s.item())
            m = m.item()
            if norm.cdf(-m / s) > 0.001:
                margin.append(-s * math.exp(-(m / s)**2 / 2) /
                              math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
            else:
                margin.append(-3 * s)
        return torch.FloatTensor(margin).unsqueeze(1).unsqueeze(2).unsqueeze(
            0).detach()
