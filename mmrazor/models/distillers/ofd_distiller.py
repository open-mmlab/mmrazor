# Copyright (c) OpenMMLab. All rights reserved.
import math
from operator import attrgetter
from typing import Dict, Optional

import torch
import torch.nn as nn
from scipy.stats import norm

from mmrazor.registry import MODELS
from ..architectures.connectors import OFDConnector
from ..losses import OFDLoss
from .configurable_distiller import ConfigurableDistiller


@MODELS.register_module()
class OFDDistiller(ConfigurableDistiller):

    def __init__(self,
                 student_recorders: Optional[Dict[str, Dict]] = None,
                 teacher_recorders: Optional[Dict[str, Dict]] = None,
                 distill_deliveries: Optional[Dict[str, Dict]] = None,
                 connectors: Optional[Dict[str, Dict]] = None,
                 distill_losses: Optional[Dict[str, Dict]] = None,
                 loss_forward_mappings: Optional[Dict[str, Dict]] = None,
                 **kwargs) -> None:
        super().__init__(student_recorders, teacher_recorders,
                         distill_deliveries, connectors, distill_losses,
                         loss_forward_mappings, **kwargs)

    def _init_ofd_connectors(self, teacher):
        """Initialize OFD connectors' `margin`."""
        for loss_key, loss_forward_mapping in self.loss_forward_mappings.items(
        ):
            if isinstance(self.distill_losses[loss_key], OFDLoss):
                for _input_keys, _input_mapping in loss_forward_mapping:
                    recorder_dict = self.student_recorders if _input_mapping[
                        'from_student'] else self.teacher_recorders
                    recorder = recorder_dict[_input_mapping['recorder']]
                    module_key = recorder.source
                    bn_module = attrgetter(module_key)(teacher)

                    assert isinstance(
                        bn_module, (nn.BatchNorm2d, nn.SyncBatchNorm)
                    ), ('Overhaul distillation only support connection on ',
                        'layers: [`BatchNorm2d`, `SyncBatchNorm`]')

                    if 'connector' in _input_mapping:
                        connector = self.connectors[
                            _input_mapping['connector']]
                        assert isinstance(connector, OFDConnector), (
                            'OFD loss mapping expect type `OFDConnector`, ',
                            f'but get `{type(connector)}`')
                        margin = self._get_margin_from_BN(bn_module)
                        connector.init_margin(margin)

    def _get_margin_from_BN(self, bn: nn.BatchNorm2d) -> torch.Tensor:
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
