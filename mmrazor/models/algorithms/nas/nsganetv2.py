# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.mutators.base_mutator import BaseMutator
from mmrazor.registry import MODELS
from mmrazor.utils import ValidFixMutable
from ..base import BaseAlgorithm, LossResults

VALID_MUTATOR_TYPE = Union[BaseMutator, Dict]
VALID_MUTATORS_TYPE = Dict[str, Union[BaseMutator, Dict]]
VALID_DISTILLER_TYPE = Union[ConfigurableDistiller, Dict]


@MODELS.register_module()
class NSGANetV2(BaseAlgorithm):
    """NSGANetV2 algorithm."""

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutators: VALID_MUTATORS_TYPE,
                 fix_subnet: Optional[ValidFixMutable] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 drop_path_rate: float = 0.2,
                 backbone_dropout_stages: List = [6, 7],
                 norm_training: bool = False,
                 init_cfg: Optional[dict] = None):
        super().__init__(architecture, data_preprocessor, init_cfg)

        if isinstance(mutators, dict):
            built_mutators: Dict = dict()
            for name, mutator_cfg in mutators.items():
                if 'parse_cfg' in mutator_cfg and isinstance(
                        mutator_cfg['parse_cfg'], dict):
                    assert mutator_cfg['parse_cfg'][
                        'type'] == 'Predefined', \
                            'BigNAS only support predefined.'
                mutator: BaseMutator = MODELS.build(mutator_cfg)
                built_mutators[name] = mutator
                mutator.prepare_from_supernet(self.architecture)
            self.mutators = built_mutators
        else:
            raise TypeError('mutator should be a `dict` but got '
                            f'{type(mutators)}')

        self.drop_path_rate = drop_path_rate
        self.backbone_dropout_stages = backbone_dropout_stages
        self.norm_training = norm_training
        self.is_supernet = True

        if fix_subnet:
            # Avoid circular import
            from mmrazor.structures import load_fix_subnet

            # According to fix_subnet, delete the unchosen part of supernet
            load_fix_subnet(self, fix_subnet)
            self.is_supernet = False

    def sample_subnet(self, kind='random') -> Dict:
        """Random sample subnet by mutator."""
        subnet = dict()
        for mutator in self.mutators.values():
            subnet.update(mutator.sample_choices(kind))
        return subnet

    def set_subnet(self, subnet: Dict[str, Dict[int, Union[int,
                                                           list]]]) -> None:
        """Set the subnet sampled by :meth:sample_subnet."""
        for mutator in self.mutators.values():
            mutator.set_choices(subnet)

    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""
        if self.is_supernet:
            self.set_subnet(self.sample_subnet())
            return self.architecture(batch_inputs, data_samples, mode='loss')
        else:
            return self.architecture(batch_inputs, data_samples, mode='loss')

    def train(self, mode=True):
        """Convert the model into eval mode while keep normalization layer
        unfreezed."""

        super().train(mode)
        if self.norm_training and not mode:
            for module in self.architecture.modules():
                if isinstance(module, _BatchNorm):
                    module.training = True
