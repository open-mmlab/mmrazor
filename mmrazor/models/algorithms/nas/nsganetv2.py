# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.mutators.base_mutator import BaseMutator
from mmrazor.models.mutators import OneShotModuleMutator
from mmrazor.registry import MODELS
from mmrazor.structures.subnet.fix_subnet import load_fix_subnet
from mmrazor.utils import SingleMutatorRandomSubnet, ValidFixMutable
from ..base import BaseAlgorithm, LossResults

VALID_MUTATOR_TYPE = Union[BaseMutator, Dict]
VALID_MUTATORS_TYPE = Dict[str, Union[BaseMutator, Dict]]
VALID_DISTILLER_TYPE = Union[ConfigurableDistiller, Dict]


@MODELS.register_module()
class NSGANetV2(BaseAlgorithm):
    """

    """

    # TODO fix ea's name in doc-string.

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: VALID_MUTATORS_TYPE,
                #  distiller: VALID_DISTILLER_TYPE,
                #  norm_training: bool = False,
                 fix_subnet: Optional[ValidFixMutable] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None,
                 drop_prob: float = 0.2):
        super().__init__(architecture, data_preprocessor, init_cfg)

        if fix_subnet:
            # Avoid circular import
            from mmrazor.structures import load_fix_subnet

            # According to fix_subnet, delete the unchosen part of supernet
            load_fix_subnet(self.architecture, fix_subnet)
            self.is_supernet = False
        else:
            # Mutator is an essential component of the NAS algorithm. It
            # provides some APIs commonly used by NAS.
            # Before using it, you must do some preparations according to
            # the supernet.
            self.mutator.prepare_from_supernet(self.architecture)
            self.is_supernet = True

        self.drop_prob = drop_prob

    def _build_mutator(self, mutator: VALID_MUTATOR_TYPE) -> BaseMutator:
        """build mutator."""
        assert mutator is not None, \
            'mutator cannot be None when fix_subnet is None.'
        if isinstance(mutator, OneShotModuleMutator):
            self.mutator = mutator
        elif isinstance(mutator, dict):
            self.mutator = MODELS.build(mutator)
        else:
            raise TypeError('mutator should be a `dict` or '
                            f'`OneShotModuleMutator` instance, but got '
                            f'{type(mutator)}')
        return mutator

    def sample_subnet(self) -> SingleMutatorRandomSubnet:
        """Random sample subnet by mutator."""
        return self.mutator.sample_choices()

    def set_subnet(self, subnet: SingleMutatorRandomSubnet):
        """Set the subnet sampled by :meth:sample_subnet."""
        self.mutator.set_choices(subnet)

    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""
        if self.is_supernet:
            random_subnet = self.sample_subnet()
            self.set_subnet(random_subnet)
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
