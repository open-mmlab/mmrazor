# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.distillers import ConfigurableDistiller
from mmrazor.models.mutators import NasMutator
from mmrazor.registry import MODELS
from mmrazor.utils import ValidFixMutable
from ..base import BaseAlgorithm, LossResults

VALID_MUTATOR_TYPE = Union[NasMutator, Dict]
VALID_DISTILLER_TYPE = Union[ConfigurableDistiller, Dict]


@MODELS.register_module()
class NSGANetV2(BaseAlgorithm):
    """Implementation of `NSGANetV2 <https://arxiv.org/abs/2007.10396>`_

    NSGANetV2 generates task-specific models that are competitive under
    multiple competing objectives.

    NSGANetV2 comprises of two surrogates, one at the architecture level to
    improve sample efficiency and one at the weights level, through a supernet,
    to improve gradient descent training efficiency.

    The logic of the search part is implemented in
    :class:`mmrazor.engine.NSGA2SearchLoop`

    Args:
        architecture (dict|:obj:`BaseModel`): The config of :class:`BaseModel`
            or built model. Corresponding to supernet in NAS algorithm.
        mutator (VALID_MUTATOR_TYPE): The config of :class:`NasMutator` or
            built mutator.
        fix_subnet (str | dict | :obj:`FixSubnet`): The path of yaml file or
            loaded dict or built :obj:`FixSubnet`. Defaults to None.
        data_preprocessor (Optional[Union[dict, nn.Module]]): The pre-process
            config of :class:`BaseDataPreprocessor`. Defaults to None.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.2.
        backbone_dropout_stages (List): Stages to be set dropout. Defaults to
            [6, 7].
        norm_training (bool): Whether to set norm layers to training mode,
            namely, not freeze running stats (mean and var). Note: Effect on
            Batch Norm and its variants only. Defaults to False.
        init_cfg (Optional[dict]): Init config for ``BaseModule``.
            Defaults to None.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: VALID_MUTATOR_TYPE,
                 fix_subnet: Optional[ValidFixMutable] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 drop_path_rate: float = 0.2,
                 backbone_dropout_stages: List = [6, 7],
                 norm_training: bool = False,
                 init_cfg: Optional[dict] = None):
        super().__init__(architecture, data_preprocessor, init_cfg)

        if fix_subnet:
            # Avoid circular import
            from mmrazor.structures import load_fix_subnet

            # According to fix_subnet, delete the unchosen part of supernet
            load_fix_subnet(self, fix_subnet)
            self.is_supernet = False
        else:
            self.mutator = self._build_mutator(mutator)
            self.mutator.prepare_from_supernet(self.architecture)

            self.is_supernet = True

        self.drop_path_rate = drop_path_rate
        self.backbone_dropout_stages = backbone_dropout_stages
        self.norm_training = norm_training

    def _build_mutator(self, mutator: VALID_MUTATOR_TYPE = None) -> NasMutator:
        """Build mutator."""
        if isinstance(mutator, dict):
            mutator = MODELS.build(mutator)
        if not isinstance(mutator, NasMutator):
            raise TypeError('mutator should be a `dict` or `NasMutator` '
                            f'instance, but got {type(mutator)}.')
        return mutator

    def loss(
        self,
        batch_inputs: torch.Tensor,
        data_samples: Optional[List[BaseDataElement]] = None,
    ) -> LossResults:
        """Calculate losses from a batch of inputs and data samples."""
        if self.is_supernet:
            self.mutator.set_choices(self.mutator.sample_choices())
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
