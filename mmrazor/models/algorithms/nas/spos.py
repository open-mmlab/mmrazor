# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutators import NasMutator
from mmrazor.registry import MODELS
from ..base import BaseAlgorithm, LossResults

VALID_MUTATOR_TYPE = Union[NasMutator, Dict]


@MODELS.register_module()
class SPOS(BaseAlgorithm):
    """Implementation of `SPOS <https://arxiv.org/abs/1904.00420>`_

    SPOS means Single Path One-Shot, a classic NAS algorithm.
    :class:`SPOS` implements the APIs required by the Single Path One-Shot
    algorithm, as well as the supernet training and subnet retraining logic
    for each iter.

    The logic of the search part is implemented in
    :class:`mmrazor.core.EvolutionSearch`

    Args:
        architecture (dict|:obj:`BaseModel`): The config of :class:`BaseModel`
            or built model. Corresponding to supernet in NAS algorithm.
        mutator (VALID_MUTATOR_TYPE): The config of :class:`NasMutator` or
            built mutator.
        norm_training (bool): Whether to set norm layers to training mode,
            namely, not freeze running stats (mean and var). Note: Effect on
            Batch Norm and its variants only. Defaults to False.
        data_preprocessor (Optional[Union[dict, nn.Module]]): The pre-process
            config of :class:`BaseDataPreprocessor`. Defaults to None.
        init_cfg (Optional[dict]): Init config for ``BaseModule``.
            Defaults to None.

    Note:
        During supernet training, since each op is not fully trained, the
        statistics of :obj:_BatchNorm are inaccurate. This problem affects the
        evaluation of the performance of each subnet in the search phase. There
        are usually two ways to solve this problem, both need to set
        `norm_training` to True:

        1) Using a large batch size, BNs use the mean and variance of the
           current batch during forward.
        2) Recalibrate the statistics of BN before searching.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: VALID_MUTATOR_TYPE = None,
                 norm_training: bool = False,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(architecture, data_preprocessor, init_cfg)

        self.mutator = self._build_mutator(mutator)
        # Mutator is an essential component of the NAS algorithm. It
        # provides some APIs commonly used by NAS.
        # Before using it, you must do some preparations according to
        # the supernet.
        self.mutator.prepare_from_supernet(self.architecture)

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
        self.mutator.set_choices(self.mutator.sample_choices())
        return self.architecture(batch_inputs, data_samples, mode='loss')

    def train(self, mode=True):
        """Convert the model into eval mode while keep normalization layer
        unfreezed."""

        super().train(mode)
        if self.norm_training and not mode:
            for module in self.architecture.modules():
                if isinstance(module, _BatchNorm):
                    module.training = True
