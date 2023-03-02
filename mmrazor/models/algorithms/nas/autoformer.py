# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from torch import nn

from mmrazor.models.mutators import NasMutator
from mmrazor.registry import MODELS
from ..base import BaseAlgorithm, LossResults

VALID_MUTATOR_TYPE = Union[NasMutator, Dict]


@MODELS.register_module()
class Autoformer(BaseAlgorithm):
    """Implementation of `Autoformer <https://arxiv.org/abs/2107.00651>`_

    AutoFormer is dedicated to vision transformer search. AutoFormer
    entangles the weights of different blocks in the same layers during
    supernet training.
    The logic of the search part is implemented in
    :class:`mmrazor.engine.EvolutionSearchLoop`

    Args:
        architecture (dict|:obj:`BaseModel`): The config of :class:`BaseModel`
            or built model. Corresponding to supernet in NAS algorithm.
        mutator (VALID_MUTATOR_TYPE): The config of :class:`NasMutator` or
            built mutator.
        data_preprocessor (Optional[Union[dict, nn.Module]]): The pre-process
            config of :class:`BaseDataPreprocessor`. Defaults to None.
        init_cfg (Optional[dict]): Init config for ``BaseModule``.
            Defaults to None.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: VALID_MUTATOR_TYPE = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(architecture, data_preprocessor, init_cfg)

        self.mutator = self._build_mutator(mutator)
        self.mutator.prepare_from_supernet(self.architecture)

    def _build_mutator(self, mutator: VALID_MUTATOR_TYPE = None) -> NasMutator:
        """build mutator."""
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
