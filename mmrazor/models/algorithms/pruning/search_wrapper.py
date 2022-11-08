# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement

from mmrazor.models.mutators import ChannelMutator
from mmrazor.registry import MODELS
from ..base import BaseAlgorithm

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


@MODELS.register_module()
class SearchWrapper(BaseAlgorithm):

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator_cfg: Union[Dict, ChannelMutator] = dict(
                     type='ChannelMutator',
                     channel_unit_cfg=dict(
                         type='SequentialMutableChannelUnit')),
                 data_preprocessor: Optional[Union[Dict, nn.Module]] = None,
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(architecture, data_preprocessor, init_cfg)

        # mutator
        self.mutator: ChannelMutator = MODELS.build(mutator_cfg)
        self.mutator.prepare_from_supernet(self.architecture)

    def sample_subnet(self):
        return self.mutator.sample_choices()

    def set_subnet(self, chocies):
        self.mutator.set_choices(chocies)

    def curent_subnet(self):
        return self.mutator.current_choices
