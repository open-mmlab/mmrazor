# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import mmengine.dist as dist
import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from torch import nn

from mmrazor.models.mutators.base_mutator import BaseMutator
from mmrazor.registry import MODELS
from mmrazor.utils import ValidFixMutable
from ..base import BaseAlgorithm, LossResults


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
        mutators (Optional[dict]): The dict of different Mutators config.
            Defaults to None.
        fix_subnet (str | dict | :obj:`FixSubnet`): The path of yaml file or
            loaded dict or built :obj:`FixSubnet`. Defaults to None.
        data_preprocessor (Optional[Union[dict, nn.Module]]): The pre-process
            config of :class:`BaseDataPreprocessor`. Defaults to None.
        init_cfg (Optional[dict]): Init config for ``BaseModule``.
            Defaults to None.
    Note:
        Autoformer uses two mutators which are ``DynamicValueMutator`` and
        ``ChannelMutator``. DynamicValueMutator handle the mutable object
        ``OneShotMutableValue`` in Autoformer while ChannelMutator handle
        the mutable object ``OneShotMutableChannel`` in Autoformer.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutators: Optional[Union[BaseMutator, Dict]] = None,
                 fix_subnet: Optional[ValidFixMutable] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(architecture, data_preprocessor, init_cfg)

        # Autoformer support supernet training and subnet retraining.
        # fix_subnet is not None, means subnet retraining.
        if fix_subnet:
            # Avoid circular import
            from mmrazor.structures import load_fix_subnet

            # According to fix_subnet, delete the unchosen part of supernet
            load_fix_subnet(self.architecture, fix_subnet)
            self.is_supernet = False
        else:
            assert mutators is not None, \
                'mutator cannot be None when fix_subnet is None.'
            if isinstance(mutators, BaseMutator):
                self.mutator = mutators
            elif isinstance(mutators, dict):
                built_mutators: Dict = dict()
                for name, mutator_cfg in mutators.items():
                    if name == 'channel_mutator':
                        mutator = MODELS.build(mutator_cfg)
                        built_mutators[name] = mutator
                        mutator.prepare_from_supernet(self.architecture) # ?
                self.mutators = built_mutators
            else:
                raise TypeError('mutator should be a `dict` or belong to '
                                f'`BaseMutator` instance, but got '
                                f'{type(mutator)}')

            self.is_supernet = True

    def sample_subnet(self) -> Dict:
        """Random sample subnet by mutator."""
        subnet_dict = dict()
        for name, mutator in self.mutators.items():
            # 只channel_mutator进行采样
            if name == 'channel_mutator':
                subnet_dict[name] = mutator.sample_choices()
        dist.broadcast_object_list([subnet_dict])
        return subnet_dict

    def set_subnet(self, subnet_dict: Dict) -> None:
        """Set the subnet sampled by :meth:sample_subnet."""
        for name, mutator in self.mutators.items():
            if name == 'channel_mutator':
                mutator.set_choices(subnet_dict[name])

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
