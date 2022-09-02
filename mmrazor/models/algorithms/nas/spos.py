# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel
from mmengine.structures import BaseDataElement
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutators import OneShotModuleMutator
from mmrazor.registry import MODELS
from mmrazor.utils import SingleMutatorRandomSubnet, ValidFixMutable
from ..base import BaseAlgorithm, LossResults


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
        mutator (dict|:obj:`OneShotModuleMutator`): The config of
            :class:`OneShotModuleMutator` or built mutator.
        fix_subnet (str | dict | :obj:`FixSubnet`): The path of yaml file or
            loaded dict or built :obj:`FixSubnet`.
        norm_training (bool): Whether to set norm layers to training mode,
            namely, not freeze running stats (mean and var). Note: Effect on
            Batch Norm and its variants only. Defaults to False.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Defaults to None.
        init_cfg (dict): Init config for ``BaseModule``.

    Note:
        SPOS has two training mode: supernet training and subnet retraining.
        If `fix_subnet` is None, it means supernet training.
        If `fix_subnet` is not None, it means subnet training.

    Note:
        During supernet training, since each op is not fully trained, the
        statistics of :obj:_BatchNorm are inaccurate. This problem affects the
        evaluation of the performance of each subnet in the search phase. There
        are usually two ways to solve this problem, both need to set
        `norm_training` to True:

        1) Using a large batch size, BNs use the mean and variance of the
           current batch during forward.
        2) Recalibrate the statistics of BN before searching.

    Note:
        SPOS only uses one mutator. If you want to inherit SPOS to develop
        more complex algorithms, it is also feasible to use multiple mutators.
        For example, one part of the supernet uses SPOS(OneShotModuleMutator)
        to search, and the other part uses Darts(DiffModuleMutator) to search.
    """

    # TODO fix ea's name in doc-string.

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: Optional[Union[OneShotModuleMutator, Dict]] = None,
                 fix_subnet: Optional[ValidFixMutable] = None,
                 norm_training: bool = False,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(architecture, data_preprocessor, init_cfg)

        # SPOS has two training mode: supernet training and subnet retraining.
        # fix_subnet is not None, means subnet retraining.
        if fix_subnet:
            # Avoid circular import
            from mmrazor.structures import load_fix_subnet

            # According to fix_subnet, delete the unchosen part of supernet
            load_fix_subnet(self.architecture, fix_subnet)
            self.is_supernet = False
        else:
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

            # Mutator is an essential component of the NAS algorithm. It
            # provides some APIs commonly used by NAS.
            # Before using it, you must do some preparations according to
            # the supernet.
            self.mutator.prepare_from_supernet(self.architecture)
            self.is_supernet = True

        self.norm_training = norm_training

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
