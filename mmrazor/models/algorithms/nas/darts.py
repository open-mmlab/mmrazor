# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutators import DiffModuleMutator
from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from mmrazor.utils import FixMutable
from ..base import BaseAlgorithm


@MODELS.register_module()
class Darts(BaseAlgorithm):
    """Implementation of `DARTS <https://arxiv.org/abs/1806.09055>`_

    DARTS means Differentiable Architecture Search, a classic NAS algorithm.
    :class:`Darts` implements the APIs required by the DARTS, as well as the
    supernet training and subnet retraining logic for each iter.

    Args:
        architecture (dict|:obj:`BaseModel`): The config of :class:`BaseModel`
            or built model. Corresponding to supernet in NAS algorithm.
        mutator (dict|:obj:`DiffModuleMutator`): The config of
            :class:`DiffModuleMutator` or built mutator.
        fix_subnet (str | dict | :obj:`FixSubnet`): The path of yaml file or
            loaded dict or built :obj:`FixSubnet`.
        norm_training (bool): Whether to set norm layers to training mode,
            namely, not freeze running stats (mean and var). Note: Effect on
            Batch Norm and its variants only. Defaults to False.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Defaults to None.
        init_cfg (dict): Init config for ``BaseModule``.

    Note:
        Darts has two training mode: supernet training and subnet retraining.
        If `fix_subnet` is None, it means supernet training.
        If `fix_subnet` is not None, it means subnet training.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: Optional[Union[DiffModuleMutator, Dict]] = None,
                 fix_subnet: Optional[FixMutable] = None,
                 unroll: bool = False,
                 norm_training: bool = False,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(architecture, data_preprocessor, init_cfg)

        # Darts has two training mode: supernet training and subnet retraining.
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
            if isinstance(mutator, DiffModuleMutator):
                self.mutator = mutator
            elif isinstance(mutator, dict):
                self.mutator = MODELS.build(mutator)
            else:
                raise TypeError('mutator should be a `dict` or '
                                f'`DiffModuleMutator` instance, but got '
                                f'{type(mutator)}')

            # Mutator is an essential component of the NAS algorithm. It
            # provides some APIs commonly used by NAS.
            # Before using it, you must do some preparation according to
            # the supernet.
            self.mutator.prepare_from_supernet(self.architecture)
            self.is_supernet = True

        self.norm_training = norm_training
        # TODO support unroll
        self.unroll = unroll

    def search_subnet(self):
        """Search subnet by mutator."""

        # Avoid circular import
        from mmrazor.structures import export_fix_subnet

        subnet = self.mutator.sample_choices()
        self.mutator.set_choices(subnet)
        return export_fix_subnet(self)

    def train(self, mode=True):
        """Convert the model into eval mode while keep normalization layer
        unfreezed."""

        super().train(mode)
        if self.norm_training and not mode:
            for module in self.architecture.modules():
                if isinstance(module, _BatchNorm):
                    module.training = True

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        if isinstance(data, (tuple, list)) and isinstance(
                optim_wrapper, OptimWrapperDict):
            assert len(data) == len(optim_wrapper), \
                f'The length of data ({len(data)}) should be equal to that '\
                f'of optimizers ({len(optim_wrapper)}).'
            # TODO check the order of data
            supernet_data, mutator_data = data

            log_vars = dict()
            # TODO support unroll
            with optim_wrapper['mutator'].optim_context(self):
                mutator_batch_inputs, mutator_data_samples = \
                    self.data_preprocessor(mutator_data, True)
                mutator_loss = self(
                    mutator_batch_inputs, mutator_data_samples, mode='loss')
            mutator_losses, mutator_log_vars = self.parse_losses(mutator_loss)
            optim_wrapper['mutator'].update_params(mutator_losses)
            log_vars.update(add_prefix(mutator_log_vars, 'mutator'))

            with optim_wrapper['architecture'].optim_context(self):
                supernet_batch_inputs, supernet_data_samples = \
                    self.data_preprocessor(supernet_data, True)
                supernet_loss = self(
                    supernet_batch_inputs, supernet_data_samples, mode='loss')
            supernet_losses, supernet_log_vars = self.parse_losses(
                supernet_loss)
            optim_wrapper['architecture'].update_params(supernet_losses)
            log_vars.update(add_prefix(supernet_log_vars, 'supernet'))

        else:
            # Enable automatic mixed precision training context.
            with optim_wrapper.optim_context(self):
                batch_inputs, data_samples = self.data_preprocessor(data, True)
                losses = self(batch_inputs, data_samples, mode='loss')
            parsed_losses, log_vars = self.parse_losses(losses)
            optim_wrapper.update_params(parsed_losses)

        return log_vars


@MODEL_WRAPPERS.register_module()
class DartsDDP(MMDistributedDataParallel):

    def __init__(self,
                 *,
                 device_ids: Optional[Union[List, int, torch.device]] = None,
                 **kwargs) -> None:
        if device_ids is None:
            if os.environ.get('LOCAL_RANK') is not None:
                device_ids = [int(os.environ['LOCAL_RANK'])]
        super().__init__(device_ids=device_ids, **kwargs)

    def train_step(self, data: List[dict],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating are also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.
        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        if isinstance(data, (tuple, list)) and isinstance(
                optim_wrapper, OptimWrapperDict):
            assert len(data) == len(optim_wrapper), \
                f'The length of data ({len(data)}) should be equal to that '\
                f'of optimizers ({len(optim_wrapper)}).'
            # TODO check the order of data
            supernet_data, mutator_data = data

            log_vars = dict()
            # TODO process the input

            with optim_wrapper['mutator'].optim_context(self):
                mutator_batch_inputs, mutator_data_samples = \
                    self.module.data_preprocessor(mutator_data, True)
                mutator_loss = self(
                    mutator_batch_inputs, mutator_data_samples, mode='loss')
            mutator_losses, mutator_log_vars = self.module.parse_losses(
                mutator_loss)
            optim_wrapper['mutator'].update_params(mutator_losses)
            log_vars.update(add_prefix(mutator_log_vars, 'mutator'))

            with optim_wrapper['architecture'].optim_context(self):
                supernet_batch_inputs, supernet_data_samples = \
                    self.module.data_preprocessor(supernet_data, True)
                supernet_loss = self(
                    supernet_batch_inputs, supernet_data_samples, mode='loss')
            supernet_losses, supernet_log_vars = self.module.parse_losses(
                supernet_loss)
            optim_wrapper['architecture'].update_params(supernet_losses)
            log_vars.update(add_prefix(supernet_log_vars, 'supernet'))

        else:
            # Enable automatic mixed precision training context.
            with optim_wrapper.optim_context(self):
                batch_inputs, data_samples = self.module.data_preprocessor(
                    data, True)
                losses = self(batch_inputs, data_samples, mode='loss')
            parsed_losses, log_vars = self.module.parse_losses(losses)
            optim_wrapper.update_params(parsed_losses)

        return log_vars
