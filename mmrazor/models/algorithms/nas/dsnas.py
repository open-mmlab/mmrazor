# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.logging import MessageHub
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.optim import OptimWrapper, OptimWrapperDict
from mmrazor.models.mutators import DiffModuleMutator
from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from mmrazor.utils import FixMutable
from ..base import BaseAlgorithm


@MODELS.register_module()
class Dsnas(BaseAlgorithm):
    """Implementation of `DSNAS <https://arxiv.org/abs/2002.09128>`_

    The DSNAS algorithm contains 3 stages without subnet retraining:
    1. [Optional] Supernet pretraining stage (cur_epoch < pretrain_epochs):
        freeze the arch optimizer;
    2. Searching stage: when cur_epoch in [pretrain_epochs, finetune_epochs];
    3. Finetuning Stage: Fix the arch params and finetune the searched
                         subnet only, use `finetune_mode` as flag.
    Note:
        The 3rd stage is implemented by registering `SubnetFinetuneHook`.

    Args:
        model: the supernet.
        mutator: the mutator for searching the best arch params.
        constraints (dict): resource constraints while searching.
            Default: flops(M): 290.0.
        with_constraints (bool):
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: Optional[Union[DiffModuleMutator, Dict]] = None,
                 fix_subnet: Optional[FixMutable] = None,
                 pretrain_epochs: int = 0,
                 finetune_epochs: int = 80,
                 norm_training: bool = False,
                 with_constraints: bool = False,
                 constraints: Dict = dict(flops=320.0),
                 flops_coef: float = 1e-6,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__(architecture, data_preprocessor, **kwargs)

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
            # Before using it, you must do some preparations according to
            # the supernet.
            self.mutator.prepare_from_supernet(
                self.architecture, is_random=False)
            self.is_supernet = True

        self.norm_training = norm_training
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        assert pretrain_epochs < finetune_epochs, \
            f'finetune stage(>={finetune_epochs} eps) must be later than \
              pretrain stage(<={pretrain_epochs} eps).'

        self.with_constraints = with_constraints
        self.constraints = constraints
        self.flops_coef = flops_coef

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

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``.
        """
        # WARNING: need search space here
        k_list = []
        for n, m in self.architecture.backbone.layers.named_modules():
            if n.split('._candidates')[0] not in k_list:
                k_list.append(n)

        if isinstance(data, (tuple, list)) and isinstance(
                optim_wrapper, OptimWrapperDict):
            assert len(data) == len(optim_wrapper), \
                f'The length of data ({len(data)}) should be equal to that '\
                f'of optimizers ({len(optim_wrapper)}).'
            # TODO check the order of data
            supernet_data, mutator_data = data

            log_vars = dict()
            self.message_hub = MessageHub.get_current_instance()
            cur_epoch = self.message_hub.get_info('epoch')

            # TODO process the input
            # 1. update mutator
            if cur_epoch == self.finetune_epochs:
                # synchronize arch params to start the finetune stage.
                for k, v in self.mutator.arch_params.items():
                    dist.broadcast(v, src=0)
            if cur_epoch >= self.pretrain_epochs and \
               cur_epoch < self.finetune_epochs:
                with optim_wrapper['mutator'].optim_context(self):
                    mutator_loss = self.mutator.compute_loss()
                    mutator_loss = dict(loss=mutator_loss)
                    mutator_losses, mutator_log_vars = \
                        self.parse_losses(mutator_loss)

                optim_wrapper['mutator'].update_params(
                    mutator_losses, retain_graph=True)
                log_vars.update(add_prefix(mutator_log_vars, 'mutator'))
                # deal with the grad of arch params & weights
                self.mutator.handle_grads()

            # 2. update architecture
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
class DsnasDDP(MMDistributedDataParallel):

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
        """
        if isinstance(data, (tuple, list)) and isinstance(
                optim_wrapper, OptimWrapperDict):
            assert len(data) == len(optim_wrapper), \
                f'The length of data ({len(data)}) should be equal to that '\
                f'of optimizers ({len(optim_wrapper)}).'
            # TODO check the order of data
            supernet_data, mutator_data = data

            log_vars = dict()
            self.message_hub = MessageHub.get_current_instance()
            cur_epoch = self.message_hub.get_info('epoch')

            # TODO process the input
            # 1. update mutator
            if cur_epoch == self.module.finetune_epochs:
                # synchronize arch params to start the finetune stage.
                for k, v in self.module.mutator.arch_params.items():
                    dist.broadcast(v, src=0)
            if cur_epoch >= self.module.pretrain_epochs and \
               cur_epoch < self.module.finetune_epochs:
                with optim_wrapper['mutator'].optim_context(self):
                    mutator_loss = self.module.mutator.compute_loss()
                    mutator_loss = dict(loss=mutator_loss)
                    mutator_losses, mutator_log_vars = \
                        self.module.parse_losses(mutator_loss)

                optim_wrapper['mutator'].update_params(
                    mutator_losses, retain_graph=True)
                log_vars.update(add_prefix(mutator_log_vars, 'mutator'))
                # deal with the grad of arch params & weights
                self.module.mutator.handle_grads()

            # 2. update architecture
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
