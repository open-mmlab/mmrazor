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
from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.models.mutators import DiffModuleMutator
from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from mmrazor.structures import FlopsEstimator, export_fix_subnet
from mmrazor.utils import FixMutable
from ..base import BaseAlgorithm


@MODELS.register_module()
class Dsnas(BaseAlgorithm):
    """Implementation of `DSNAS <https://arxiv.org/abs/2002.09128>`_

    Args:
        architecture (dict|:obj:`BaseModel`): The config of :class:`BaseModel`
            or built model. Corresponding to supernet in NAS algorithm.
        mutator (dict|:obj:`DiffModuleMutator`): The config of
            :class:`DiffModuleMutator` or built mutator.
        fix_subnet (str | dict | :obj:`FixSubnet`): The path of yaml file or
            loaded dict or built :obj:`FixSubnet`.
        pretrain_epochs (int): Num of epochs for supernet pretraining.
        finetune_epochs (int): Num of epochs for subnet finetuning.
        norm_training (bool): Whether to set norm layers to training mode,
            namely, not freeze running stats (mean and var). Note: Effect on
            Batch Norm and its variants only. Defaults to False.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`. Defaults to None.
        init_cfg (dict): Init config for ``BaseModule``.

    Note:
        Dsnas doesn't require retraining. It has 3 stages in searching:
        1. `cur_epoch` < `pretrain_epochs` refers to supernet pretraining.
        2. `pretrain_epochs` <= `cur_epoch` < `finetune_epochs` refers to
                normal supernet training while mutator is updated.
        3. `cur_epoch` >= `finetune_epochs` refers to subnet finetuning.
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: Optional[Union[DiffModuleMutator, Dict]] = None,
                 fix_subnet: Optional[FixMutable] = None,
                 pretrain_epochs: int = 0,
                 finetune_epochs: int = 80,
                 flops_constraints: float = 300.0,
                 norm_training: bool = False,
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

        self.is_fixed = False
        self.is_measured = False
        self.norm_training = norm_training
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        assert pretrain_epochs < finetune_epochs, \
            f'finetune stage(>={finetune_epochs} epochs) must be later than \
              pretrain stage(<={pretrain_epochs} epochs).'
        self.flops_constraints = flops_constraints

    def set_subnet(self):
        subnet = self.mutator.sample_choices()
        self.mutator.set_choices(subnet)

    def search_subnet(self):
        """Search subnet by mutator."""

        # Avoid circular import
        from mmrazor.structures import export_fix_subnet
        self.set_subnet()
        return export_fix_subnet(self)
    
    def fix_subnet(self):
        """Fix subnet when finetuning."""
        self.set_subnet()
        for name, module in self.architecture.named_modules():
            if isinstance(module, BaseMutable):
                module.fix_chosen(module.choices)

    def _get_subnet_constraints(self):
        """Get model constraints.

        Returns:
            fix_subnet_flops: The result of model constraints.
        """
        fix_mutable = self.search_subnet()
        return FlopsEstimator.get_model_complexity_info(
            self,
            fix_mutable=fix_mutable,
            as_strings=False,
            print_per_layer_stat=False)[0]
    
    def _get_mutable_constraints(self):
        """Get mutable constraints by recording per layer stat.
        
        Returns:
            Supernet with per layer stat recorded.
        """
        return FlopsEstimator.get_model_complexity_info(
            self,
            fix_mutable=None,
            as_strings=False,
            print_per_layer_stat=True,
            add_attr_for_each=True,
            return_resources_model=True)[2]

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
        if isinstance(optim_wrapper, OptimWrapperDict):
            log_vars = dict()
            self.message_hub = MessageHub.get_current_instance()
            cur_epoch = self.message_hub.get_info('epoch')

            # TODO process the input
            # 1. update mutator
            if cur_epoch >= self.pretrain_epochs and \
               cur_epoch < self.finetune_epochs:
                with optim_wrapper['mutator'].optim_context(self):
                    mutator_loss = self.mutator.compute_loss()
                    mutator_losses, mutator_log_vars = \
                        self.parse_losses(mutator_loss)

                optim_wrapper['mutator'].update_params(
                    mutator_losses, retain_graph=True)
                log_vars.update(add_prefix(mutator_log_vars, 'mutator'))
                # deal with the grad of arch params & weights
                self.mutator.handle_grads()
            if cur_epoch == self.finetune_epochs:
                if not self.is_fixed:
                    # synchronize arch params to start the finetune stage.
                    for k, v in self.mutator.arch_params.items():
                        dist.broadcast(v, src=0)
                    self.fix_subnet()
                    self.is_fixed = True

            # 2. update architecture
            with optim_wrapper['architecture'].optim_context(self):
                supernet_batch_inputs, supernet_data_samples = \
                    self.data_preprocessor(data, True)
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
        if isinstance(optim_wrapper, OptimWrapperDict):
            log_vars = dict()
            self.message_hub = MessageHub.get_current_instance()
            cur_epoch = self.message_hub.get_info('epoch')

            # TODO process the input
            # 1. update mutator
            if cur_epoch >= self.module.pretrain_epochs and \
               cur_epoch < self.module.finetune_epochs:
                if cur_epoch == self.module.pretrain_epochs + 1 and \
                   not self.module.is_measured:
                    flops_model = self.module._get_mutable_constraints()
                    self.module.is_measured = True
                subnet_flops = self.module._get_subnet_constraints()
                update_flops_loss = False
                if subnet_flops >= self.module.flops_constraints:
                    update_flops_loss = True
                with optim_wrapper['mutator'].optim_context(self):
                    mutator_loss = self.module.mutator.compute_loss(
                        flops_model, update_flops_loss)
                    mutator_losses, mutator_log_vars = \
                        self.module.parse_losses(mutator_loss)

                optim_wrapper['mutator'].update_params(
                    mutator_losses, retain_graph=True)
                log_vars.update(add_prefix(mutator_log_vars, 'mutator'))
                # deal with the grad of arch params & weights
                self.module.mutator.handle_grads()
            if cur_epoch == self.module.finetune_epochs and \
               not self.module.is_fixed:
                # synchronize arch params to start the finetune stage.
                for k, v in self.module.mutator.arch_params.items():
                    dist.broadcast(v, src=0)
                self.module.fix_subnet()
                self.module.is_fixed = True

            # 2. update architecture
            with optim_wrapper['architecture'].optim_context(self):
                supernet_batch_inputs, supernet_data_samples = \
                    self.module.data_preprocessor(data, True)
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
