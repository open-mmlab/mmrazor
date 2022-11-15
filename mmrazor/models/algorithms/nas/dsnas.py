# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmengine.dist import get_dist_info
from mmengine.logging import MessageHub
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutables.base_mutable import BaseMutable
from mmrazor.models.mutators import DiffModuleMutator
from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODEL_WRAPPERS, MODELS, TASK_UTILS
from mmrazor.structures import load_fix_subnet
from mmrazor.utils import FixMutable
from ..base import BaseAlgorithm


@MODELS.register_module()
class DSNAS(BaseAlgorithm):
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
        flops_constraints (float): Flops constraints for judging whether to
            backward flops loss or not. Default to 300.0(M).
        estimator_cfg (Dict[str, Any]): Used for building a resource estimator.
            Default to None.
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
                 estimator_cfg: Dict[str, Any] = None,
                 norm_training: bool = False,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__(architecture, data_preprocessor, **kwargs)

        # initialize estimator
        estimator_cfg = dict() if estimator_cfg is None else estimator_cfg
        if 'type' not in estimator_cfg:
            estimator_cfg['type'] = 'mmrazor.ResourceEstimator'
        self.estimator = TASK_UTILS.build(estimator_cfg)
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

            self.mutable_module_resources = self._get_module_resources()
            # Mutator is an essential component of the NAS algorithm. It
            # provides some APIs commonly used by NAS.
            # Before using it, you must do some preparations according to
            # the supernet.
            self.mutator.prepare_from_supernet(self.architecture)
            self.is_supernet = True
            self.search_space_name_list = list(
                self.mutator.name2mutable.keys())

        self.norm_training = norm_training
        self.pretrain_epochs = pretrain_epochs
        self.finetune_epochs = finetune_epochs
        if pretrain_epochs >= finetune_epochs:
            raise ValueError(f'Pretrain stage (optional) must be done before '
                             f'finetuning stage. Got `{pretrain_epochs}` >= '
                             f'`{finetune_epochs}`.')

        self.flops_loss_coef = 1e-2
        self.flops_constraints = flops_constraints
        _, self.world_size = get_dist_info()

    def search_subnet(self):
        """Search subnet by mutator."""

        # Avoid circular import
        from mmrazor.structures import export_fix_subnet

        subnet = self.mutator.sample_choices()
        self.mutator.set_choices(subnet)
        return export_fix_subnet(self)

    def fix_subnet(self):
        """Fix subnet when finetuning."""
        subnet = self.mutator.sample_choices()
        self.mutator.set_choices(subnet)
        for module in self.architecture.modules():
            if isinstance(module, BaseMutable):
                if not module.is_fixed:
                    module.fix_chosen(module.current_choice)
        self.is_supernet = False

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
            need_update_mutator = self.need_update_mutator(cur_epoch)

            # TODO process the input
            if cur_epoch == self.finetune_epochs and self.is_supernet:
                # synchronize arch params to start the finetune stage.
                for k, v in self.mutator.arch_params.items():
                    dist.broadcast(v, src=0)
                self.fix_subnet()

            # 1. update architecture
            with optim_wrapper['architecture'].optim_context(self):
                pseudo_data = self.data_preprocessor(data, True)
                supernet_batch_inputs = pseudo_data['inputs']
                supernet_data_samples = pseudo_data['data_samples']
                supernet_loss = self(
                    supernet_batch_inputs, supernet_data_samples, mode='loss')

            supernet_losses, supernet_log_vars = self.parse_losses(
                supernet_loss)
            optim_wrapper['architecture'].backward(
                supernet_losses, retain_graph=need_update_mutator)
            optim_wrapper['architecture'].step()
            optim_wrapper['architecture'].zero_grad()
            log_vars.update(add_prefix(supernet_log_vars, 'supernet'))

            # 2. update mutator
            if need_update_mutator:
                with optim_wrapper['mutator'].optim_context(self):
                    mutator_loss = self.compute_mutator_loss()
                mutator_losses, mutator_log_vars = \
                    self.parse_losses(mutator_loss)
                optim_wrapper['mutator'].update_params(mutator_losses)
                log_vars.update(add_prefix(mutator_log_vars, 'mutator'))
                # handle the grad of arch params & weights
                self.handle_grads()

        else:
            # Enable automatic mixed precision training context.
            with optim_wrapper.optim_context(self):
                pseudo_data = self.data_preprocessor(data, True)
                batch_inputs = pseudo_data['inputs']
                data_samples = pseudo_data['data_samples']
                losses = self(batch_inputs, data_samples, mode='loss')
            parsed_losses, log_vars = self.parse_losses(losses)
            optim_wrapper.update_params(parsed_losses)

        return log_vars

    def _get_module_resources(self):
        """Get resources of spec modules."""

        spec_modules = []
        for name, module in self.architecture.named_modules():
            if isinstance(module, BaseMutable):
                for choice in module.choices:
                    spec_modules.append(name + '._candidates.' + choice)

        mutable_module_resources = self.estimator.estimate_separation_modules(
            self.architecture, dict(spec_modules=spec_modules))

        return mutable_module_resources

    def need_update_mutator(self, cur_epoch: int) -> bool:
        """Whether to update mutator."""
        if cur_epoch >= self.pretrain_epochs and \
           cur_epoch < self.finetune_epochs:
            return True
        return False

    def compute_mutator_loss(self) -> Dict[str, torch.Tensor]:
        """Compute mutator loss.

        In this method, arch_loss & flops_loss[optional] are computed
        by traversing arch_weights & probs in search groups.

        Returns:
            Dict: Loss of the mutator.
        """
        arch_loss = 0.0
        flops_loss = 0.0
        for name, module in self.architecture.named_modules():
            if isinstance(module, BaseMutable):
                k = str(self.search_space_name_list.index(name))
                probs = F.softmax(self.mutator.arch_params[k], -1)
                arch_loss += torch.log(
                    (module.arch_weights * probs).sum(-1)).sum()

                # get the index of op with max arch weights.
                index = (module.arch_weights == 1).nonzero().item()
                _module_key = name + '._candidates.' + module.choices[index]
                flops_loss += probs[index] * \
                    self.mutable_module_resources[_module_key]['flops']

        mutator_loss = dict(arch_loss=arch_loss / self.world_size)

        copied_model = copy.deepcopy(self)
        fix_mutable = copied_model.search_subnet()
        load_fix_subnet(copied_model, fix_mutable)

        subnet_flops = self.estimator.estimate(copied_model)['flops']
        if subnet_flops >= self.flops_constraints:
            mutator_loss['flops_loss'] = \
                (flops_loss * self.flops_loss_coef) / self.world_size

        return mutator_loss

    def handle_grads(self):
        """Handle grads of arch params & arch weights."""
        for name, module in self.architecture.named_modules():
            if isinstance(module, BaseMutable):
                k = str(self.search_space_name_list.index(name))
                self.mutator.arch_params[k].grad.data.mul_(
                    module.arch_weights.grad.data.sum())
                module.arch_weights.grad.zero_()


@MODEL_WRAPPERS.register_module()
class DSNASDDP(MMDistributedDataParallel):

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
            need_update_mutator = self.module.need_update_mutator(cur_epoch)

            # TODO process the input
            if cur_epoch == self.module.finetune_epochs and \
               self.module.is_supernet:
                # synchronize arch params to start the finetune stage.
                for k, v in self.module.mutator.arch_params.items():
                    dist.broadcast(v, src=0)
                self.module.fix_subnet()

            # 1. update architecture
            with optim_wrapper['architecture'].optim_context(self):
                pseudo_data = self.module.data_preprocessor(data, True)
                supernet_batch_inputs = pseudo_data['inputs']
                supernet_data_samples = pseudo_data['data_samples']
                supernet_loss = self(
                    supernet_batch_inputs, supernet_data_samples, mode='loss')

            supernet_losses, supernet_log_vars = self.module.parse_losses(
                supernet_loss)
            optim_wrapper['architecture'].backward(
                supernet_losses, retain_graph=need_update_mutator)
            optim_wrapper['architecture'].step()
            optim_wrapper['architecture'].zero_grad()
            log_vars.update(add_prefix(supernet_log_vars, 'supernet'))

            # 2. update mutator
            if need_update_mutator:
                with optim_wrapper['mutator'].optim_context(self):
                    mutator_loss = self.module.compute_mutator_loss()
                mutator_losses, mutator_log_vars = \
                    self.module.parse_losses(mutator_loss)
                optim_wrapper['mutator'].update_params(mutator_losses)
                log_vars.update(add_prefix(mutator_log_vars, 'mutator'))
                # handle the grad of arch params & weights
                self.module.handle_grads()

        else:
            # Enable automatic mixed precision training context.
            with optim_wrapper.optim_context(self):
                pseudo_data = self.module.data_preprocessor(data, True)
                batch_inputs = pseudo_data['inputs']
                data_samples = pseudo_data['data_samples']
                losses = self(batch_inputs, data_samples, mode='loss')
            parsed_losses, log_vars = self.module.parse_losses(losses)
            optim_wrapper.update_params(parsed_losses)

        return log_vars
