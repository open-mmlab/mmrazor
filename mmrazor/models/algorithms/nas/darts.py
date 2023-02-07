# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Dict, List, Optional, Union

import torch
from mmengine.model import BaseModel, MMDistributedDataParallel
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutators import NasMutator
from mmrazor.models.utils import add_prefix
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from ..base import BaseAlgorithm

VALID_MUTATOR_TYPE = Union[NasMutator, Dict]


@MODELS.register_module()
class Darts(BaseAlgorithm):
    """Implementation of `DARTS <https://arxiv.org/abs/1806.09055>`_

    DARTS means Differentiable Architecture Search, a classic NAS algorithm.
    :class:`Darts` implements the APIs required by the DARTS, as well as the
    supernet training and subnet retraining logic for each iter.

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
    """

    def __init__(self,
                 architecture: Union[BaseModel, Dict],
                 mutator: VALID_MUTATOR_TYPE = None,
                 unroll: bool = False,
                 norm_training: bool = False,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(architecture, data_preprocessor, init_cfg)

        self.mutator = self._build_mutator(mutator)
        # Mutator is an essential component of the NAS algorithm. It
        # provides some APIs commonly used by NAS.
        # Before using it, you must do some preparation according to
        # the supernet.
        self.mutator.prepare_from_supernet(self.architecture)
        self.mutator.prepare_arch_params()

        self.norm_training = norm_training
        self.unroll = unroll

    def _build_mutator(self, mutator: VALID_MUTATOR_TYPE = None) -> NasMutator:
        """Build mutator."""
        if isinstance(mutator, dict):
            mutator = MODELS.build(mutator)
        if not isinstance(mutator, NasMutator):
            raise TypeError('mutator should be a `dict` or `NasMutator` '
                            f'instance, but got {type(mutator)}.')
        return mutator

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

            supernet_data, mutator_data = data

            log_vars = dict()

            # Update the parameter of mutator
            if self.unroll:
                with optim_wrapper['mutator'].optim_context(self):
                    optim_wrapper['mutator'].zero_grad()
                    mutator_log_vars = self._unrolled_backward(
                        mutator_data, supernet_data, optim_wrapper)
                optim_wrapper['mutator'].step()
                log_vars.update(add_prefix(mutator_log_vars, 'mutator'))
            else:
                with optim_wrapper['mutator'].optim_context(self):
                    pseudo_data = self.data_preprocessor(mutator_data, True)
                    mutator_batch_inputs = pseudo_data['inputs']
                    mutator_data_samples = pseudo_data['data_samples']
                    mutator_loss = self(
                        mutator_batch_inputs,
                        mutator_data_samples,
                        mode='loss')
                mutator_losses, mutator_log_vars = self.parse_losses(
                    mutator_loss)
                optim_wrapper['mutator'].update_params(mutator_losses)
                log_vars.update(add_prefix(mutator_log_vars, 'mutator'))

            # Update the parameter of supernet
            with optim_wrapper['architecture'].optim_context(self):
                pseudo_data = self.data_preprocessor(supernet_data, True)
                supernet_batch_inputs = pseudo_data['inputs']
                supernet_data_samples = pseudo_data['data_samples']
                supernet_loss = self(
                    supernet_batch_inputs, supernet_data_samples, mode='loss')
            supernet_losses, supernet_log_vars = self.parse_losses(
                supernet_loss)
            optim_wrapper['architecture'].update_params(supernet_losses)
            log_vars.update(add_prefix(supernet_log_vars, 'supernet'))

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

    def _unrolled_backward(self, mutator_data, supernet_data, optim_wrapper):
        """Compute unrolled loss and backward its gradients."""
        backup_params = copy.deepcopy(tuple(self.architecture.parameters()))

        # Do virtual step on training data
        lr = optim_wrapper['architecture'].param_groups[0]['lr']
        momentum = optim_wrapper['architecture'].param_groups[0]['momentum']
        weight_decay = optim_wrapper['architecture'].param_groups[0][
            'weight_decay']
        self._compute_virtual_model(supernet_data, lr, momentum, weight_decay,
                                    optim_wrapper['architecture'])

        # Calculate unrolled loss on validation data
        # Keep gradients for model here for compute hessian
        pseudo_data = self.data_preprocessor(mutator_data, True)
        mutator_batch_inputs = pseudo_data['inputs']
        mutator_data_samples = pseudo_data['data_samples']
        mutator_loss = self(
            mutator_batch_inputs, mutator_data_samples, mode='loss')
        mutator_losses, mutator_log_vars = self.parse_losses(mutator_loss)

        # Here we use the backward function of optimWrapper to calculate
        # the gradients of mutator loss. The gradients of model and arch
        # can directly obtained. For more information, please refer to
        # https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py
        optim_wrapper['mutator'].backward(mutator_losses)
        d_model = [param.grad for param in self.architecture.parameters()]
        d_arch = [param.grad for param in self.mutator.parameters()]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, supernet_data,
                                        optim_wrapper['architecture'])

        w_arch = tuple(self.mutator.parameters())

        with torch.no_grad():
            for param, d, h in zip(w_arch, d_arch, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)
        return mutator_log_vars

    def _compute_virtual_model(self, supernet_data, lr, momentum, weight_decay,
                               optim_wrapper):
        """Compute unrolled weights w`"""
        # don't need zero_grad, using autograd to calculate gradients
        pseudo_data = self.data_preprocessor(supernet_data, True)
        supernet_batch_inputs = pseudo_data['inputs']
        supernet_data_samples = pseudo_data['data_samples']
        supernet_loss = self(
            supernet_batch_inputs, supernet_data_samples, mode='loss')
        supernet_loss, _ = self.parse_losses(supernet_loss)

        optim_wrapper.backward(supernet_loss)
        gradients = [param.grad for param in self.architecture.parameters()]

        with torch.no_grad():
            for w, g in zip(self.architecture.parameters(), gradients):
                m = optim_wrapper.optimizer.state[w].get('momentum_buffer', 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _restore_weights(self, backup_params):
        """restore weight from backup params."""
        with torch.no_grad():
            for param, backup in zip(self.architecture.parameters(),
                                     backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, supernet_data,
                         optim_wrapper) -> List:
        """compute hession metric
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) }  \
                - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1E-8:
            print(
                'In computing hessian, norm is smaller than 1E-8, \
                cause eps to be %.6f.', norm.item())

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.architecture.parameters(), dw):
                    p += e * d

            pseudo_data = self.data_preprocessor(supernet_data, True)
            supernet_batch_inputs = pseudo_data['inputs']
            supernet_data_samples = pseudo_data['data_samples']
            supernet_loss = self(
                supernet_batch_inputs, supernet_data_samples, mode='loss')
            supernet_loss, _ = self.parse_losses(supernet_loss)

            optim_wrapper.backward(supernet_loss)
            dalpha = [param.grad for param in self.mutator.parameters()]
            dalphas.append(dalpha)

        # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        dalpha_pos, dalpha_neg = dalphas
        hessian = [(p - n) / (2. * eps)
                   for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian


class BatchNormWrapper(nn.Module):
    """Wrapper for BatchNorm.

    For more information, Please refer to
    https://github.com/NVIDIA/apex/issues/121
    """

    def __init__(self, m):
        super(BatchNormWrapper, self).__init__()
        self.m = m
        # Set the batch norm to eval mode
        self.m.eval()

    def forward(self, x):
        """Convert fp16 to fp32 when forward."""
        input_type = x.dtype
        x = self.m(x.float())
        return x.to(input_type)


@MODEL_WRAPPERS.register_module()
class DartsDDP(MMDistributedDataParallel):
    """DDP for Darts and rewrite train_step of MMDDP."""

    def __init__(self,
                 *,
                 device_ids: Optional[Union[List, int, torch.device]] = None,
                 **kwargs) -> None:
        if device_ids is None:
            if os.environ.get('LOCAL_RANK') is not None:
                device_ids = [int(os.environ['LOCAL_RANK'])]
        super().__init__(device_ids=device_ids, **kwargs)

        fp16 = True
        if fp16:

            def add_fp16_bn_wrapper(model):
                for child_name, child in model.named_children():
                    if isinstance(child, nn.BatchNorm2d):
                        setattr(model, child_name, BatchNormWrapper(child))
                    else:
                        add_fp16_bn_wrapper(child)

            add_fp16_bn_wrapper(self.module)

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

            supernet_data, mutator_data = data

            log_vars = dict()

            # Update the parameter of mutator
            if self.module.unroll:
                with optim_wrapper['mutator'].optim_context(self):
                    optim_wrapper['mutator'].zero_grad()
                    mutator_log_vars = self._unrolled_backward(
                        mutator_data, supernet_data, optim_wrapper)
                optim_wrapper['mutator'].step()
                log_vars.update(add_prefix(mutator_log_vars, 'mutator'))
            else:
                with optim_wrapper['mutator'].optim_context(self):
                    pseudo_data = self.module.data_preprocessor(
                        mutator_data, True)
                    mutator_batch_inputs = pseudo_data['inputs']
                    mutator_data_samples = pseudo_data['data_samples']
                    mutator_loss = self(
                        mutator_batch_inputs,
                        mutator_data_samples,
                        mode='loss')

                    mutator_losses, mutator_log_vars = self.module.parse_losses(  # noqa: E501
                        mutator_loss)
                    optim_wrapper['mutator'].update_params(mutator_losses)
                    log_vars.update(add_prefix(mutator_log_vars, 'mutator'))

            # Update the parameter of supernet
            with optim_wrapper['architecture'].optim_context(self):
                pseudo_data = self.module.data_preprocessor(
                    supernet_data, True)
                supernet_batch_inputs = pseudo_data['inputs']
                supernet_data_samples = pseudo_data['data_samples']
                supernet_loss = self(
                    supernet_batch_inputs, supernet_data_samples, mode='loss')

                supernet_losses, supernet_log_vars = self.module.parse_losses(
                    supernet_loss)

                optim_wrapper['architecture'].update_params(supernet_losses)
                log_vars.update(add_prefix(supernet_log_vars, 'supernet'))

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

    def _unrolled_backward(self, mutator_data, supernet_data, optim_wrapper):
        """Compute unrolled loss and backward its gradients."""
        backup_params = copy.deepcopy(
            tuple(self.module.architecture.parameters()))

        # do virtual step on training data
        lr = optim_wrapper['architecture'].param_groups[0]['lr']
        momentum = optim_wrapper['architecture'].param_groups[0]['momentum']
        weight_decay = optim_wrapper['architecture'].param_groups[0][
            'weight_decay']
        self._compute_virtual_model(supernet_data, lr, momentum, weight_decay,
                                    optim_wrapper['architecture'])

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        pseudo_data = self.module.data_preprocessor(mutator_data, True)
        mutator_batch_inputs = pseudo_data['inputs']
        mutator_data_samples = pseudo_data['data_samples']
        mutator_loss = self(
            mutator_batch_inputs, mutator_data_samples, mode='loss')
        mutator_losses, mutator_log_vars = self.module.parse_losses(
            mutator_loss)

        # Here we use the backward function of optimWrapper to calculate
        # the gradients of mutator loss. The gradients of model and arch
        # can directly obtained. For more information, please refer to
        # https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/optimizer_wrapper.py
        optim_wrapper['mutator'].backward(mutator_losses)
        d_model = [
            param.grad for param in self.module.architecture.parameters()
        ]
        d_arch = [param.grad for param in self.module.mutator.parameters()]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model, supernet_data,
                                        optim_wrapper['architecture'])

        w_arch = tuple(self.module.mutator.parameters())

        with torch.no_grad():
            for param, da, he in zip(w_arch, d_arch, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = da - lr * he

        # restore weights
        self._restore_weights(backup_params)
        return mutator_log_vars

    def _compute_virtual_model(self, supernet_data, lr, momentum, weight_decay,
                               optim_wrapper):
        """Compute unrolled weights w`"""
        # don't need zero_grad, using autograd to calculate gradients
        pseudo_data = self.module.data_preprocessor(supernet_data, True)
        supernet_batch_inputs = pseudo_data['inputs']
        supernet_data_samples = pseudo_data['data_samples']
        supernet_loss = self(
            supernet_batch_inputs, supernet_data_samples, mode='loss')
        supernet_loss, _ = self.module.parse_losses(supernet_loss)

        optim_wrapper.backward(supernet_loss)
        gradients = [
            param.grad for param in self.module.architecture.parameters()
        ]

        with torch.no_grad():
            for w, g in zip(self.module.architecture.parameters(), gradients):
                m = optim_wrapper.optimizer.state[w].get('momentum_buffer', 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _restore_weights(self, backup_params):
        """restore weight from backup params."""
        with torch.no_grad():
            for param, backup in zip(self.module.architecture.parameters(),
                                     backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, supernet_data,
                         optim_wrapper) -> List:
        """compute hession metric
            dw = dw` { L_val(w`, alpha) }
            w+ = w + eps * dw
            w- = w - eps * dw
            hessian = (dalpha { L_trn(w+, alpha) }  \
                - dalpha { L_trn(w-, alpha) }) / (2*eps)
            eps = 0.01 / ||dw||
        """
        self._restore_weights(backup_params)
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        if norm < 1E-8:
            print(
                'In computing hessian, norm is smaller than 1E-8, \
                cause eps to be %.6f.', norm.item())

        dalphas = []
        for e in [eps, -2. * eps]:
            # w+ = w + eps*dw`, w- = w - eps*dw`
            with torch.no_grad():
                for p, d in zip(self.module.architecture.parameters(), dw):
                    p += e * d

            pseudo_data = self.module.data_preprocessor(supernet_data, True)
            supernet_batch_inputs = pseudo_data['inputs']
            supernet_data_samples = pseudo_data['data_samples']
            supernet_loss = self(
                supernet_batch_inputs, supernet_data_samples, mode='loss')
            supernet_loss, _ = self.module.parse_losses(supernet_loss)

            optim_wrapper.backward(supernet_loss)
            dalpha = [param.grad for param in self.module.mutator.parameters()]
            dalphas.append(dalpha)

        # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        dalpha_pos, dalpha_neg = dalphas
        hessian = [(p - n) / (2. * eps)
                   for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
