# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Dict, List, Optional, Union

import torch
from mmengine import BaseDataElement
from mmengine.model import BaseModel
from mmengine.optim import OptimWrapper, OptimWrapperDict
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmrazor.models.mutators import DiffModuleMutator
from mmrazor.models.subnet import (SINGLE_MUTATOR_RANDOM_SUBNET, FixSubnet,
                                   FixSubnetMixin)
from mmrazor.registry import MODELS
from ..base import BaseAlgorithm, LossResults

VALID_FIX_SUBNET = Union[str, FixSubnet, Dict[str, Dict[str, Any]]]


@MODELS.register_module()
class Darts(BaseAlgorithm, FixSubnetMixin):
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
                 mutator: Optional[Union[DiffModuleMutator, Dict]] = None,
                 fix_subnet: Optional[VALID_FIX_SUBNET] = None,
                 unroll: bool = False,
                 norm_training: bool = False,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(architecture, data_preprocessor, init_cfg)

        # Darts has two training mode: supernet training and subnet retraining.
        # fix_subnet is not None, means subnet retraining.
        if fix_subnet:
            # According to fix_subnet, delete the unchosen part of supernet
            self.load_fix_subnet(fix_subnet, prefix='architecture.')
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
            self.mutator.prepare_from_supernet(self.architecture)
            self.is_supernet = True

        self.norm_training = norm_training
        self.unroll = unroll

    def sample_subnet(self) -> SINGLE_MUTATOR_RANDOM_SUBNET:
        """Random sample subnet by mutator."""
        return self.mutator.sample_choices()

    def set_subnet(self, subnet: SINGLE_MUTATOR_RANDOM_SUBNET):
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
                f'The length of data {len(data)} should be equal to that of optimizers {len(optim_wrapper)}.'  # noqa: E501

            # TODO check the order of data
            train_supernet_data, train_arch_data = data

            # TODO mutator optimizer zero_grad
            optim_wrapper.zero_grad()

            if self.unroll:
                self._unrolled_backward(train_arch_data, train_supernet_data,
                                        optim_wrapper)  # TODO optimizer
            else:
                # TODO process the input
                arch_loss = self.loss(train_arch_data)  # noqa: F841
                # arch_loss.backward()

            # TODO mutator optimizer step
            optim_wrapper.step()

            model_loss = self.loss(train_supernet_data)

            # TODO optimizer architecture zero_grad
            optim_wrapper.zero_grad()
            # model_loss.backward()

            nn.utils.clip_grad_norm_(
                self.architecture.parameters(), max_norm=5, norm_type=2)

            # TODO optimizer architecture step
            optim_wrapper.step()

            outputs = dict(
                loss=model_loss,
                num_samples=len(train_supernet_data['img'].data))
        else:
            outputs = super().train_step(data, optim_wrapper)

        return outputs

    def _unrolled_backward(self, train_arch_data, train_supernet_data,
                           optimizer):
        """Compute unrolled loss and backward its gradients."""
        backup_params = copy.deepcopy(tuple(self.architecture.parameters()))

        # do virtual step on training data
        lr = optimizer['architecture'].param_groups[0]['lr']
        momentum = optimizer['architecture'].param_groups[0]['momentum']
        weight_decay = optimizer['architecture'].param_groups[0][
            'weight_decay']
        self._compute_virtual_model(train_supernet_data, lr, momentum,
                                    weight_decay, optimizer)

        # calculate unrolled loss on validation data
        # keep gradients for model here for compute hessian
        losses = self(**train_arch_data)
        loss, _ = self._parse_losses(losses)
        w_model, w_arch = tuple(self.architecture.parameters()), tuple(
            self.mutator.parameters())
        w_grads = torch.autograd.grad(loss, w_model + w_arch)
        d_model, d_arch = w_grads[:len(w_model)], w_grads[len(w_model):]

        # compute hessian and final gradients
        hessian = self._compute_hessian(backup_params, d_model,
                                        train_supernet_data)
        with torch.no_grad():
            for param, d, h in zip(w_arch, d_arch, hessian):
                # gradient = dalpha - lr * hessian
                param.grad = d - lr * h

        # restore weights
        self._restore_weights(backup_params)

    def _compute_virtual_model(self, data, lr, momentum, weight_decay,
                               optimizer):
        """Compute unrolled weights w`"""
        # don't need zero_grad, using autograd to calculate gradients
        losses = self(**data)
        loss, _ = self._parse_losses(losses)
        gradients = torch.autograd.grad(loss, self.architecture.parameters())
        with torch.no_grad():
            for w, g in zip(self.architecture.parameters(), gradients):
                m = optimizer['architecture'].state[w].get(
                    'momentum_buffer', 0.)
                w = w - lr * (momentum * m + g + weight_decay * w)

    def _restore_weights(self, backup_params):
        with torch.no_grad():
            for param, backup in zip(self.architecture.parameters(),
                                     backup_params):
                param.copy_(backup)

    def _compute_hessian(self, backup_params, dw, data):
        """
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

            losses = self(**data)
            loss, _ = self._parse_losses(losses)
            dalphas.append(
                torch.autograd.grad(loss, tuple(self.mutator.parameters())))
        # dalpha { L_trn(w+) }, # dalpha { L_trn(w-) }
        dalpha_pos, dalpha_neg = dalphas
        hessian = [(p - n) / (2. * eps)
                   for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
