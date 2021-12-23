# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Microsoft Corporation.
import copy

import torch
from torch import nn

from mmrazor.models.builder import ALGORITHMS
from .base import BaseAlgorithm


@ALGORITHMS.register_module()
class Darts(BaseAlgorithm):

    def __init__(self, unroll, **kwargs):

        super(Darts, self).__init__(**kwargs)
        self.unroll = unroll

    def train_step(self, data, optimizer):
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

        if isinstance(data, (tuple, list)) and isinstance(optimizer, dict):
            assert len(data) == len(optimizer)

            train_arch_data, train_supernet_data = data

            optimizer['mutator'].zero_grad()
            if self.unroll:
                self._unrolled_backward(train_arch_data, train_supernet_data,
                                        optimizer)
            else:

                arch_losses = self(**train_arch_data)
                arch_loss, _ = self._parse_losses(arch_losses)
                arch_loss.backward()
            optimizer['mutator'].step()

            model_losses = self(**train_supernet_data)
            model_loss, log_vars = self._parse_losses(model_losses)

            optimizer['architecture'].zero_grad()
            model_loss.backward()
            nn.utils.clip_grad_norm_(
                self.architecture.parameters(), max_norm=5, norm_type=2)
            optimizer['architecture'].step()

            outputs = dict(
                loss=model_loss,
                log_vars=log_vars,
                num_samples=len(train_supernet_data['img'].data))

        else:

            outputs = super(Darts, self).train_step(data, optimizer)
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
