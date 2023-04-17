# Copyright (c) OpenMMLab. All rights reserved.
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.models.architectures.dynamic_ops import (DynamicConv2d,
                                                      DynamicLinear)
from .utils import ModuleProtocol, torch_setting


class SparseGptMixIn(ModuleProtocol):

    # init

    def _sparse_gpt_mix_in_init(self):
        self.sparse_gpt_handles = []
        self.rows = self.weight_matrix.shape[0]
        self.columns = self.weight_matrix.shape[1]

        _hessian = torch.zeros([self.columns, self.columns])
        self.register_buffer('_hessian', _hessian)
        self._hessian: torch.Tensor
        self.hessian_batch = 0

    # weight and input adaptive

    @property
    def weight_matrix(self):
        """Return weight with shape (out in)"""
        return self.weight.flatten(1)  # out in

    @weight_matrix.setter
    def weight_matrix(self, value: torch.Tensor):
        with torch.no_grad():
            value = value.reshape(self.weight.shape).to(self.weight.device).to(
                self.weight.dtype)
            self.weight.data = value

    def format_input(self, input: torch.Tensor):
        """Return input with shape (B N C)"""
        if len(input.shape) == 2:  # N C
            input = input.unsqueeze(0)  # 1 N C
        return input

    # compute hessian

    @property
    def hessian(self):
        """hessian always return float."""
        self._hessian = self._hessian.float()
        return self._hessian

    @hessian.setter
    def hessian(self, value: torch.Tensor):
        with torch.no_grad():
            self._hessian = value.float()

    @torch.no_grad()
    def update_hessian(self, input: torch.Tensor):

        input = self.format_input(input).float()

        assert len(input.shape) == 3
        B = input.shape[0]  # B N C
        input = input.transpose(0, -1).flatten(1)  # C D

        H = input @ input.T * 2  # C C
        self.hessian = (self.hessian * self.hessian_batch + H) / (
            self.hessian_batch + B)
        self.hessian_batch = self.hessian_batch + B

    def start_init_hessian(self):

        @torch.no_grad()
        def forward_pre_hook(module: Protocol, input: tuple):
            assert len(input) == 1
            self.update_hessian(input[0])

        handle = self.register_forward_pre_hook(forward_pre_hook)
        self.sparse_gpt_handles.append(handle)

    def end_init_hessian(self):
        for h in self.sparse_gpt_handles:
            h.remove()

    # prune

    @torch.no_grad()
    def prune(self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01):
        with torch_setting(dtype=torch.float):
            # Converted from https://github.com/ist-daslab/sparsegpt

            assert self.hessian is not None
            W: torch.Tensor = self.weight_matrix.float()  # out in

            H = self.hessian

            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            Losses = torch.zeros(self.rows, device=W.device)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=W.device)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            mask = None

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                if prunen == 0:
                    if mask is not None:
                        mask1 = mask[:, i1:i2]
                    else:
                        tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1)))**2
                        thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() *
                                                                  sparsity)]
                        mask1 = tmp <= thresh
                else:
                    mask1 = torch.zeros_like(W1) == 1

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if prunen != 0 and i % prunem == 0:
                        tmp = W1[:, i:(i + prunem)]**2 / (torch.diag(Hinv1)[i:(
                            i + prunem)].reshape((1, -1)))**2
                        mask1.scatter_(
                            1, i +
                            torch.topk(tmp, prunen, dim=1, largest=False)[1],
                            True)

                    q = w.clone()
                    q[mask1[:, i]] = 0

                    Q1[:, i] = q
                    Losses1[:, i] = (w - q)**2 / d**2

                    err1 = (w - q) / d
                    W1[:,
                       i:] -= err1.unsqueeze(1).matmul(Hinv1[i,
                                                             i:].unsqueeze(0))
                    Err1[:, i] = err1

                W[:, i1:i2] = Q1
                Losses += torch.sum(Losses1, 1) / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            torch.cuda.synchronize()
            from .sparse24_utils import is_weight_sparse_24
            if prunen == 2 and prunem == 4:
                assert is_weight_sparse_24(
                    W, -1), f'Weight dose not satisfy 24 with shape {W.shape}'
            error = torch.sum(Losses)

            if torch.isnan(error).any():
                raise Exception('get nan error')
            else:
                self.weight_matrix = W.data

            return error


# SparseGpt Ops for Linear and Conv2d


class SparseGptLinear(DynamicLinear, SparseGptMixIn):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sparse_gpt_mix_in_init()

    @classmethod
    def convert_from(cls, module: nn.Conv2d) -> 'DynamicConv2d':
        new_module = super().convert_from(module)
        new_module.load_state_dict(module.state_dict(), strict=False)

        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
        new_module = new_module.to(device).to(dtype)

        return new_module


class SparseGptConv2d(DynamicConv2d, SparseGptMixIn):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sparse_gpt_mix_in_init()

    @classmethod
    def convert_from(cls, module: nn.Conv2d) -> 'DynamicConv2d':
        new_module = super().convert_from(module)
        new_module.load_state_dict(module.state_dict(), strict=False)

        device = next(module.parameters()).device
        dtype = next(module.parameters()).dtype
        new_module = new_module.to(device).to(dtype)

        return new_module

    def format_input(self, input: torch.Tensor):
        # input B C H W
        input = F.unfold(
            input, self.kernel_size, padding=self.padding,
            stride=self.stride)  # B C D
        return input.transpose(-1, -2)
