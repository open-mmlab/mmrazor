# Copyright (c) OpenMMLab. All rights reserved.
import sys

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

import numpy as np
import torch
import torch.distributed as dist

from mmrazor.implementations.pruning.sparse_gpt.utils import torch_setting


class ModuleProtocol(Protocol):
    """Custom module protocol for algorithm mixin."""
    weight: torch.Tensor

    def forward(self, x):
        """The abstract method."""
        pass

    def register_forward_hook(self, hook):
        """The abstract method."""
        pass

    def register_backward_hook(self, hook):
        """The abstract method."""
        pass

    def register_forward_pre_hook(self, hook):
        """The abstract method."""
        pass

    def register_buffer(self, name, tensor):
        """The abstract method."""
        pass


class GPTQMixIn(ModuleProtocol):
    """The core algorithm implementation for GPTQ."""

    def _gptq_mix_in_init(self):
        """Init mixin."""
        self.gptq_handles = []
        self.rows = self.weight_matrix.shape[0]
        self.columns = self.weight_matrix.shape[1]

        self._hessian: torch.Tensor = None
        self.hessian_batch = 0

    # weight and input adaptive

    @property
    def weight_matrix(self):
        """Return weight with shape (out in)"""
        return self.weight.flatten(1)  # out in

    @weight_matrix.setter
    def weight_matrix(self, value: torch.Tensor):
        """Set weight."""
        with torch.no_grad():
            value = value.reshape(self.weight.shape).to(self.weight.device).to(
                self.weight.dtype)
            self.weight.data.copy_(value)

    def format_input(self, input: torch.Tensor):
        """Return input with shape (B N C)"""
        if len(input.shape) == 2:  # N C
            input = input.unsqueeze(0)  # 1 N C
        return input

    # compute hessian

    @property
    def hessian(self):
        """hessian always return float."""
        if dist.is_initialized():
            if dist.get_rank() == 0:
                assert self._hessian is not None, 'hessian is not initialized.'
                hessian = self._hessian.to(self.weight_matrix.device)
            else:
                hessian = torch.zeros(
                    self.columns,
                    self.columns,
                    device=self.weight_matrix.device)
            dist.broadcast(hessian, 0)
            return hessian
        else:
            return self._hessian

    @hessian.setter
    def hessian(self, value: torch.Tensor):
        """Set hessian."""
        with torch.no_grad():
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    assert self._hessian is not None, 'hessian is not initialized.'  # noqa
                    self._hessian.data.copy_(
                        value.data.to(self._hessian.device))
                else:
                    self._hessian = None
            else:
                self._hessian.data.copy_(value.data.to(self._hessian.device))

    @torch.no_grad()
    def update_hessian(self, input: torch.Tensor):
        """Update hessian."""
        input = self.format_input(input).float()
        H_save = self.hessian
        H_save = H_save.to(input.device)

        assert len(input.shape) == 3
        B = input.shape[0]  # B N C
        input = input.transpose(0, -1).flatten(1)  # C D

        H = input @ input.T * 2  # C C

        if dist.is_initialized():
            dist.all_reduce(H)
            B *= dist.get_world_size()
        H_save = (H_save * self.hessian_batch + H) / (self.hessian_batch + B)
        self.hessian = H_save
        self.hessian_batch = self.hessian_batch + B

    def register_hessian_hook(self):
        """Register updating hessian hook."""

        @torch.no_grad()
        def forward_pre_hook(module: Protocol, input: tuple):
            assert len(input) == 1
            self.update_hessian(input[0])

        handle = self.register_forward_pre_hook(forward_pre_hook)
        self.gptq_handles.append(handle)

    def remove_hessian_hook(self):
        """Remove updating hessian hook."""
        for h in self.gptq_handles:
            h.remove()

    def init_hessian(self, device=None):
        """Init hessian."""
        if dist.is_initialized():
            if dist.get_rank() == 0:
                self._hessian = torch.zeros([self.columns, self.columns],
                                            device=device,
                                            dtype=torch.float)
            else:
                self._hessian = None
        else:
            self._hessian = torch.zeros([self.columns, self.columns],
                                        device=device,
                                        dtype=torch.float)

    def pack(self, scales, zeros, g_idx=None):
        """Pack and update qparams with groupsize_idx."""
        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if self.bias is not None:
            self.bias.half()

        intweight = []
        for idx in range(self.in_features):
            intweight.append(
                torch.round(
                    (self.weight.data[:, idx] + scale_zeros[self.g_idx[idx]]) /
                    self.scales[self.g_idx[idx]]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.cpu().numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * self.bits, intweight.shape[1]),
            dtype=np.uint32)
        i = 0
        row = 0
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError('Only 2,4,8 bits are supported.')

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight).to(self.weight.device)

        zeros -= 1
        zeros = zeros.cpu().numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits),
                          dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError('Only 2,4,8 bits are supported.')

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros).to(self.weight.device)

    @torch.no_grad()
    def quant(self,
              quantizer,
              blocksize=128,
              percdamp=0.01,
              groupsize=-1,
              actorder=False):
        """The implementation for GPTQ."""
        with torch_setting(dtype=torch.float):
            assert self.hessian is not None
            W: torch.Tensor = self.weight_matrix.float()  # out in

            if not quantizer.ready():
                quantizer.find_params(W, weight=True)

            H = self.hessian.float().to(W.device)
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            if actorder:
                perm = torch.argsort(torch.diag(H), descending=True)
                W = W[:, perm]
                H = H[perm][:, perm]

            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=W.device)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            g_idx = []
            scale = []
            zero = []
            now_idx = 1

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if groupsize != -1:
                        if (i1 + i) % groupsize == 0:
                            quantizer.find_params(
                                W[:, (i1 + i):(i1 + i + groupsize)],
                                weight=True)

                        if ((i1 + i) // groupsize) - now_idx == -1:
                            scale.append(quantizer.scale)
                            zero.append(quantizer.zero)
                            now_idx += 1

                    q = quantizer.quantize(w.unsqueeze(1)).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q)**2 / d**2

                    err1 = (w - q) / d
                    W1[:,
                       i:] -= err1.unsqueeze(1).matmul(Hinv1[i,
                                                             i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            torch.cuda.synchronize()
            error = torch.sum(Losses).item()

            groupsize = groupsize if groupsize != -1 else self.columns
            g_idx = [i // groupsize for i in range(self.columns)]
            g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
            if actorder:
                invperm = torch.argsort(perm)
                Q = Q[:, invperm]
                g_idx = g_idx[invperm]

            if scale == []:
                scale.append(quantizer.scale)
                zero.append(quantizer.zero)
            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)
            self.weight_matrix = Q.data.to(self.weight_matrix.dtype)
            if self.is_custom_kernel:
                self.pack(scale, zero, g_idx)
                del self.weight
            return error

    def free(self):
        """Free some cache and memory."""
        self._hessian = None
        torch.cuda.empty_cache()
