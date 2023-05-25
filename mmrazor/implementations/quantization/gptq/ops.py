# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd

from mmrazor.models.architectures.dynamic_ops import (DynamicConv2d,
                                                      DynamicLinear)
# from mmrazor.implementations.pruning.sparse_gpt.utils import torch_setting
from .gptq import GPTQMixIn

try:
    import triton
    import triton.language as tl

    from . import custom_autotune

    # code based https://github.com/fpgaminer/GPTQ-triton
    @custom_autotune.autotune(
        configs=[
            triton.Config(
                {
                    'BLOCK_SIZE_M': 64,
                    'BLOCK_SIZE_N': 256,
                    'BLOCK_SIZE_K': 32,
                    'GROUP_SIZE_M': 8
                },
                num_stages=4,
                num_warps=4),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 128,
                    'BLOCK_SIZE_N': 128,
                    'BLOCK_SIZE_K': 32,
                    'GROUP_SIZE_M': 8
                },
                num_stages=4,
                num_warps=4),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 64,
                    'BLOCK_SIZE_N': 128,
                    'BLOCK_SIZE_K': 32,
                    'GROUP_SIZE_M': 8
                },
                num_stages=4,
                num_warps=4),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 128,
                    'BLOCK_SIZE_N': 32,
                    'BLOCK_SIZE_K': 32,
                    'GROUP_SIZE_M': 8
                },
                num_stages=4,
                num_warps=4),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 64,
                    'BLOCK_SIZE_N': 64,
                    'BLOCK_SIZE_K': 32,
                    'GROUP_SIZE_M': 8
                },
                num_stages=4,
                num_warps=4),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 64,
                    'BLOCK_SIZE_N': 128,
                    'BLOCK_SIZE_K': 32,
                    'GROUP_SIZE_M': 8
                },
                num_stages=2,
                num_warps=8),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 64,
                    'BLOCK_SIZE_N': 64,
                    'BLOCK_SIZE_K': 64,
                    'GROUP_SIZE_M': 8
                },
                num_stages=3,
                num_warps=8),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 32,
                    'BLOCK_SIZE_N': 32,
                    'BLOCK_SIZE_K': 128,
                    'GROUP_SIZE_M': 8
                },
                num_stages=2,
                num_warps=4),
        ],
        key=['M', 'N', 'K'],
        nearest_power_of_two=True,
        prune_configs_by={
            'early_config_prune':
            custom_autotune.matmul248_kernel_config_pruner,
            'perf_model': None,
            'top_k': None,
        },
    )
    @triton.jit
    def matmul_248_kernel(a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr, M,
                          N, K, bits, maxq, stride_am, stride_ak, stride_bk,
                          stride_bn, stride_cm, stride_cn, stride_scales,
                          stride_zeros, BLOCK_SIZE_M: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr,
                          BLOCK_SIZE_K: tl.constexpr,
                          GROUP_SIZE_M: tl.constexpr):
        """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, K) float16
        B is of shape (K//8, N) int32
        C is of shape (M, N) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N) float16
        g_ptr is of shape (K) int32
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8
        # times
        b_ptrs = b_ptr + (
            (offs_k[:, None] // infearure_per_bits) * stride_bk +
            offs_bn[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        g_ptrs = g_ptr + offs_k
        # shifter is used to extract the N bits of each element in the 32-bit
        # word from B
        scales_ptrs = scales_ptr + offs_bn[None, :]
        zeros_ptrs = zeros_ptr + (offs_bn[None, :] // infearure_per_bits)

        shifter = (offs_k % infearure_per_bits) * bits
        zeros_shifter = (offs_bn % infearure_per_bits) * bits
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, num_pid_k):
            g_idx = tl.load(g_ptrs)

            # Fetch scales and zeros; these are per-outfeature and thus reused
            # in the inner loop
            scales = tl.load(scales_ptrs + g_idx[:, None] *
                             stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(
                zeros_ptrs +
                g_idx[:, None] * stride_zeros)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

            zeros = (zeros >> zeros_shifter[None, :]) & maxq
            zeros = (zeros + 1)

            a = tl.load(
                a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
            b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

            # Now we need to unpack b (which is N-bit values) into 32-bit
            # values
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
            b = (b - zeros) * scales  # Scale and shift

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
            g_ptrs += BLOCK_SIZE_K

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[
            None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    @custom_autotune.autotune(
        configs=[
            triton.Config(
                {
                    'BLOCK_SIZE_M': 64,
                    'BLOCK_SIZE_N': 32,
                    'BLOCK_SIZE_K': 256,
                    'GROUP_SIZE_M': 8
                },
                num_stages=4,
                num_warps=4),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 128,
                    'BLOCK_SIZE_N': 32,
                    'BLOCK_SIZE_K': 128,
                    'GROUP_SIZE_M': 8
                },
                num_stages=4,
                num_warps=4),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 64,
                    'BLOCK_SIZE_N': 32,
                    'BLOCK_SIZE_K': 128,
                    'GROUP_SIZE_M': 8
                },
                num_stages=4,
                num_warps=4),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 128,
                    'BLOCK_SIZE_N': 32,
                    'BLOCK_SIZE_K': 32,
                    'GROUP_SIZE_M': 8
                },
                num_stages=4,
                num_warps=4),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 64,
                    'BLOCK_SIZE_N': 32,
                    'BLOCK_SIZE_K': 64,
                    'GROUP_SIZE_M': 8
                },
                num_stages=4,
                num_warps=4),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 64,
                    'BLOCK_SIZE_N': 32,
                    'BLOCK_SIZE_K': 128,
                    'GROUP_SIZE_M': 8
                },
                num_stages=2,
                num_warps=8),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 64,
                    'BLOCK_SIZE_N': 64,
                    'BLOCK_SIZE_K': 64,
                    'GROUP_SIZE_M': 8
                },
                num_stages=3,
                num_warps=8),
            triton.Config(
                {
                    'BLOCK_SIZE_M': 32,
                    'BLOCK_SIZE_N': 128,
                    'BLOCK_SIZE_K': 32,
                    'GROUP_SIZE_M': 8
                },
                num_stages=2,
                num_warps=4),
        ],
        key=['M', 'N', 'K'],
        nearest_power_of_two=True)
    @triton.jit
    def transpose_matmul_248_kernel(
            a_ptr, b_ptr, c_ptr, scales_ptr, zeros_ptr, g_ptr, M, N, K, bits,
            maxq, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
            stride_cn, stride_scales, stride_zeros, BLOCK_SIZE_M: tl.constexpr,
            BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
            GROUP_SIZE_M: tl.constexpr):
        """
        Compute the matrix multiplication C = A x B.
        A is of shape (M, N) float16
        B is of shape (K//8, N) int32
        C is of shape (M, K) float16
        scales is of shape (G, N) float16
        zeros is of shape (G, N) float16
        g_ptr is of shape (K) int32
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_k
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_k = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_n[None, :] * stride_ak
        )  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        a_mask = (offs_am[:, None] < M)
        # b_ptrs is set up such that it repeats elements along the K axis 8
        # times
        b_ptrs = b_ptr + (
            (offs_bk[:, None] // infearure_per_bits) * stride_bk +
            offs_n[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        g_ptrs = g_ptr + offs_bk
        g_idx = tl.load(g_ptrs)

        # shifter is used to extract the N bits of each element in the 32-bit
        # word from B
        scales_ptrs = scales_ptr + offs_n[
            None, :] + g_idx[:, None] * stride_scales
        zeros_ptrs = zeros_ptr + (offs_n[None, :] // infearure_per_bits
                                  ) + g_idx[:, None] * stride_zeros

        shifter = (offs_bk % infearure_per_bits) * bits
        zeros_shifter = (offs_n % infearure_per_bits) * bits
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

        for n in range(0, num_pid_n):
            # Fetch scales and zeros; these are per-outfeature and thus reused
            # in the inner loop
            scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)

            zeros = (zeros >> zeros_shifter[None, :]) & maxq
            zeros = (zeros + 1)

            a = tl.load(
                a_ptrs, mask=a_mask, other=0.)  # (BLOCK_SIZE_M, BLOCK_SIZE_N)
            b = tl.load(b_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

            # Now we need to unpack b (which is N-bit values) into 32-bit
            # values
            b = (b >> shifter[:, None]) & maxq  # Extract the N-bit values
            b = (b - zeros) * scales  # Scale and shift
            b = tl.trans(b)

            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_N
            b_ptrs += BLOCK_SIZE_N
            scales_ptrs += BLOCK_SIZE_N
            zeros_ptrs += (BLOCK_SIZE_N // infearure_per_bits)

        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bk[
            None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bk[None, :] < K)
        tl.store(c_ptrs, accumulator, mask=c_mask)
except:  # noqa: E722
    print('triton not installed.')


def matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    """matmul248 function with matmul_248_kernel."""
    with torch.cuda.device(input.device):
        output = torch.empty((input.shape[0], qweight.shape[1]),
                             device=input.device,
                             dtype=torch.float16)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(  # noqa: E731
                input.shape[0], META['BLOCK_SIZE_M']) * triton.  # noqa: E731
            cdiv(  # noqa: E731
                qweight.shape[1], META['BLOCK_SIZE_N']), )  # noqa: E731
        matmul_248_kernel[grid](input, qweight, output, scales, qzeros, g_idx,
                                input.shape[0], qweight.shape[1],
                                input.shape[1], bits, maxq, input.stride(0),
                                input.stride(1), qweight.stride(0),
                                qweight.stride(1), output.stride(0),
                                output.stride(1), scales.stride(0),
                                qzeros.stride(0))
        return output


def transpose_matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq):
    """transpose_matmul248 function with transpose_matmul_248_kernel."""
    with torch.cuda.device(input.device):
        output_dim = (qweight.shape[0] * 32) // bits
        output = torch.empty((input.shape[0], output_dim),
                             device=input.device,
                             dtype=torch.float16)
        grid = lambda META: (  # noqa: E731
            triton.cdiv(input.shape[0], META['BLOCK_SIZE_M'])  # noqa: E731
            * triton.cdiv(output_dim, META['BLOCK_SIZE_K']), )  # noqa: E731
        transpose_matmul_248_kernel[grid](input, qweight, output, scales,
                                          qzeros, g_idx, input.shape[0],
                                          qweight.shape[1], output_dim,
                                          bits, maxq, input.stride(0),
                                          input.stride(1), qweight.stride(0),
                                          qweight.stride(1), output.stride(0),
                                          output.stride(1), scales.stride(0),
                                          qzeros.stride(0))
        return output


class QuantLinearFunction(torch.autograd.Function):
    """Custom QuantLinearFunction."""

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, qweight, scales, qzeros, g_idx, bits, maxq):
        """Custom forward."""
        output = matmul248(input, qweight, scales, qzeros, g_idx, bits, maxq)
        ctx.save_for_backward(qweight, scales, qzeros, g_idx)
        ctx.bits, ctx.maxq = bits, maxq
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """Custom backward."""
        qweight, scales, qzeros, g_idx = ctx.saved_tensors
        bits, maxq = ctx.bits, ctx.maxq
        grad_input = None

        if ctx.needs_input_grad[0]:
            grad_input = transpose_matmul248(grad_output, qweight, scales,
                                             qzeros, g_idx, bits, maxq)
        return grad_input, None, None, None, None, None, None


class TritonGPTQLinear(nn.Module, GPTQMixIn):
    """Custom Linear for GPTQ with custom triton kernel."""

    def __init__(self, bits, groupsize, weight, in_features, out_features,
                 bias):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError('Only 2,4,8 bits are supported.')
        self.weight = weight
        self.bias = bias

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else in_features

        self.register_buffer(
            'qweight',
            torch.zeros((in_features // 32 * self.bits, out_features),
                        dtype=torch.int32))
        self.register_buffer(
            'qzeros',
            torch.zeros((math.ceil(
                in_features / self.groupsize), out_features // 32 * self.bits),
                        dtype=torch.int32))
        self.register_buffer(
            'scales',
            torch.zeros(
                (math.ceil(in_features / self.groupsize), out_features),
                dtype=torch.float16))
        self.register_buffer(
            'g_idx',
            torch.tensor([i // self.groupsize for i in range(in_features)],
                         dtype=torch.int32))

        self._gptq_mix_in_init()

    @property
    def is_custom_kernel(self):
        """Whether use custom kernel."""
        return True

    @classmethod
    def convert_from(cls, module: nn.Linear, bits, groupsize):
        """Convert to cls from torch's module."""
        new_module = cls(
            bits,
            groupsize,
            weight=module.weight,
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias)

        return new_module

    def forward(self, x):
        """Custom forward."""
        if torch.all(self.qweight == 0):
            out = F.linear(x, self.weight, self.bias)
        else:
            # import pdb;pdb.set_trace()
            out_shape = x.shape[:-1] + (self.out_features, )
            out = QuantLinearFunction.apply(
                x.reshape(-1, x.shape[-1]), self.qweight, self.scales,
                self.qzeros, self.g_idx, self.bits, self.maxq)
            out = out + self.bias if self.bias is not None else out
            out = out.reshape(out_shape)
            # import pdb;pdb.set_trace()
        return out


class GPTQLinear(DynamicLinear, GPTQMixIn):
    """Custom Linear for GPTQ without custom triton kernel."""

    def __init__(self, a_fakequant=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gptq_mix_in_init()
        self.a_fakequant = a_fakequant
        self.fix_qparams = False

    @property
    def is_custom_kernel(self):
        """Whether use custom kernel."""
        return False

    @classmethod
    def convert_from(cls,
                     module: nn.Linear,
                     a_fakequant=None) -> 'DynamicLinear':
        """Convert to cls from torch's module."""
        new_module = cls(
            a_fakequant=a_fakequant,
            in_features=module.in_features,
            out_features=module.out_features,
            bias=True if module.bias is not None else False)
        new_module.load_state_dict(module.state_dict(), strict=False)

        dtype = next(module.parameters()).dtype
        new_module = new_module.to(dtype)

        return new_module

    def forward(self, input: Tensor) -> Tensor:
        """Custom forward."""
        if self.a_fakequant:
            dtype = self.weight.dtype
            if not self.fix_qparams:
                self.a_fakequant.find_params(input)
            input = self.a_fakequant.quantize(input).to(dtype)
        return super().forward(input)


class GPTQConv2d(DynamicConv2d, GPTQMixIn):
    """Custom Conv2d for GPTQ without custom triton kernel."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._gptq_mix_in_init()

    @property
    def is_custom_kernel(self):
        """Whether use custom kernel."""
        return False

    @classmethod
    def convert_from(cls, module: nn.Conv2d) -> 'DynamicConv2d':
        """Convert to cls from torch's module."""
        new_module = super().convert_from(module)
        new_module.load_state_dict(module.state_dict(), strict=False)

        dtype = next(module.parameters()).dtype
        new_module = new_module.to(dtype)

        return new_module

    def format_input(self, input: torch.Tensor):
        """Format input shape."""
        # input B C H W
        input = F.unfold(
            input, self.kernel_size, padding=self.padding,
            stride=self.stride)  # B C D
        return input.transpose(-1, -2)
