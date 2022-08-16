# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any

import torch
import torch.distributed as dist


class GatherTensors(torch.autograd.Function):
    """Gather tensors from all GPUS, supporting backward propagation. see more
    details in torch.distributed.all_gather and torch.distributed.all_reduce.

    Args:
        ctx: Context to be used for forward propagation.
        input (torch.Tensor): Tensor to be broadcast from current process.
    """

    # TODO: The return type of this function will report an error in python3.7.
    # error: Incompatible return value type (got "Tuple[Any, ...]",
    # expected "Tuple[Any]")
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor):
        """Forward function.

        It must accept a context ctx as the first argument.

        The context can be used to store tensors that can be then retrieved
        during the backward pass.

        Args:
            input (torch.Tensor): Tensor to be broadcast from current process.
        """
        output = [
            torch.empty_like(input) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> torch.Tensor:
        """Backward function.

        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs did :func:`forward` return, and it should return as
        many tensors, as there were inputs to :func:`forward`. Each argument is
        the gradient w.r.t the given output, and each returned value should be
        the gradient w.r.t. the corresponding input.

        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computated w.r.t. the
        output.

        Args:
            grads (torch.Tensor): Grads to be merged from current process.
        """
        rank = dist.get_rank()
        merged = torch.stack(grads)
        dist.all_reduce(merged)
        return merged[rank]
