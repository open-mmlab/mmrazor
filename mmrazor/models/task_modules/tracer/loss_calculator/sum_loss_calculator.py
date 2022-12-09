# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import TASK_UTILS


@TASK_UTILS.register_module()
class SumPseudoLoss:
    """Calculate the pseudo loss to trace the topology by summing all output
    tensors.

    Args:
        input_shape (Tuple): The shape of the pseudo input. Defaults to
            (2, 3, 224, 224).
    """

    def __init__(self, input_shape=(2, 3, 224, 224)):
        self.input_shape = input_shape

    def __call__(self, model) -> torch.Tensor:
        pseudo_img = torch.rand(self.input_shape)
        model.eval()
        pseudo_output = model(pseudo_img)
        return self._sum_of_output(pseudo_output)

    def _sum_of_output(self, tensor):
        """Get a loss by summing all tensors."""
        if isinstance(tensor, torch.Tensor):
            return tensor.sum()
        elif isinstance(tensor, list) or isinstance(tensor, tuple):
            loss = 0
            for t in tensor:
                loss = loss + self._sum_of_output(t)
            return loss
        elif isinstance(tensor, dict):
            loss = 0
            for t in tensor.values():
                loss = loss + self._sum_of_output(t)
            return loss
        else:
            raise NotImplementedError(
                f'unsuppored type{type(tensor)} to get shape of tensors.')
