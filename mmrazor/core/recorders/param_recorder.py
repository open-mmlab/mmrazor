# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from torch import nn

from mmrazor.registry import TASK_UTILS
from .base_recorder import BaseRecorder


@TASK_UTILS.register_module()
class ParameterRecorder(BaseRecorder):
    """Recorder for Pytorch model's parameters.

    Examples:
        >>> from torch import nn
        >>> class ToyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.toy_conv = nn.Conv2d(1,1,1)
        ...     def forward(self, x):
        ...         return self.toy_conv(x)

        >>> model = ToyModel()
        >>> [ name for name,_ in model.named_parameters() ]
        ['toy_conv.weight', 'toy_conv.bias']

        >>> recorder = ParameterRecorder('toy_conv.weight')
        >>> recorder.initialize(model)

        >>> recorder.data_buffer
        [Parameter containing: tensor([[[[0.3244]]]], requires_grad=True)]
        >>> recorder.get_record_data()
        Parameter containing: tensor([[[[0.3244]]]], requires_grad=True)
    """

    def prepare_from_model(self, model: Optional[nn.Module] = None) -> None:
        """Record the Pytorch model's parameters."""
        assert model is not None, \
            'model can not be None when use ParameterRecorder.'

        founded = False
        for param_name, param in model.named_parameters():
            if param_name == self.source:
                self.data_buffer.append(param)
                founded = True
                break

        assert founded, f'"{self.source}" is not in the model.'

    def reset_data_buffer(self):
        """Clear data in data_buffer.

        Note:
            The data_buffer stores the address of the parameter in memory and
            does not need to be reset.
        """
        pass
