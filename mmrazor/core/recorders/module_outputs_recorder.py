# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Tuple

from torch import nn

from mmrazor.registry import TASK_UTILS
from .base_recorder import BaseRecorder


@TASK_UTILS.register_module()
class ModuleOutputsRecorder(BaseRecorder):
    """Recorder for intermediate results which are Pytorch moudle's outputs.

    Examples:
            >>> from torch import nn
            >>> class ToyModel(nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.conv1 = nn.Conv2d(8,8,1)
            ...         self.conv2 = nn.Conv2d(1,1,1)
            ...     def forward(self, x):
            ...         x1 = self.conv1(x)
            ...         x2 = self.conv1(x+1)
            ...         return self.conv2(x1 + x2)

            >>> model = ToyModel()
            >>> [ name for name,_ in model.named_modules() ]
            ['conv1', 'conv2']

            >>> r1 = ModuleOutputsRecorder('conv1')
            >>> r1.initialize(model)

            >>> with r1:
            >>>     res = model(torch.randn(1,1,1,1))

            >>> r1.data_buffer
            [tensor([[[[0.6734]]]]), tensor([[[[1.2514]]]]) ]
            >>> r1.get_record_data(record_idx=1)
            tensor([[[[1.2514]]]])

            >>> r2 = ModuleOutputsRecorder('conv2')
            >>> r2.initialize(model)

            >>> with r2:
            >>>     res = model(torch.randn(1,1,1,1))

            >>> r2.data_buffer
            [tensor([[[[0.9534]]]])]
            >>> r2.get_record_data()
            tensor([[[[0.9534]]]])
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._recording = False

    @property
    def recording(self) -> bool:
        """bool: whether to record data in forward hook."""
        return self._recording

    def prepare_from_model(self, model: Optional[nn.Module] = None) -> None:
        """Register Pytorch forward hook to corresponding module."""

        assert model is not None, 'model can not be None.'

        founded = False
        for name, module in model.named_modules():
            if name == self.source:
                module.register_forward_hook(self.forward_hook)
                founded = True
                break

        assert founded, f'"{self.source}" is not in the model.'

    def forward_hook(self, module: nn.Module, inputs: Tuple,
                     outputs: Any) -> None:
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs : The output of the module.
        """
        if self._recording:
            self.data_buffer.append(outputs)

    def __enter__(self):
        """Enter the context manager."""
        super().__enter__()
        self._recording = True

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        super().__exit__(exc_type, exc_value, traceback)
        self._recording = False
