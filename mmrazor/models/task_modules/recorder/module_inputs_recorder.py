# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Tuple

from torch import nn

from mmrazor.registry import TASK_UTILS
from .module_outputs_recorder import ModuleOutputsRecorder


@TASK_UTILS.register_module()
class ModuleInputsRecorder(ModuleOutputsRecorder):
    """Recorder for intermediate results which are Pytorch moudle's inputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_hook(self, module: nn.Module, inputs: Tuple,
                     outputs: Any) -> None:
        """Save the module's forward input.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs : The output of the module.
        """
        if self.recording:
            self.data_buffer.append(inputs)
