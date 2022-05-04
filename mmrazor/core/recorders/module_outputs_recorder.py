# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from torch import nn

from ..builder import RECORDERS
from .base_recorder import BaseRecorder


@RECORDERS.register_module()
class ModuleOutputsRecorder(BaseRecorder):
    """Recorder for intermediate results which are Pytorch moudle's outputs.

    Args:
        sources List(str): The names of the Pytorch modules whose output needs
        to be recorded.

    Examples:
            >>> import copy
            >>> from torch.nn import Module, ModuleList, Conv2d
            >>> from mmrazor.core import build_recorder

            >>> # example module
            >>> class RepeatModule(Module):
            >>>     def __init__(self) -> None:
            >>>         super().__init__()
            >>>     def forward(self, x):
            >>>         outputs = list()
            >>>         for i in range(2):
            >>>             outputs.append(x * i)
            >>>         return outputs

            >>> # example model
            >>> class Model(Module):
            >>>     def __init__(self) -> None:
            >>>        super().__init__()
            >>>        self.repeat_module1 = RepeatModule()
            >>>        self.repeat_module2 = RepeatModule()
            >>>     def forward(self, x):
            >>>        out = self.repeat_module1(x)
            >>>        out = self.repeat_module2(x)
            >>>        return out

            >>> # example recorder config
            >>> recorder_cfg = dict(
            >>>     type='ModuleOutputs',
            >>>     sources=['repeat_module1', 'repeat_module2'])

            >>> recorder_cfg_ = copy.deepcopy(recorder_cfg)
            >>> recorder_cfg_.type = recorder_cfg.type + 'Recorder'

            >>> ctx = build_recorder(recorder_cfg_)
            >>> model = Model()
            >>> ctx.initialize(model)

            >>> with ctx:
            >>>     res = model(torch.ones(2))

            >>> ctx.data_buffer
            >>> {'repeat_module1': [[tensor([0., 0.]), tensor([1., 1.])]],
                 'repeat_module2': [[tensor([0., 0.]), tensor([1., 1.])]]}
            >>> ctx.get_record_data('repeat_module1')
            >>> [[tensor([0., 0.]), tensor([1., 1.])]]
            >>> ctx.get_record_data('repeat_module1', data_index=1)
            >>> [tensor([0., 0.]), tensor([1., 1.])]
    """

    def __init__(self, sources: List):
        super().__init__()
        self.sources: List(str) = sources

    def prepare_from_model(self, model: nn.Module) -> None:
        """Register Pytorch forward hook to corresponding module."""

        self.module2name: Dict(nn.Module, str) = dict()
        for module_name, module in model.named_modules():
            self.module2name[module] = module_name

        self.name2module: Dict(str, nn.Module) = dict(model.named_modules())

        for module_name in self.sources:
            # init data_buffer, data_buffer must be Dict(str, list)
            # assume a module execute N times, there will be N outputs need to
            # save.
            self.data_buffer[module_name] = list()
            module = self.name2module[module_name]
            module.register_forward_hook(self.forward_output_hook)

    def forward_output_hook(self, module, inputs, outputs) -> None:
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        if self.recording:
            module_name = self.module2name[module]
            # self.data_buffer: Dict(str, list)
            self.data_buffer[module_name].append(outputs)
