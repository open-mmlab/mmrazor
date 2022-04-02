# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import RECORDERS
from .base_recorder import BaseRecorder


@RECORDERS.register_module()
class ModuleOutputsRecorder(BaseRecorder):

    def __init__(self, sources):
        super().__init__()
        self.sources = sources

    def prepare_from_model(self, model):
        self.module2name = {}

        for module_name, module in model.named_modules():
            self.module2name[module] = module_name
        self.name2module = dict(model.named_modules())

        for module_name in self.sources:
            self.data_buffer[module_name] = list()
            module = self.name2module[module_name]
            module.register_forward_hook(self.forward_output_hook)

    def reset_data_buffer(self):
        for key in self.data_buffer.keys():
            self.data_buffer[key] = list()

    def forward_output_hook(self, module, inputs, outputs):
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        if self.recording:
            module_name = self.module2name[module]
            self.data_buffer[module_name].append(outputs)
