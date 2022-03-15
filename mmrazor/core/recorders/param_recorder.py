# Copyright (c) OpenMMLab. All rights reserved.
from .base_recorder import BaseRecorder
from ..builder import RECORDERS

@RECORDERS.register_module()
class ParameterRecorder(BaseRecorder):
    def __init__(self, sources):
        super().__init__()
        self.sources = sources

    def prepare_from_model(self, model):
        for param_name, param in model.named_parameters():
            if param_name in self.sources:
                self.data_buffer[param_name] = param

    def reset_data_buffer(self):
        pass
