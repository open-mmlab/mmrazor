# Copyright (c) OpenMMLab. All rights reserved.
import functools

from mmcv.utils import import_modules_from_strings

from ..builder import RECORDERS
from .base_recorder import BaseRecorder


@RECORDERS.register_module()
class MethodOutputsRecorder(BaseRecorder):

    def __init__(self, sources, import_modules):
        super().__init__()
        self.sources = sources
        self.import_modules = import_modules

    def prepare_from_model(self, model):
        for method_name, module_name in zip(self.sources, self.import_modules):
            imported_module = import_modules_from_strings(  # noqa: F841
                module_name)
            origin_method = eval(f'imported_module.{method_name}')
            buffer_key = f'{module_name}.{method_name}'

            self.data_buffer[buffer_key] = list()

            wrapped_method = self.method_record_wrapper(  # noqa: F841
                origin_method, buffer_key)

            exec(f'imported_module.{method_name} = wrapped_method')

    def reset_data_buffer(self):
        for key in self.data_buffer.keys():
            self.data_buffer[key] = list()

    def method_record_wrapper(self, orgin_method, buffer_key):

        @functools.wraps(orgin_method)
        def wrap_method(*args, **kwargs):
            outputs = orgin_method(*args, **kwargs)
            if self.recording:
                self.data_buffer[buffer_key].append(outputs)
            return outputs

        return wrap_method
