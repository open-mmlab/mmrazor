# Copyright (c) OpenMMLab. All rights reserved.
from .base_recorder import BaseRecorder
import functools
from mmcv.utils import import_modules_from_strings
from ..builder import RECORDERS

@RECORDERS.register_module()
class FunctionOutputsRecorder(BaseRecorder):
    def __init__(self, sources, import_modules):
        super().__init__()
        self.sources = sources
        self.import_modules = import_modules

    def prepare_from_model(self, model):
        for func_name, module_name in zip(self.sources,self.import_modules):
            imported_module = import_modules_from_strings(module_name)
            origin_func = eval(f'imported_module.{func_name}')
            buffer_key = f'{module_name}.{func_name}'

            self.data_buffer[buffer_key] = list()

            wrapped_func = self.func_record_wrapper(  # noqa: F841
                    origin_func, buffer_key)

            exec(f'imported_module.{func_name} = wrapped_func')

    def reset_data_buffer(self):
        for key in self.data_buffer.keys():
            self.data_buffer[key] = list()


    def func_record_wrapper(self, origin_func, buffer_key):

        @functools.wraps(origin_func)
        def wrap_func(*args, **kwargs):
            outputs = origin_func(*args, **kwargs)
            if self.recording:
                self.data_buffer[buffer_key].append(outputs)
            return outputs

        return wrap_func