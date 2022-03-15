# Copyright (c) OpenMMLab. All rights reserved.
import functools

from mmcv.utils import import_modules_from_strings
from .distill_deliver import DistillDeliver
from ..builder import DELIVERS

@DELIVERS.register_module()
class MethodOutputsDeliver(DistillDeliver):

    def __init__(self, method, import_module, **kwargs):
        super().__init__(**kwargs)
        
        imported_module = import_modules_from_strings(  # noqa: F841
            import_module)
        origin_method = eval(f'imported_module.{method}')
        wrapped_method = self.deliver_wrapper(origin_method)  # noqa: F841
        exec(f'imported_module.{method} = wrapped_method')

    def deliver_wrapper(self, origin_method):
        @functools.wraps(origin_method)
        def wrap_method(*args, **kwargs):
            
            if self.current_mode == self.target:
                assert len(self.data_queue) > 0
                outputs = self.data_queue.pop(0)
            elif self.current_mode == self.source:
                outputs = origin_method(*args, **kwargs)
                assert len(self.data_queue) < self.max_keep_data
                self.data_queue.append(outputs)
            elif self.current_mode == 'eval':
                outputs = origin_method(*args, **kwargs)
            else:
                raise RuntimeError
            return outputs

        return wrap_method