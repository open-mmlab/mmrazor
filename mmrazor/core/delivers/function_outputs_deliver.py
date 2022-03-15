# Copyright (c) OpenMMLab. All rights reserved.
import functools

from mmcv.utils import import_modules_from_strings

from ..builder import DELIVERS
from .distill_deliver import DistillDeliver


@DELIVERS.register_module()
class FunctionOutputsDeliver(DistillDeliver):

    def __init__(self, function, import_module, **kwargs):
        super().__init__(**kwargs)

        imported_module = import_modules_from_strings(  # noqa: F841
            import_module)
        origin_func = eval(f'imported_module.{function}')
        wrapped_func = self.deliver_wrapper(origin_func)  # noqa: F841
        exec(f'imported_module.{function} = wrapped_func')

    def deliver_wrapper(self, origin_func):

        @functools.wraps(origin_func)
        def wrap_func(*args, **kwargs):
            if self.current_mode == self.target:
                assert len(self.data_queue) > 0
                outputs = self.data_queue.pop(0)
            elif self.current_mode == self.source:
                outputs = origin_func(*args, **kwargs)
                assert len(self.data_queue) < self.max_keep_data
                self.data_queue.append(outputs)
            elif self.current_mode == 'eval':
                outputs = origin_func(*args, **kwargs)
            else:
                raise RuntimeError
            return outputs

        return wrap_func
