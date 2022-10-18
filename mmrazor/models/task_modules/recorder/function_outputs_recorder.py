# Copyright (c) OpenMMLab. All rights reserved.
import functools
from types import FunctionType, ModuleType
from typing import Callable, List, Optional

from mmengine.utils import import_modules_from_strings
from torch import nn

from mmrazor.registry import TASK_UTILS
from .base_recorder import BaseRecorder


@TASK_UTILS.register_module()
class FunctionOutputsRecorder(BaseRecorder):
    """Recorder for intermediate results which are ``FunctionType``'s outputs.

     Notes:
        The form of `source` needs special attention. For example,
        `anchor_inside_flags` is a function in mmdetection to check whether the
        anchors are inside the border. This function is in
        `mmdet/core/anchor/utils.py` and used in
        `mmdet/models/dense_heads/anchor_head.py`. Then the source should be
        `mmdet.models.dense_heads.anchor_head.anchor_inside_flags` but not
        `mmdet.core.anchor.utils.anchor_inside_flags`.


    Examples:
        >>> # Below code in toy_module.py
        >>> import random
        >>> def toy_func():
        ...     return random.randint(0, 1000)
        >>> def toy_list_func():
        ...     return [random.randint(0,1000) for _ in range(3)]

        >>> # Below code in main.py
        >>> # Now, we want to get teacher's outputs by recorder.

        >>> import toy_module
        >>> r1 = FunctionOutputsRecorder('toy_module.toy_func')
        >>> r1.initialize()
        >>> with r1:
        ...     output_teacher1 = toy_module.toy_func()
        ...     output_teacher2 = toy_module.toy_func()
        ...     output_teacher3 = toy_module.toy_func()

        >>> r1.data_buffer
        [33, 41, 12]
        >>> recorder.get_record_data(record_idx=2)
        12
        >>> output_teacher1==33 and output_teacher2==41 and output_teacher3==41
        True

        >>> r2 = FunctionOutputsRecorder('toy_module.toy_list_func')
        >>> r2.initialize()
        >>> with r2:
        ...     output_teacher1 = toy_module.toy_list_func()
        ...     output_teacher2 = toy_module.toy_list_func()
        ...     output_teacher3 = toy_module.toy_list_func()

        >>> r2.data_buffer
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> r2.get_record_data(record_idx=2, data_idx=2)
        9
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._check_valid_source(self.source)

    @staticmethod
    def _check_valid_source(source):
        """Check if the source's format is valid."""
        if not isinstance(source, str):
            raise TypeError(f'source should be a str '
                            f'instance, but got {type(source)}')

        assert len(source.split('.')) > 1, \
            'source must have at least one `.`'

    @property
    def func_name(self):
        """Get the function name according to `func_path`."""
        return self.source.split('.')[-1]

    @property
    def module_string(self):
        """Get the module name according to `func_path`."""
        return '.'.join(self.source.split('.')[:-1])

    def prepare_from_model(self, model: Optional[nn.Module] = None) -> None:
        """The `model` is useless in `FunctionOutputsRecorder`."""
        pass

    def func_record_wrapper(self, origin_func: Callable,
                            data_buffer: List) -> Callable:
        """Save the function's outputs.

        Args:
            origin_func (FunctionType): The method whose outputs need to be
                recorded.
            data_buffer (list): A list of data.
        """

        @functools.wraps(origin_func)
        def wrap_func(*args, **kwargs):
            outputs = origin_func(*args, **kwargs)
            # assume a func execute N times, there will be N outputs need to
            # save.
            data_buffer.append(outputs)
            return outputs

        return wrap_func

    def __enter__(self):
        """Enter the context manager."""
        super().__enter__()

        # import the function corrosponding module
        try:
            mod = import_modules_from_strings(self.module_string)
        except ImportError:
            raise ImportError(
                f'{self.module_string} is not imported correctly.')

        self.imported_module: ModuleType = mod

        assert hasattr(mod, self.func_name), \
            f'{self.func_name} is not in {self.module_string}.'

        origin_func = getattr(mod, self.func_name)
        if not isinstance(origin_func, FunctionType):
            raise TypeError(f'{self.func_name} should be a FunctionType '
                            f'instance, but got {type(origin_func)}')

        self.origin_func: Callable = origin_func

        # add record wrapper to origin function.
        record_func = self.func_record_wrapper(origin_func, self.data_buffer)

        assert hasattr(mod, self.func_name), \
            f'{self.func_name} is not in {self.module_string}.'

        # rewrite the origin function
        setattr(mod, self.func_name, record_func)

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        super().__exit__(exc_type, exc_value, traceback)

        mod = self.imported_module
        origin_func = self.origin_func

        assert hasattr(mod, self.func_name), \
            f'{self.func_name} is not in {self.module_string}.'

        # restore the origin function
        setattr(mod, self.func_name, origin_func)

        # self.imported_module and self.origin_func can not be pickled.
        # Delete these two attributes to avoid errors when ema model is used.
        del self.imported_module
        del self.origin_func
