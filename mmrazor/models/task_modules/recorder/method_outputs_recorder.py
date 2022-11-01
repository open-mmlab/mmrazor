# Copyright (c) OpenMMLab. All rights reserved.
import functools
from types import FunctionType, ModuleType
from typing import Callable, List, Optional

from mmengine.utils import import_modules_from_strings
from torch import nn

from mmrazor.registry import TASK_UTILS
from .base_recorder import BaseRecorder


@TASK_UTILS.register_module()
class MethodOutputsRecorder(BaseRecorder):
    """Recorder for intermediate results which are ``MethodType``'s outputs.

    Note:
        Different from ``FunctionType``, ``MethodType`` is the type of methods
        of class instances.

    Examples:
        >>> # Below code in toy_module.py
        >>> import random
        >>> class Toy():
        ...     def toy_func(self):
        ...         return random.randint(0, 1000)
        ...     def toy_list_func(self):
        ...         return [random.randint(0, 1000) for _ in range(3)]

        >>> # Below code in main.py
        >>> # Now, we want to get teacher's outputs by recorder.

        >>> from toy_module import Toy
        >>> toy = Toy()
        >>> r1 = MethodOutputsRecorder('toy_module.Toy.toy_func')
        >>> r1.initialize()
        >>> with r1:
        ...     output_teacher1 = toy.toy_func()
        ...     output_teacher2 = toy.toy_func()
        ...     output_teacher3 = toy.toy_func()

        >>> r1.data_buffer
        [33, 41, 12]
        >>> r1.get_record_data(record_idx=2)
        12
        >>> output_teacher1==33 and output_teacher2==41 and output_teacher3==41
        True

        >>> r2 = MethodOutputsRecorder('toy_module.Toy.toy_list_func'
        >>> r2.initialize()
        >>> with r2:
        ...     output_teacher1 = toy.toy_list_func()
        ...     output_teacher2 = toy.toy_list_func()
        ...     output_teacher3 = toy.toy_list_func()

        >>> r2.data_buffer
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> r2.get_record_data(record_idx=2, data_idx=2)
        9
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._check_valid_source(self.source)

        # import the function corrosponding module
        try:
            mod: ModuleType = import_modules_from_strings(self.module_string)
        except ImportError:
            raise ImportError(
                f'{self.module_string} is not imported correctly.')

        assert hasattr(mod, self.cls_name),  \
            f'{self.cls_name} is not in {self.module_string}.'

        imported_cls: type = getattr(mod, self.cls_name)
        if not isinstance(imported_cls, type):
            raise TypeError(f'{self.cls_name} should be a type '
                            f'instance, but got {type(imported_cls)}')
        self.imported_class = imported_cls

        assert hasattr(imported_cls, self.method_name), \
            f'{self.method_name} is not in {self.cls_name}.'

        origin_method = getattr(imported_cls, self.method_name)
        if not isinstance(origin_method, FunctionType):
            raise TypeError(f'{self.method_name} should be a FunctionType '
                            f'instance, but got {type(origin_method)}')
        self.origin_method = origin_method

    @staticmethod
    def _check_valid_source(source: str) -> None:
        """Check if the `source` is valid."""
        if not isinstance(source, str):
            raise TypeError(f'source should be a str '
                            f'instance, but got {type(source)}')

        assert len(source.split('.')) > 2, \
            'source must have at least two `.`'

    @property
    def method_name(self):
        """Get the method name according to `method_path`."""
        return self.source.split('.')[-1]

    @property
    def cls_name(self):
        """Get the class name corresponding to this method according to
        `method_path`."""
        return self.source.split('.')[-2]

    @property
    def module_string(self):
        """Get the module name according to `method_path`."""
        return '.'.join(self.source.split('.')[:-2])

    def prepare_from_model(self, model: Optional[nn.Module] = None) -> None:
        """Wrapper the origin source methods.

        The ``model`` is useless in this recorder, just to be consistent with
        other recorders.
        """
        pass

    def method_record_wrapper(self, orgin_method: Callable,
                              data_buffer: List) -> Callable:
        """Save the method's outputs.

        Args:
            origin_method (MethodType): The method whose outputs need to be
                recorded.
            data_buffer (list): A list of data.
        """

        @functools.wraps(orgin_method)
        def wrap_method(*args, **kwargs):
            outputs = orgin_method(*args, **kwargs)
            # assume a func execute N times, there will be N outputs need to
            # save.
            data_buffer.append(outputs)
            return outputs

        return wrap_method

    def __enter__(self):
        """Enter the context manager."""
        super().__enter__()

        imported_cls = self.imported_class
        origin_method = self.origin_method
        # add record wrapper to origin method.
        record_method = self.method_record_wrapper(origin_method,
                                                   self.data_buffer)

        # rewrite the origin method.
        setattr(imported_cls, self.method_name, record_method)

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        super().__exit__(exc_type, exc_value, traceback)

        imported_cls = self.imported_class
        origin_method = self.origin_method

        # restore the origin method
        setattr(imported_cls, self.method_name, origin_method)
