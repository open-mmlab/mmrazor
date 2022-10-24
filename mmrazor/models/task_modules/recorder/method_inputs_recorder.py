# Copyright (c) OpenMMLab. All rights reserved.
import functools
from inspect import signature
from typing import Callable, List

from mmrazor.registry import TASK_UTILS
from .method_outputs_recorder import MethodOutputsRecorder


@TASK_UTILS.register_module()
class MethodInputsRecorder(MethodOutputsRecorder):
    """Recorder for intermediate results which are ``MethodType``'s inputs.

    Note:
        Different from ``FunctionType``, ``MethodType`` is the type of methods
        of class instances.

    Examples:
        >>> # Below code in toy_module.py
        >>> import random
        >>> class Toy():
        ...     def toy_func(self, x, y=0):
        ...         return x + y

        >>> # Below code in main.py
        >>> # Now, we want to get teacher's inputs by recorder.

        >>> from toy_module import Toy
        >>> toy = Toy()
        >>> r1 = MethodInputsRecorder('toy_module.Toy.toy_func')
        >>> r1.initialize()
        >>> with r1:
        ...     _ = toy.toy_func(1, 2)

        >>> r1.data_buffer
        [[1, 2]]
        >>> r1.get_record_data(record_idx=0, data_idx=0)
        1
        >>> r1.get_record_data(record_idx=0, data_idx=1)
        2

        >>> from toy_module import Toy
        >>> toy = Toy()
        >>> r1 = MethodInputsRecorder('toy_module.Toy.toy_func')
        >>> r1.initialize()
        >>> with r1:
        ...     _ = toy.toy_func(1, 2)
        ...     _ = toy.toy_func(y=2, x=1)

        >>> r1.data_buffer
        [[1, 2], [1, 2]]
        >>> r1.get_record_data(record_idx=1, data_idx=0)
        1
        >>> r1.get_record_data(record_idx=1, data_idx=1)
        2
    """

    def method_record_wrapper(self, orgin_method: Callable,
                              data_buffer: List) -> Callable:
        """Save the method's inputs.

        Args:
            origin_method (MethodType): The method whose inputs need to be
                recorded.
            data_buffer (list): A list of data.
        """

        method_input_params = signature(orgin_method).parameters.keys()

        @functools.wraps(orgin_method)
        def wrap_method(*args, **kwargs):
            outputs = orgin_method(*args, **kwargs)
            # the first element of a class method is the class itself
            inputs = list(args[1:])
            for keyword in method_input_params:
                if keyword in kwargs:
                    inputs.append(kwargs[keyword])
            # Assume a func execute N times, there will be N inputs need to
            # save.
            data_buffer.append(inputs)
            return outputs

        return wrap_method
