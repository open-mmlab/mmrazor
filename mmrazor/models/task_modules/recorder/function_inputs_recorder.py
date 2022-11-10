# Copyright (c) OpenMMLab. All rights reserved.
import functools
from inspect import signature
from typing import Callable, List

from mmrazor.registry import TASK_UTILS
from .function_outputs_recorder import FunctionOutputsRecorder


@TASK_UTILS.register_module()
class FunctionInputsRecorder(FunctionOutputsRecorder):
    """Recorder for intermediate results which are ``FunctionType``'s inputs.

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
        >>> def toy_func(a, b):
        ...     return a, b
        >>> def execute_toy_func(a, b):
        ...     toy_func(a, b)

        >>> # Below code in main.py
        >>> # Now, we want to get teacher's inputs by recorder.

        >>> from toy_module import execute_toy_func
        >>> r1 = FunctionInputsRecorder('toy_module.toy_func')
        >>> r1.initialize()
        >>> with r1:
        ...     execute_toy_func(1, 2)
        ...     execute_toy_func(1, b=2)
        ...     execute_toy_func(b=2, a=1)

        >>> r1.data_buffer
        [[1, 2], [1, 2], [1, 2]]
    """

    def func_record_wrapper(self, origin_func: Callable,
                            data_buffer: List) -> Callable:
        """Save the function's inputs.

        Args:
            origin_func (FunctionType): The method whose inputs need to be
                recorded.
            data_buffer (list): A list of data.
        """

        func_input_params = signature(origin_func).parameters.keys()

        @functools.wraps(origin_func)
        def wrap_func(*args, **kwargs):
            outputs = origin_func(*args, **kwargs)
            inputs = list(args)
            for keyword in func_input_params:
                if keyword in kwargs:
                    inputs.append(kwargs[keyword])
            # assume a func execute N times, there will be N inputs need to
            # save.
            data_buffer.append(inputs)
            return outputs

        return wrap_func
