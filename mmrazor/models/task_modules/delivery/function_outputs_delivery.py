# Copyright (c) OpenMMLab. All rights reserved.
import functools
from types import FunctionType
from typing import Callable

from mmengine.utils import import_modules_from_strings

from mmrazor.registry import TASK_UTILS
from .distill_delivery import DistillDelivery


@TASK_UTILS.register_module()
class FunctionOutputsDelivery(DistillDelivery):
    """Delivery for intermediate results which are ``FunctionType``'s outputs.

    Args:
        func_path (str): The name of the function whose output needs to be
            delivered.
        max_keep_data (int): The length limitation of the queue. Outputs from
            the source model are pushed in the queue in order.

    Notes:
        The form of `func_path` needs special attention. For example,
        `anchor_inside_flags` is a function in mmdetection to check whether the
        anchors are inside the border. This function is in
        `mmdet/core/anchor/utils.py` and used in
        `mmdet/models/dense_heads/anchor_head`. Then the `func_path` should be
        `mmdet.models.dense_heads.anchor_head.anchor_inside_flags` but not
        `mmdet.core.anchor.utils.anchor_inside_flags`.

    Examples:
        >>> # Below code in toy_module.py
        >>> import random
        >>> def toy_func():
        >>>     return random.randint(0, 1000)

        >>> # Below code in main.py
        >>> # Teacher and student both will execute toy_func.
        >>> # Now, we want to deliver outputs from the teacher to
        >>> # the student
        >>> import toy_module
        >>> delivery = FunctionOutputsDeliver(
        ...     max_keep_data=1, func_path='toy_module.toy_func')

        >>> delivery.override_data = False
        >>> with delivery:
        ...     output_teacher = toy_module.toy_func()

        >>> delivery.override_data = True
        >>> with delivery:
        ...     output_student = toy_module.toy_func()

        >>> output_teacher == output_student
        True

        >>> # If a function (method) is executed more than once during the
        >>> # forward of the source model, all the outputs of this function
        >>> # (method) will be used to override function (method) outputs from
        >>> # the target model.
        >>> delivery = FunctionOutputsDeliver(
        ...     max_keep_data=2, func_path='toy_module.toy_func')

        >>> delivery.override_data = False
        >>> with delivery:
        ...     output1_tea = toy_module.toy_func()
        ...     output2_tea = toy_module.toy_func()

        >>> delivery.override_data = True
        >>> with delivery:
        ...     output1_stu = toy_module.toy_func()
        ...     output2_stu = toy_module.toy_func()

        >>> output1_stu == output1_tea and output2_stu == output2_tea
        True
    """

    def __init__(self, func_path: str, max_keep_data: int):
        super().__init__(max_keep_data)

        self._check_valid_path(func_path)
        self.func_path = func_path

    @staticmethod
    def _check_valid_path(func_path: str) -> None:
        """Check if the `func_path` is valid."""
        if not isinstance(func_path, str):
            raise TypeError(f'func_path should be a FunctionType '
                            f'instance, but got {type(func_path)}')

        assert len(func_path.split('.')) > 1, \
            'func_path must have at least one `.`'

    @staticmethod
    def _get_func_name(func_path: str) -> str:
        """Get the function name according to `func_path`."""
        return func_path.split('.')[-1]

    @staticmethod
    def _get_module_path(func_path: str) -> str:
        """Get the module name according to `func_path`."""
        return '.'.join(func_path.split('.')[:-1])

    def __enter__(self) -> None:
        """Enter the context manager.

        Wrap the origin function.
        """
        module_path = self._get_module_path(self.func_path)
        try:
            module = import_modules_from_strings(module_path)
        except ImportError:
            raise ImportError(f'{module_path} is not imported correctly.')
        self.module = module

        func_name = self._get_func_name(self.func_path)
        assert hasattr(module, func_name), \
            f'{func_name} is not in {module_path}.'
        self.func_name = func_name

        origin_func = getattr(module, func_name)
        if not isinstance(origin_func, FunctionType):
            raise TypeError(f'{func_name} should be a FunctionType '
                            f'instance, but got {type(origin_func)}')
        self.origin_func = origin_func

        wrapped_func = self.deliver_wrapper(self.origin_func)
        setattr(self.module, self.func_name, wrapped_func)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager.

        Reset the origin function.
        """
        setattr(self.module, self.func_name, self.origin_func)

        # self.module and self.origin_func can not be pickled.
        # Delete these two attributes to avoid errors when ema model is used.
        del self.module
        del self.origin_func

    def deliver_wrapper(self, origin_func: Callable) -> Callable:
        """Wrap the specific function to make the intermediate results of the
        model can be delivered."""

        @functools.wraps(origin_func)
        def wrap_func(*args, **kwargs):

            if self.override_data:
                assert len(self.data_queue) > 0, 'pop from an empty queue'
                outputs = self.data_queue.popleft()
            else:
                assert len(self.data_queue) < self.data_queue.maxlen,\
                    'push into an full queue'
                outputs = origin_func(*args, **kwargs)
                self.data_queue.append(outputs)
            return outputs

        return wrap_func
