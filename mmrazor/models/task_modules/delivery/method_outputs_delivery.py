# Copyright (c) OpenMMLab. All rights reserved.
import functools
from types import FunctionType, ModuleType
from typing import Callable

from mmengine.utils import import_modules_from_strings

from mmrazor.registry import TASK_UTILS
from .distill_delivery import DistillDelivery


@TASK_UTILS.register_module()
class MethodOutputsDelivery(DistillDelivery):
    """Delivery for intermediate results which are ``MethodType``'s outputs.

    Note:
        Different from ``FunctionType``, ``MethodType`` is the type of methods
        of class instances.

    Args:
        method_path (str): The name of the method whose output needs to be
            delivered.
        max_keep_data (int): The length limitation of the queue. Outputs from
            the source model are pushed in the queue in order.

    Examples:
        >>> from mmcls.models.utils import Augments

        >>> augments_cfg = dict(
        ...     type='BatchMixup', alpha=1., num_classes=10, prob=1.0)
        >>> augments = Augments(augments_cfg)
        >>> imgs = torch.randn(2, 3, 32, 32)
        >>> label = torch.randint(0, 10, (2,))

        >>> # Without ``MethodOutputsDeliver``, outputs of the teacher and the
        >>> # student are very likely to be different.
        >>> imgs_tea, label_tea = augments(imgs, label)
        >>> imgs_stu, label_stu = augments(imgs, label)
        >>> torch.equal(label_tea, label_stu)
        False
        >>> torch.equal(imgs_tea, imgs_stu)
        False

        >>> # Suppose we want to deliver outputs from the teacher to
        >>> # the student
        >>> delivery = MethodOutputsDeliver(
        ...     max_keep_data=1,
        ...     method_path='mmcls.models.utils.Augments.__call__')

        >>> delivery.override_data = False
        >>> with delivery:
        ...     imgs_tea, label_tea = augments(imgs, label)

        >>> delivery.override_data = True
        >>> with delivery:
        ...     imgs_stu, label_stu = augments(imgs, label)

        >>> torch.equal(label_tea, label_stu)
        True
        >>> torch.equal(imgs_tea, imgs_stu)
        True
    """

    def __init__(self, method_path: str, max_keep_data: int):
        super().__init__(max_keep_data)

        self._check_valid_path(method_path)
        module_path = self._get_module_path(method_path)
        try:
            module: ModuleType = import_modules_from_strings(module_path)
        except ImportError:
            raise ImportError(f'{module_path} is not imported correctly.')

        cls_name = self._get_cls_name(method_path)
        assert hasattr(module, cls_name), \
            f'{cls_name} is not in {module_path}.'

        imported_cls: type = getattr(module, cls_name)
        if not isinstance(imported_cls, type):
            raise TypeError(f'{cls_name} should be a type '
                            f'instance, but got {type(imported_cls)}')
        self.imported_cls = imported_cls

        method_name = self._get_method_name(method_path)
        assert hasattr(imported_cls, method_name), \
            f'{method_name} is not in {cls_name}.'
        self.method_name = method_name

        origin_method = getattr(imported_cls, method_name)
        # Before instantiation of a class, the type of a method of a class
        # is FunctionType
        if not isinstance(origin_method, FunctionType):
            raise TypeError(f'{method_name} should be a FunctionType  '
                            f'instance, but got {type(origin_method)}')
        self.origin_method = origin_method

    @staticmethod
    def _check_valid_path(method_path: str) -> None:
        """Check if the `method_path` is valid."""
        if not isinstance(method_path, str):
            raise TypeError(f'method_path should be a str instance, '
                            f'but got {type(method_path)}')

        assert len(method_path.split('.')) > 2, \
            'method_path must have at least one `.`'

    @staticmethod
    def _get_method_name(method_path: str) -> str:
        """Get the method name according to `method_path`."""
        return method_path.split('.')[-1]

    @staticmethod
    def _get_cls_name(method_path: str) -> str:
        """Get the class name corresponding to this method according to
        `method_path`."""
        return method_path.split('.')[-2]

    @staticmethod
    def _get_module_path(method_path: str) -> str:
        """Get the module name according to `method_path`."""
        return '.'.join(method_path.split('.')[:-2])

    def __enter__(self) -> None:
        """Enter the context manager.

        Wrap the origin method.
        """
        wrapped_method = self.deliver_wrapper(self.origin_method)
        setattr(self.imported_cls, self.method_name, wrapped_method)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager.

        Reset the origin method.
        """
        setattr(self.imported_cls, self.method_name, self.origin_method)

    def deliver_wrapper(self, origin_method: Callable) -> Callable:
        """Wrap the specific method to make the intermediate results of the
        model can be delivered."""

        @functools.wraps(origin_method)
        def wrap_method(*args, **kwargs):

            if self.override_data:
                assert len(self.data_queue) > 0, 'pop from an empty queue'
                outputs = self.data_queue.popleft()
            else:
                assert len(self.data_queue) < self.data_queue.maxlen,\
                    'push into an full queue'
                outputs = origin_method(*args, **kwargs)
                self.data_queue.append(outputs)
            return outputs

        return wrap_method
