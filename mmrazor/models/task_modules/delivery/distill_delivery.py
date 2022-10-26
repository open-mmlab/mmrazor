# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import deque
from typing import Callable


# TODO: Support overriding part of the outputs of a function or method
class DistillDelivery(metaclass=ABCMeta):
    """Base class for deliveries for distillation.

    DistillDelivery is a context manager used to override function(method)
    outputs during teacher(student) forward.

    A delivery can only handle one function or method. Some algorithms may use
    multiple deliveries, which can be managed uniformly using
    ``DistillDeliverManager``.

    Args:
        max_keep_data (int): The length limitation of the queue, should be
            larger than the execute times of the function or method. Defaults
            to 1.

    Notes:
        If a function (method) is executed more than once during the forward
        of the source model, all the outputs of this function (method) will be
        used to override function (method) outputs from the target model.

        If a function or method is executed more than once during the forward
        of the target model, its' outputs from the source model are pushed
        into the queue in order.
    """

    def __init__(self, max_keep_data: int = 1) -> None:

        self._override_data = False
        self.data_queue: deque = deque([], maxlen=max_keep_data)
        self.max_keep_data = max_keep_data

    @property
    def override_data(self) -> bool:
        """bool: indicate whether to override the data with the recorded data.
        """
        return self._override_data

    @override_data.setter
    def override_data(self, override: bool) -> None:
        """Set the override_data property to this delivery.

        If the `override_data` of a deliver is False, the deliver will record
        and keep the origin data. If the current_mode of a deliver is True, the
        deliver will override the origin data with the recorded data.
        """
        self._override_data = override

    @abstractmethod
    def deliver_wrapper(self, origin: Callable) -> Callable:
        """Wrap the specific object to make the intermediate results of the
        model can be delivered."""

    @abstractmethod
    def __enter__(self) -> None:
        """Enter the context manager."""

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager."""
