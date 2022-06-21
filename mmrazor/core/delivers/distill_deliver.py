# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from queue import Queue
from typing import Callable


class DistillDeliver(metaclass=ABCMeta):
    """Base class for delivers for distillation.

    DistillDeliver is a context manager used to override function (method)
    outputs from the target model with function (method) outputs from the
    source model.
    In MMRazor, there will be different types of delivers to deliver different
    types of data. They can be used in combination with the
    ``DistillDeliverManager``.

    Notes:
        If a function (method) is executed more than once during the forward
        of the source model, all the outputs of this function (method) will be
        used to override function (method) outputs from the target model.

    TODO:
        Support overriding some of the outputs of a function (method)

    Args:
        max_keep_data (int): The length limitation of the queue. If a function
            (method) is executed more than once during the forward of the
            target model, function (method) outputs from the source model are
            pushed into the queue in order. Default to 1.
    """

    def __init__(self, max_keep_data: int = 1):

        self._override_data = False
        self.data_queue: Queue = Queue(maxsize=max_keep_data)
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
