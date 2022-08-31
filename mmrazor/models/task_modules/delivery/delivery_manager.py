# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

from mmrazor.registry import TASK_UTILS
from .distill_delivery import DistillDelivery

SUPPORT_DELIVERIES = ['FunctionOutputs', 'MethodOutputs']


class DistillDeliveryManager:
    """Various types deliveries' manager. The ``DistillDeliveryManager`` is
    also a context manager, managing various types of deliveries.

    When entering the ``DistillDeliveryManager``, all deliveries managed by it
    will be started.

    Notes:
        DistillDelivery is a context manager used to override function(method)
        outputs during teacher(student) forward.

    Args:
        deliveries (dict): Configs of all deliveries.

    Examples:
        >>> from mmcls.models.utils import Augments

        >>> augments_cfg = dict(
        ...     type='BatchMixup', alpha=1., num_classes=10, prob=1.0)
        >>> augments = Augments(augments_cfg)
        >>> imgs = torch.randn(2, 3, 32, 32)
        >>> label = torch.randint(0, 10, (2,))

        >>> # Without ``MethodOutputsDelivery``, outputs of the teacher and
        >>> # the student are different.
        >>> imgs_tea, label_tea = augments(imgs, label)
        >>> imgs_stu, label_stu = augments(imgs, label)
        >>> torch.equal(label_tea, label_stu)
        False
        >>> torch.equal(imgs_tea, imgs_stu)
        False

        >>> distill_deliveries = ConfigDict(
        ...     aug=dict(type='MethodOutputs', max_keep_data=1,
        ...             method_path='mmcls.models.utils.Augments.__call__'))
        >>> manager = DistillDeliveryManager(distill_deliveries)

        >>> manager.override_data = False
        >>> with manager:
        ...     imgs_tea, label_tea = augments(imgs, label)

        >>> manager.override_data = True
        >>> with manager:
        ...     imgs_stu, label_stu = augments(imgs, label)

        >>> torch.equal(label_tea, label_stu)
        True
        >>> torch.equal(imgs_tea, imgs_stu)
        True
    """

    def __init__(self, deliveries: Optional[Dict[str, Dict]] = None) -> None:

        self._deliveries: Dict[str, DistillDelivery] = dict()
        if deliveries:
            for delivery_name, delivery_cfg in deliveries.items():
                delivery_cfg_ = copy.deepcopy(delivery_cfg)
                delivery_type_ = delivery_cfg_.get('type', '')
                assert isinstance(delivery_type_, str)
                assert delivery_type_ in SUPPORT_DELIVERIES

                delivery_type_ = delivery_type_ + 'Delivery'
                delivery_cfg_.update(dict(type=delivery_type_))

                delivery = TASK_UTILS.build(delivery_cfg_)
                self.deliveries[delivery_name] = delivery

        self._override_data = False

    @property
    def deliveries(self) -> Dict[str, DistillDelivery]:
        """dict: all deliveries."""
        return self._deliveries

    @property
    def override_data(self) -> bool:
        """bool: indicate whether to override the data with the recorded data.
        """
        return self._override_data

    @override_data.setter
    def override_data(self, override: bool) -> None:
        """Set the override_data property to all the deliveries.

        If the `override_data` of a delivery is False, the delivery will
        record the origin data.

        If the `override_data` of a delivery is True, the delivery will
        override the origin data with the recorded data.
        """
        self._override_data = override
        for delivery in self.deliveries.values():
            delivery.override_data = override

    def __enter__(self) -> None:
        """Enter the context manager."""
        for delivery in self.deliveries.values():
            delivery.__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager."""
        for delivery in self.deliveries.values():
            delivery.__exit__(exc_type, exc_value, traceback)
