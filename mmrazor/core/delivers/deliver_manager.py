# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List

from mmrazor.registry import TASK_UTILS


class DistillDeliverManager:
    """Various types delivers' manager. The ``DistillDeliverManager`` is also a
    context manager, managing various types of delivers. When entering the
    ``DistillDeliverManager``, all delivers managed by it will be started.

    Notes:
        DistillDeliver is a context manager used to override function (method)
        outputs from the target model with function (method) outputs from the
        source model.

    Args:
        deliveries (list(dict)): Configs of all deliveries.

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

        >>> distill_deliveries = [
        ...     ConfigDict(type='MethodOutputs', max_keep_data=1,
        ...         method_path='mmcls.models.utils.Augments.__call__')]
        >>> manager = DistillDeliverManager(distill_deliveries)

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

    def __init__(self, deliveries: List) -> None:

        # As there may be several delivers belong to a same deliver type,
        # we use a list to save delivers rather than a dict.
        self.deliveries = list()
        for cfg in deliveries:
            deliver_cfg = copy.deepcopy(cfg)
            deliver_type = cfg.type
            deliver_type = deliver_type + 'Deliver'
            deliver_cfg.type = deliver_type
            self.deliveries.append(TASK_UTILS.build(deliver_cfg))

        self._override_data = False

    @property
    def override_data(self):
        """bool: indicate whether to override the data with the recorded data.
        """
        return self._override_data

    @override_data.setter
    def override_data(self, override):
        """Set the override_data property to all the delivers.

        If the `override_data` of a deliver is False, the deliver will record
        and keep the origin data. If the current_mode of a deliver is True, the
        deliver will override the origin data with the recorded data.
        """
        self._override_data = override
        for deliver in self.deliveries:
            deliver.override_data = override

    def __enter__(self) -> None:
        """Enter the context manager."""
        for deliver in self.deliveries:
            deliver.__enter__()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager."""
        for deliver in self.deliveries:
            deliver.__exit__(exc_type, exc_value, traceback)
