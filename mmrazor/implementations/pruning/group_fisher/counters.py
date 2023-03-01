# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.task_modules.estimators.counters.op_counters.dynamic_op_counters import (  # noqa
    DynamicConv2dCounter, DynamicLinearCounter)
from mmrazor.registry import TASK_UTILS


@TASK_UTILS.register_module()
class GroupFisherConv2dCounter(DynamicConv2dCounter):
    """Counter of GroupFisherConv2d."""
    pass


@TASK_UTILS.register_module()
class GroupFisherLinearCounter(DynamicLinearCounter):
    """Counter of GroupFisherLinear."""
    pass
