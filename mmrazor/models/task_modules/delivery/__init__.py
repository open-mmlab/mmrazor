# Copyright (c) OpenMMLab. All rights reserved.
from .delivery_manager import DistillDeliveryManager
from .function_outputs_delivery import FunctionOutputsDelivery
from .method_outputs_delivery import MethodOutputsDelivery

__all__ = [
    'FunctionOutputsDelivery', 'MethodOutputsDelivery',
    'DistillDeliveryManager'
]
