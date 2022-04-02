# Copyright (c) OpenMMLab. All rights reserved.
from .deliver_manager import DistillDeliverManager
from .function_outputs_deliver import FunctionOutputsDeliver
from .method_outputs_deliver import MethodOutputsDeliver

__all__ = [
    'FunctionOutputsDeliver', 'MethodOutputsDeliver', 'DistillDeliverManager'
]
