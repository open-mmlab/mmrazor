# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModel

from mmrazor.registry import TASK_UTILS
from mmrazor.utils import get_placeholder
from .demo_inputs import (BaseDemoInput, DefaultMMClsDemoInput,
                          DefaultMMDemoInput, DefaultMMDetDemoInput,
                          DefaultMMRotateDemoInput, DefaultMMSegDemoInput)

try:
    from mmdet.models import BaseDetector
except Exception:
    BaseDetector = get_placeholder('mmdet')

try:
    from mmcls.models import ImageClassifier
except Exception:
    ImageClassifier = get_placeholder('mmcls')

try:
    from mmseg.models import BaseSegmentor
except Exception:
    BaseSegmentor = get_placeholder('mmseg')

default_demo_input_class = {
    BaseDetector: DefaultMMRotateDemoInput,
    ImageClassifier: DefaultMMClsDemoInput,
    BaseSegmentor: DefaultMMSegDemoInput,
    BaseModel: DefaultMMDemoInput,
    nn.Module: BaseDemoInput
}
default_demo_input_class_for_scope = {
    'mmcls': DefaultMMClsDemoInput,
    'mmdet': DefaultMMDetDemoInput,
    'mmseg': DefaultMMSegDemoInput,
    'mmrotate': DefaultMMRotateDemoInput,
    'torchvision': BaseDemoInput,
}


def defaul_demo_inputs(model, input_shape, training=False, scope=None):
    if scope is not None:
        for scope_name, demo_input in default_demo_input_class_for_scope.items(
        ):
            if scope == scope_name:
                return demo_input(input_shape, training).get_data(model)

    for module_type, demo_input in default_demo_input_class.items(  # noqa
    ):  # noqa
        if isinstance(model, module_type):
            return demo_input(input_shape, training).get_data(model)
    # default
    return BaseDemoInput(input_shape, training).get_data(model)


@TASK_UTILS.register_module()
class DefaultDemoInput(BaseDemoInput):

    def __init__(self,
                 input_shape=[1, 3, 224, 224],
                 training=False,
                 scope=None) -> None:
        super().__init__(input_shape, training)
        self.scope = scope

    def _get_data(self, model, input_shape, training):
        return defaul_demo_inputs(model, input_shape, training, self.scope)
