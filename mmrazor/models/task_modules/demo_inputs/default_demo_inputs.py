# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModel

from mmrazor.registry import TASK_UTILS
from mmrazor.utils import get_placeholder
from .demo_inputs import (BaseDemoInput, DefaultMMClsDemoInput,
                          DefaultMMDemoInput, DefaultMMDetDemoInput,
                          DefaultMMSegDemoInput)

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

default_concrete_args_fun = {
    BaseDetector: DefaultMMDetDemoInput,
    ImageClassifier: DefaultMMClsDemoInput,
    BaseSegmentor: DefaultMMSegDemoInput,
    BaseModel: DefaultMMDemoInput,
    nn.Module: BaseDemoInput
}


def defaul_demo_inputs(model, input_shape, training=False):
    for module_type, demo_input_class in default_concrete_args_fun.items(  # noqa
    ):  # noqa
        if isinstance(model, module_type):
            return demo_input_class(input_shape, training).get_data(model)
    # default
    return BaseDemoInput(input_shape, training).get_data(model)


@TASK_UTILS.register_module()
class DefaultDemoInput(BaseDemoInput):

    def get_data(self, model):
        return defaul_demo_inputs(model, self.input_shape, self.training)
