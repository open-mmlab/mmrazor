# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch.nn as nn
from mmengine.model import BaseModel

from mmrazor.registry import TASK_UTILS
from mmrazor.utils import get_placeholder
from ...algorithms.base import BaseAlgorithm
from .demo_inputs import (BaseDemoInput, DefaultMMClsDemoInput,
                          DefaultMMDemoInput, DefaultMMDetDemoInput,
                          DefaultMMPoseDemoInput, DefaultMMRotateDemoInput,
                          DefaultMMSegDemoInput, DefaultMMYoloDemoInput)

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

# New
try:
    from mmpose.models import TopdownPoseEstimator
except Exception:
    TopdownPoseEstimator = get_placeholder('mmpose')

default_demo_input_class = OrderedDict([
    (BaseDetector, DefaultMMDetDemoInput),
    (ImageClassifier, DefaultMMClsDemoInput),
    (BaseSegmentor, DefaultMMSegDemoInput),
    (TopdownPoseEstimator, DefaultMMPoseDemoInput),
    (BaseModel, DefaultMMDemoInput),
    (nn.Module, BaseDemoInput),
])

default_demo_input_class_for_scope = {
    'mmcls': DefaultMMClsDemoInput,
    'mmdet': DefaultMMDetDemoInput,
    'mmseg': DefaultMMSegDemoInput,
    'mmrotate': DefaultMMRotateDemoInput,
    'mmyolo': DefaultMMYoloDemoInput,
    'mmpose': DefaultMMPoseDemoInput,
    'torchvision': BaseDemoInput,
}


def get_default_demo_input_class(model, scope):
    """Get demo input generator according to a model and scope."""
    if scope is not None:
        for scope_name, demo_input in default_demo_input_class_for_scope.items(
        ):
            if scope == scope_name:
                return demo_input

    for module_type, demo_input in default_demo_input_class.items(  # noqa
    ):  # noqa
        if isinstance(model, module_type):
            return demo_input
    # default
    return BaseDemoInput


def defaul_demo_inputs(model, input_shape, training=False, scope=None):
    """Get demo input according to a model and scope."""
    if isinstance(model, BaseAlgorithm):
        return defaul_demo_inputs(model.architecture, input_shape, training,
                                  scope)
    else:
        demo_input = get_default_demo_input_class(model, scope)
        return demo_input().get_data(model, input_shape, training)


@TASK_UTILS.register_module()
class DefaultDemoInput(BaseDemoInput):
    """Default demo input generator.

    Args:
        input_shape: default input shape . Defaults to None.
        training (bool, optional): Whether is training mode. Defaults to False.
        scope (str, optional): mm scope name. Defaults to None.
    """

    def __init__(
        self,
        input_shape=None,
        training=False,
        scope: str = None,
        kwargs={},
    ) -> None:

        default_demo_input_class = get_default_demo_input_class(None, scope)
        if input_shape is None:
            input_shape = default_demo_input_class.default_shape
        super().__init__(input_shape, training, kwargs=kwargs)
        self.scope = scope

    def _get_data(self, model, input_shape, training):
        """Helper for get_data, including core logic to generate demo input."""
        return defaul_demo_inputs(model, input_shape, training, self.scope)
