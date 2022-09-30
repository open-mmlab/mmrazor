# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.registry import TASK_UTILS
from .base_predictor import BasePredictor


@TASK_UTILS.register_module()
class ZeroShotPredictor(BasePredictor):

    def __init__(self):
        super().__init__()
