# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

from mmrazor.registry import TASK_UTILS


class BasePredictor():
    """Base predictor."""

    def __init__(self, handler_cfg: dict):
        """init."""
        self.handler_cfg = handler_cfg
        self.handler = TASK_UTILS.build(handler_cfg)

    @abstractmethod
    def predict(self, model, predict_args):
        """predict result."""
        pass
