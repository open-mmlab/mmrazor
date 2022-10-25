# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcls.models import BaseClassifier
from mmdet.models.detectors.base import BaseDetector
from mmengine.model import BaseModule

from mmrazor.registry import MODELS
from .custom_tracer import CustomTracer


@MODELS.register_module()
class FXModelWrapper(BaseModule):

    def __init__(self,
                 model,
                 customed_skipped_method=None,
                 mode='loss') -> None:
        super().__init__()
        if isinstance(model, dict):
            self.model = MODELS.build(model)
        elif isinstance(model, nn.Module):
            self.model = model

        assert isinstance(self.model, BaseModule)
        assert mode in ['tensor', 'loss', 'predict']
        self.mode = mode
        self.tracer = CustomTracer(
            customed_skipped_module=customed_skipped_method)
        # self.model = self.tracer.trace(self.model)

    def forward(self, inputs, data_samples):
        if self.mode == 'loss':
            return self.model.loss(inputs, data_samples)
        elif self.mode == 'predict':
            return self.model.predict(inputs, data_samples)
        elif self.mode == 'tensor':
            return self.tensor(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{self.mode}". '
                               'Only supports loss, predict and tensor mode')

    def tensor(self, inputs, data_samples):
        if isinstance(self.model, BaseDetector):
            return self.model._forward(inputs, data_samples)
        elif isinstance(self.model, BaseClassifier):
            raise NotImplementedError('TODO')
