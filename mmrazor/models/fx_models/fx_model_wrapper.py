# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcls.models import BaseClassifier
from mmdet.models.detectors.base import BaseDetector
from mmengine.model import BaseModel

from mmrazor.registry import MODELS


@MODELS.register_module()
class FXModelWrapper(BaseModel):

    def __init__(self,
                 model,
                 customed_skipped_method=None,
                 data_preprocessor=None,
                 mode='loss') -> None:
        super().__init__()
        if isinstance(model, dict):
            if data_preprocessor is not None:
                model.setdefault('data_preprocessor', data_preprocessor)
            self.model = MODELS.build(model)
        elif isinstance(model, nn.Module):
            self.model = model

        assert isinstance(self.model, BaseModel)
        assert mode in ['tensor', 'loss', 'predict']
        self.mode = mode
        self.customed_skipped_method = customed_skipped_method

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
            raise NotImplementedError(
                'TODO: integrate mmcls `tensor` function')

    @property
    def data_preprocessor(self):
        return self.model.data_preprocessor

    def train_step(self):
        pass
