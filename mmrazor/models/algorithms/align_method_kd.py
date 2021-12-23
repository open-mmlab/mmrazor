# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.builder import ALGORITHMS
from .general_distill import GeneralDistill


@ALGORITHMS.register_module()
class AlignMethodDistill(GeneralDistill):

    def __init__(self, **kwargs):
        super(AlignMethodDistill, self).__init__(**kwargs)

    def train_step(self, data, optimizer):

        with self.distiller.context_manager:
            outputs = super().train_step(data, optimizer)
        return outputs
