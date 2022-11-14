# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BaseDemoInput():

    def __init__(self, input_shape, training=False) -> None:
        self.input_shape = input_shape
        self.training = training

    def get_data(self, model, input_shape=None, training=None):
        if input_shape is None:
            input_shape = self.input_shape
        if training is None:
            training = self.training

        return torch.rand(self.input_shape)


@TASK_UTILS.register_module()
class DefaultMMDemoInput(BaseDemoInput):

    def get_data(self, model, input_shape=None, training=None):
        if input_shape is None:
            input_shape = self.input_shape
        if training is None:
            training = self.training

        data = self.get_mm_data(model, input_shape, training)
        data['mode'] = 'tensor'
        return data

    def get_mm_data(self, model, input_shape, training=False):
        return {'inputs': torch.rand(input_shape), 'data_samples': None}


@TASK_UTILS.register_module()
class DefaultMMClsDemoInput(DefaultMMDemoInput):

    def get_mm_data(self, model, input_shape, training=False):
        from mmcls.structures import ClsDataSample
        x = torch.rand(input_shape)
        mm_inputs = {
            'inputs':
            x,
            'data_samples':
            [ClsDataSample().set_gt_label(1) for _ in range(input_shape[0])],
        }
        return mm_inputs


@TASK_UTILS.register_module()
class DefaultMMDetDemoInput(DefaultMMDemoInput):

    def get_mm_data(self, model, input_shape, training=False):
        from mmdet.models import BaseDetector
        from mmdet.testing._utils import demo_mm_inputs
        assert isinstance(model, BaseDetector)

        data = demo_mm_inputs(1, [input_shape[1:]])
        data = model.data_preprocessor(data, False)
        return data


@TASK_UTILS.register_module()
class DefaultMMSegDemoInput(DefaultMMDemoInput):

    def get_mm_data(self, model, input_shape, training=False):
        from mmseg.models import BaseSegmentor
        assert isinstance(model, BaseSegmentor)
        from .mmseg_demo_input import demo_mmseg_inputs
        data = demo_mmseg_inputs(model, input_shape)
        return data
