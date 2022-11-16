# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BaseDemoInput():
    default_shape = (1, 3, 224, 224)

    def __init__(self, input_shape=default_shape, training=None) -> None:
        self.input_shape = input_shape
        self.training = training

    def get_data(self, model, input_shape=None, training=None):
        if input_shape is None:
            input_shape = self.input_shape
        if training is None:
            training = self.training

        return self._get_data(model, input_shape, training)

    def _get_data(self, model, input_shape, training):
        return torch.rand(input_shape)

    def __call__(self,
                 model=None,
                 input_shape=[1, 3, 224, 224],
                 training=False):
        return self.get_data(model, input_shape, training)


@TASK_UTILS.register_module()
class DefaultMMDemoInput(BaseDemoInput):

    def _get_data(self, model, input_shape=None, training=None):

        data = self._get_mm_data(model, input_shape, training)
        data['mode'] = 'tensor'
        return data

    def _get_mm_data(self, model, input_shape, training=False):
        return {'inputs': torch.rand(input_shape), 'data_samples': None}


@TASK_UTILS.register_module()
class DefaultMMClsDemoInput(DefaultMMDemoInput):

    def _get_mm_data(self, model, input_shape, training=False):
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

    def _get_mm_data(self, model, input_shape, training=False):
        from mmdet.models import BaseDetector
        from mmdet.testing._utils import demo_mm_inputs
        assert isinstance(model, BaseDetector)

        data = demo_mm_inputs(1, [input_shape[1:]])
        data = model.data_preprocessor(data, training)
        return data


@TASK_UTILS.register_module()
class DefaultMMSegDemoInput(DefaultMMDemoInput):

    def _get_mm_data(self, model, input_shape, training=False):
        from mmseg.models import BaseSegmentor
        assert isinstance(model, BaseSegmentor)
        from .mmseg_demo_input import demo_mmseg_inputs
        data = demo_mmseg_inputs(model, input_shape)
        return data


@TASK_UTILS.register_module()
class DefaultMMRotateDemoInput(DefaultMMDemoInput):

    def _get_mm_data(self, model, input_shape, training=False):
        from mmrotate.testing._utils import demo_mm_inputs

        data = demo_mm_inputs(1, [input_shape[1:]], use_box_type=True)
        data = model.data_preprocessor(data, training)
        return data


@TASK_UTILS.register_module()
class DefaultMMYoloDemoInput(DefaultMMDetDemoInput):
    default_shape = (1, 3, 125, 320)
