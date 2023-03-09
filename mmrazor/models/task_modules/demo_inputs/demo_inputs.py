# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BaseDemoInput():
    """Base demo input generator.

    Args:
        input_shape: Default input shape. Defaults to default_shape.
        training (bool, optional): Default training mode. Defaults to None.
        kwargs (dict): Other keyword args to update the generated inputs.
    """
    default_shape = (1, 3, 224, 224)

    def __init__(self,
                 input_shape=default_shape,
                 training=None,
                 kwargs={}) -> None:

        self.input_shape = input_shape
        self.training = training
        self.kwargs = kwargs

    def get_data(self, model, input_shape=None, training=None):
        """Api to generate demo input."""
        if input_shape is None:
            input_shape = self.input_shape
        if training is None:
            training = self.training

        data = self._get_data(model, input_shape, training)
        if isinstance(data, dict):
            data.update(self.kwargs)
        return data

    def _get_data(self, model, input_shape, training):
        """Helper for get_data, including core logic to generate demo input."""
        return torch.rand(input_shape)

    def __call__(self,
                 model=None,
                 input_shape=[1, 3, 224, 224],
                 training=False):
        return self.get_data(model, input_shape, training)


@TASK_UTILS.register_module()
class DefaultMMDemoInput(BaseDemoInput):
    """Default demo input generator for openmmable models."""

    def _get_data(self, model, input_shape=None, training=None):
        """Helper for get_data, including core logic to generate demo input."""

        data = self._get_mm_data(model, input_shape, training)
        data['mode'] = 'tensor'
        return data

    def _get_mm_data(self, model, input_shape, training=False):
        data = {'inputs': torch.rand(input_shape), 'data_samples': None}
        data = model.data_preprocessor(data, training)
        return data


@TASK_UTILS.register_module()
class DefaultMMClsDemoInput(DefaultMMDemoInput):
    """Default demo input generator for mmcls models."""

    def _get_mm_data(self, model, input_shape, training=False):
        """Helper for get_data, including core logic to generate demo input."""
        from mmcls.structures import ClsDataSample
        x = torch.rand(input_shape)
        mm_inputs = {
            'inputs':
            x,
            'data_samples': [
                ClsDataSample(
                    metainfo=dict(img_shape=input_shape[i],
                                  num_classes=1000)).set_gt_label(1)
                for i in range(input_shape[0])
            ],
        }
        mm_inputs = model.data_preprocessor(mm_inputs, training)
        return mm_inputs


@TASK_UTILS.register_module()
class DefaultMMDetDemoInput(DefaultMMDemoInput):
    """Default demo input generator for mmdet models."""

    def _get_mm_data(self, model, input_shape, training=False):
        """Helper for get_data, including core logic to generate demo input."""
        from mmdet.models import BaseDetector
        from mmdet.testing._utils import demo_mm_inputs
        assert isinstance(model, BaseDetector), f'{type(model)}'

        data = demo_mm_inputs(1, [input_shape[1:]], with_mask=True)
        data = model.data_preprocessor(data, training)
        return data


@TASK_UTILS.register_module()
class DefaultMMSegDemoInput(DefaultMMDemoInput):
    """Default demo input generator for mmseg models."""

    def _get_mm_data(self, model, input_shape, training=False):
        """Helper for get_data, including core logic to generate demo input."""
        from mmseg.models import BaseSegmentor
        assert isinstance(model, BaseSegmentor)
        from .mmseg_demo_input import demo_mmseg_inputs
        data = demo_mmseg_inputs(model, input_shape)
        return data


@TASK_UTILS.register_module()
class DefaultMMRotateDemoInput(DefaultMMDemoInput):
    """Default demo input generator for mmrotate models."""

    def _get_mm_data(self, model, input_shape, training=False):
        """Helper for get_data, including core logic to generate demo input."""
        from mmrotate.testing._utils import demo_mm_inputs

        data = demo_mm_inputs(1, [input_shape[1:]], use_box_type=True)
        data = model.data_preprocessor(data, training)
        return data


@TASK_UTILS.register_module()
class DefaultMMYoloDemoInput(DefaultMMDetDemoInput):
    """Default demo input generator for mmyolo models."""

    default_shape = (1, 3, 125, 320)


@TASK_UTILS.register_module()
class DefaultMMPoseDemoInput(DefaultMMDemoInput):
    """Default demo input generator for mmpose models."""

    def _get_mm_data(self, model, input_shape, training=False):
        from mmpose.models import TopdownPoseEstimator

        from .mmpose_demo_input import demo_mmpose_inputs
        assert isinstance(model, TopdownPoseEstimator), f'{type(model)}'

        data = demo_mmpose_inputs(model, input_shape)
        return data
