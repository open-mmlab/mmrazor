import copy
import functools
import time
from abc import ABCMeta, abstractmethod
from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector)
from mmdet.utils import register_all_modules
from mmengine.registry import TASK_UTILS, VISUALIZERS
from mmengine.utils import import_modules_from_strings
from mmengine.visualization import Visualizer
from torch import nn


class BaseRecorder(metaclass=ABCMeta):

    def __init__(self,
                 source: str,
                 record_idx: int = 0,
                 data_idx: Optional[int] = None) -> None:

        self._source = source
        # Intermediate results are recorded in dictionary format according
        # to the data source.
        # One data source may generate multiple records, which need to be
        # recorded through list.
        self._data_buffer: List = list()
        # Before using the recorder for the first time, it needs to be
        # initialized.
        self._initialized = False
        self.record_idx = record_idx
        self.data_idx = data_idx

    @property
    def source(self) -> str:
        """str: source of recorded data."""
        return self._source

    @property
    def data_buffer(self) -> List:
        """list: data buffer."""
        return self._data_buffer

    @abstractmethod
    def prepare_from_model(self, model: Optional[nn.Module] = None) -> None:
        """Make the intermediate results of the model can be record."""

    def initialize(self, model: Optional[nn.Module] = None) -> None:
        """Init the recorder.

        Args:
            model (nn.Module): The model which need to record intermediate
                results.
        """
        self.prepare_from_model(model)
        self._initialized = True

    def get_record_data(self) -> Any:
        """Get data from ``data_buffer``.

        Returns:
            Any: The type of the return value is undefined, and different
                source data may have different types.
        """
        assert self.record_idx < len(self._data_buffer), \
            'record_idx is illegal. The length of data_buffer is ' \
            f'{len(self._data_buffer)}, but record_idx is ' \
            f'{self.record_idx}.'

        record = self._data_buffer[self.record_idx]

        if self.data_idx is None:
            target_data = record
        else:
            if isinstance(record, (list, tuple)):
                assert self.data_idx < len(record), \
                    'data_idx is illegal. The length of record is ' \
                    f'{len(record)}, but data_idx is {self.data_idx}.'
                target_data = record[self.data_idx]
            else:
                raise TypeError('When data_idx is not None, record should be '
                                'a list or tuple instance, but got '
                                f'{type(record)}.')
        return target_data

    def reset_data_buffer(self) -> None:
        """Clear data in data_buffer."""
        self._data_buffer = list()


@TASK_UTILS.register_module()
class ModuleOutputsRecorder(BaseRecorder):

    def prepare_from_model(self, model: Optional[nn.Module] = None) -> None:
        """Register Pytorch forward hook to corresponding module."""

        assert model is not None, 'model can not be None.'
        assert self.source in dict(
            model.named_modules()), f'"{self.source}" is not in the model.'
        module = dict(model.named_modules())[self.source]
        module.register_forward_hook(self.forward_hook)

    def forward_hook(self, module: nn.Module, inputs: Tuple,
                     outputs: Any) -> None:
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs : The output of the module.
        """
        self.data_buffer.append(outputs)


@TASK_UTILS.register_module()
class ModuleInputsRecorder(ModuleOutputsRecorder):
    """Recorder for intermediate results which are Pytorch moudle's inputs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_hook(self, module: nn.Module, inputs: Tuple,
                     outputs: Any) -> None:
        self.data_buffer.append(inputs)


@TASK_UTILS.register_module()
class FunctionOutputsRecorder(BaseRecorder):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._check_valid_source(self.source)

        # import the function corrosponding module
        try:
            mod = import_modules_from_strings(self.module_string)
        except ImportError:
            raise ImportError(
                f'{self.module_string} is not imported correctly.')

        self.imported_module: ModuleType = mod

        assert hasattr(mod, self.func_name), \
            f'{self.func_name} is not in {self.module_string}.'

        origin_func = getattr(mod, self.func_name)
        if not isinstance(origin_func, FunctionType):
            raise TypeError(f'{self.func_name} should be a FunctionType '
                            f'instance, but got {type(origin_func)}')
        self.origin_func: Callable = origin_func

    @staticmethod
    def _check_valid_source(source):
        """Check if the source's format is valid."""
        if not isinstance(source, str):
            raise TypeError(f'source should be a str '
                            f'instance, but got {type(source)}')

        assert len(source.split('.')) > 1, \
            'source must have at least one `.`'

    @property
    def func_name(self):
        """Get the function name according to `func_path`."""
        return self.source.split('.')[-1]

    @property
    def module_string(self):
        """Get the module name according to `func_path`."""
        return '.'.join(self.source.split('.')[:-1])

    def prepare_from_model(self, model: Optional[nn.Module] = None) -> None:
        """The `model` is useless in `FunctionOutputsRecorder`."""
        mod = self.imported_module
        origin_func = self.origin_func
        # add record wrapper to origin function.
        record_func = self.func_record_wrapper(origin_func, self.data_buffer)

        assert hasattr(mod, self.func_name), \
            f'{self.func_name} is not in {self.module_string}.'

        # rewrite the origin function
        setattr(mod, self.func_name, record_func)

    def func_record_wrapper(self, origin_func: Callable,
                            data_buffer: List) -> Callable:
        """Save the function's outputs.

        Args:
            origin_func (FunctionType): The method whose outputs need to be
                recorded.
            buffer_key (str): The key of the function's outputs saved in
                ``data_buffer``.
        """

        @functools.wraps(origin_func)
        def wrap_func(*args, **kwargs):
            outputs = origin_func(*args, **kwargs)
            # assume a func execute N times, there will be N outputs need to
            # save.
            data_buffer.append(outputs)
            return outputs

        return wrap_func


class RecorderManager:

    def __init__(self, recorders: Optional[Dict] = None) -> None:

        self._recorders: Dict[str, BaseRecorder] = dict()
        if recorders:
            for name, cfg in recorders.items():
                recorder_cfg = copy.deepcopy(cfg)
                recorder_type = cfg['type']
                recorder_type_ = recorder_type + 'Recorder'

                recorder_cfg['type'] = recorder_type_
                recorder = TASK_UTILS.build(recorder_cfg)

                self._recorders[name] = recorder

    @property
    def recorders(self) -> Dict[str, BaseRecorder]:
        """dict: all recorders."""
        return self._recorders

    def get_recorder(self, recorder: str) -> BaseRecorder:
        """Get the corresponding recorder according to the name."""
        return self.recorders[recorder]

    def initialize(self, model: nn.Module):
        """Init all recorders.

        Args:
            model (nn.Module): The model which need to record intermediate
                results.
        """
        for recorder in self.recorders.values():
            recorder.initialize(model)


class VisualizerWrapper:

    def __init__(self,
                 visualizer: Visualizer,
                 recorders: Optional[Dict] = None):
        self.visualizer = visualizer
        self.recorder_manager = RecorderManager(recorders)

    def initialize(self, model: nn.Module):
        self.recorder_manager.initialize(model)

    @staticmethod
    def _resize(ori_shape, img_shape, feat):
        ori_h, ori_w = ori_shape
        img_h, img_w = img_shape
        feat_h, feat_w = feat.shape[-2:]
        h, w = round(ori_h * feat_h / img_h), round(ori_w * feat_w / img_w)
        feat = feat[..., :h, :w]
        return F.interpolate(feat, ori_shape, mode='bilinear')

    def draw(self,
             ori_shape,
             img_shape,
             overlaid_image: Optional[np.ndarray] = None,
             channel_reduction: Optional[str] = 'squeeze_mean',
             topk: int = 20,
             arrangement: Tuple[int, int] = (4, 5),
             resize_shape: Optional[tuple] = None,
             alpha: float = 0.5,
             out_file: Optional[str] = None):
        for name, recorder in self.recorder_manager.recorders.items():
            feats = recorder.get_record_data()
            if isinstance(feats, torch.Tensor):
                feats = (feats, )
            for i, feat in enumerate(feats):
                feat = self._resize(ori_shape, img_shape, feat)
                feat = feat[0]  # nchw->chw
                drawn_img = self.visualizer.draw_featmap(
                    feat, overlaid_image, channel_reduction, topk, arrangement,
                    resize_shape, alpha)
                self.visualizer.add_datasample(
                    f'{name}_{i}',
                    drawn_img,
                    show=out_file is None,
                    out_file=out_file)
            # data_buffer must be reset here
            recorder.reset_data_buffer()


if __name__ == '__main__':
    register_all_modules()
    checkpoint_file = r'G:\projects\research\checkpoint\yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'
    cfg = r'G:\projects\openmmlab\mmdetection\configs\yolox\yolox_x_8xb8-300e_coco.py'
    model = init_detector(cfg, checkpoint_file, device='cpu')
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta
    visualizer_wrapper = VisualizerWrapper(
        visualizer,
        recorders=dict(
            neck=dict(type='ModuleOutputs', source='neck', data_idx=0)))
    visualizer_wrapper.initialize(model)
    img_path = r'G:\projects\openmmlab\mmdetection\demo\demo.jpg'
    _ = inference_detector(model, img_path)
    overlaid_image = mmcv.imread(img_path, channel_order='rgb')
    visualizer_wrapper.draw((427, 640), (640, 640),
                            overlaid_image=overlaid_image)
