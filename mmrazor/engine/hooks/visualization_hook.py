# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence, List, Tuple
import torch

import mmcv
from mmcv.transforms import Compose
from mmengine.fileio import FileClient
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmrazor.registry import HOOKS
from mmrazor.visualization import RazorLocalVisualizer
from mmrazor.models.task_modules import RecorderManager


def norm(feat):
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    centered = (feat - mean) / std
    centered = centered.reshape(C, N, H, W).permute(1, 0, 2, 3)
    return centered


@HOOKS.register_module()
class RazorVisualizationHook(Hook):
    """Detection Visualization Hook. Used to visualize validation and testing
    process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        test_out_dir (str, optional): directory where painted images
            will be saved in testing process.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 recorders: dict,
                 mappings: dict,
                 data_idx: Optional[int, List] = 0,
                 interval: int = 1,
                 show: bool = False,
                 wait_time: float = 0.1,
                 out_dir: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk'),
                 is_overlaid: bool = True,
                 channel_reduction: Optional[str] = 'pixel_wise_max',
                 topk: int = 20,
                 arrangement: Tuple[int, int] = (4, 5),
                 resize_shape: Optional[tuple] = None,
                 alpha: float = 0.5,
                 use_norm: bool = True):
        self._visualizer: RazorLocalVisualizer = RazorLocalVisualizer.get_current_instance()
        if isinstance(data_idx, int):
            data_idx = [data_idx]
        self.data_idx = data_idx
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.out_dir = out_dir
        self._test_index = 0
        self.interval = interval

        self.is_overlaid = is_overlaid
        self.channel_reduction = channel_reduction
        if channel_reduction is not None:
            assert channel_reduction in [
                'squeeze_mean', 'select_max', 'pixel_wise_max'], \
                f'Mode only support "squeeze_mean", "select_max", "pixel_wise_max", ' \
                f'but got {channel_reduction}'
        self.topk = topk
        self.arrangement = arrangement
        self.resize_shape = resize_shape
        self.alpha = alpha
        self.use_norm = use_norm

        self.recorder_manager = RecorderManager(recorders)
        self.mappings = mappings

    def before_run(self, runner) -> None:
        self.recorder_manager.initialize(runner.model)

    def after_train_epoch(self, runner) -> None:
        if runner.epoch % self.interval != 0:
            return

        if self.out_dir is not None:
            self.out_dir = osp.join(runner.work_dir, runner.timestamp,
                                         self.out_dir)
            mkdir_or_exist(self.out_dir)

        if self.file_client is None:
            self.file_client = FileClient(**self.file_client_args)

        cfg = runner.cfg.copy()
        test_pipeline = cfg.test_dataloader.dataset.pipeline
        new_test_pipeline = []
        for pipeline in test_pipeline:
            if pipeline['type'] != 'LoadAnnotations' and pipeline[
                'type'] != 'LoadPanopticAnnotations':
                new_test_pipeline.append(pipeline)

        test_pipeline = Compose(new_test_pipeline)
        dataset = runner.val_loop.dataloader.dataset

        for idx in self.data_idx:
            data_info = dataset.get_data_info(idx)
            img_path = data_info['img_path']
            data_ = dict(img_path=img_path, img_id=0)
            data_ = test_pipeline(data_)

            data_['inputs'] = [data_['inputs']]
            data_['data_samples'] = [data_['data_samples']]

            with torch.no_grad(), self.recorder_manager:
                runner.model.test_step(data_)

            if self.overlaid:
                img_bytes = self.file_client.get(img_path)
                overlaid_image = mmcv.imfrombytes(img_bytes, channel_order='rgb')
            else:
                overlaid_image = None

            for name, record in self.mappings.items():
                recorder = self.recorder_manager.get_recorder(record.recorder)
                record_idx = getattr(record, 'record_idx', 0)
                data_idx = getattr(record, 'data_idx')
                feats = recorder.get_record_data(record_idx, data_idx)
                if isinstance(feats, torch.Tensor):
                    feats = (feats,)

                for i, feat in enumerate(feats):
                    if self.use_norm:
                        feat = norm(feat)
                    drawn_img = self._visualizer.draw_featmap(
                        feat[0], overlaid_image, self.channel_reduction, self.topk, self.arrangement, self.resize_shape,
                        self.alpha)

                    out_file = None
                    if self.out_dir is not None:
                        out_file = f'data_idx{idx}_epoch{runner.epoch}_{name}_{i}.jpg'
                        out_file = osp.join(self.out_dir, out_file)

                    self._visualizer.add_datasample(
                        f'data_idx{idx}_epoch{runner.epoch}_{name}_{i}',
                        drawn_img,
                        show=self.show,
                        wait_time=0.1,
                        out_file=out_file)
