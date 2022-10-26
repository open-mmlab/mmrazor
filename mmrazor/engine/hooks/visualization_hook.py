# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import List, Optional, Union

import mmcv
import torch
from mmcv.transforms import Compose
from mmengine.dist import master_only
from mmengine.fileio import FileClient
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from mmrazor.models.task_modules import RecorderManager
from mmrazor.registry import HOOKS
from mmrazor.visualization.local_visualizer import modify


def norm(feat):
    assert len(feat.shape) == 4
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    centered = (feat - mean) / (std + 1e-6)
    centered = centered.reshape(C, N, H, W).permute(1, 0, 2, 3)
    return centered


@HOOKS.register_module()
class RazorVisualizationHook(Hook):
    """Razor Visualization Hook. Used to visualize training process immediate
    feature maps.

    1. If ``show`` is True, it means that only the immediate feature maps are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``out_dir`` is specified, it means that the immediate feature maps
        need to be saved to ``out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the immediate feature maps in Wandb or
        Tensorboard.

    Args:
        recorders (dict): All recorders' config.
        mappings: (Dict[str, Dict]): The mapping between feature names and
            records.
        enabled (bool): Whether to draw immediate feature maps. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 1.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        out_dir (str, optional): directory where painted images
            will be saved in testing process.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmengine.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        is_overlaid (bool): If `is_overlaid` is True, the final output image
            will be the weighted sum of img and featmap. Defaults to True.
        visualization_cfg (dict): Configs for visualization.
        use_norm (bool): Whether to apply Batch Normalization over the
            feature map. Defaults to False.
    """

    def __init__(self,
                 recorders: dict,
                 mappings: dict,
                 enabled: bool = False,
                 data_idx: Union[int, List] = 0,
                 interval: int = 1,
                 show: bool = False,
                 wait_time: float = 0.1,
                 out_dir: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk'),
                 is_overlaid: bool = True,
                 visualization_cfg=dict(
                     channel_reduction='pixel_wise_max',
                     topk=20,
                     arrangement=(4, 5),
                     resize_shape=None,
                     alpha=0.5),
                 use_norm: bool = False):
        self.enabled = enabled
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self._visualizer.draw_featmap = modify
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
        self.interval = interval

        self.is_overlaid = is_overlaid
        self.visualization_cfg = visualization_cfg
        self.use_norm = use_norm

        self.recorder_manager = RecorderManager(recorders)
        self.mappings = mappings

        self._step = 0  # Global step value to record

    @master_only
    def before_run(self, runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            self.recorder_manager.initialize(model.module)
        else:
            self.recorder_manager.initialize(model)

    @master_only
    def before_train(self, runner):
        if not self.enabled or runner.epoch % self.interval != 0:
            return
        self._visualize(runner, 'before_run')

    @master_only
    def after_train_epoch(self, runner) -> None:
        if not self.enabled or runner.epoch % self.interval != 0:
            return
        self._visualize(runner, f'epoch_{runner.epoch}')

    def _visualize(self, runner, stage):
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

            if self.is_overlaid:
                img_bytes = self.file_client.get(img_path)
                overlaid_image = mmcv.imfrombytes(
                    img_bytes, channel_order='rgb')
            else:
                overlaid_image = None

            for name, record in self.mappings.items():
                recorder = self.recorder_manager.get_recorder(record.recorder)
                record_idx = getattr(record, 'record_idx', 0)
                data_idx = getattr(record, 'data_idx', None)
                feats = recorder.get_record_data(record_idx, data_idx)
                if isinstance(feats, torch.Tensor):
                    feats = (feats, )

                for i, feat in enumerate(feats):
                    if self.use_norm:
                        feat = norm(feat)
                    drawn_img = self._visualizer.draw_featmap(
                        feat[0], overlaid_image, **self.visualization_cfg)

                    out_file = None
                    if self.out_dir is not None:
                        out_file = f'{stage}_data_idx_{idx}_{name}_{i}.jpg'
                        out_file = osp.join(self.out_dir, out_file)

                    self._visualizer.add_datasample(
                        f'{stage}_data_idx_{idx}_{name}_{i}',
                        drawn_img,
                        draw_gt=False,
                        draw_pred=False,
                        show=self.show,
                        wait_time=0.1,
                        # TODO: Supported in mmengine's Viusalizer.
                        out_file=out_file,
                        step=self._step)
                    self._step += 1
