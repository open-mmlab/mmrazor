# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import shutil
import time
from os.path import dirname
from typing import Optional
from unittest import TestCase
from unittest.mock import Mock

import torch
import torch.nn as nn
# TODO: The argument `out_file` has not been supported in MMEngine yet.
#  Temporarily, we use `ClsVisualizer` here
from mmcls.visualization import ClsVisualizer
from mmengine import ConfigDict
from mmengine.model import BaseModel

from mmrazor.engine.hooks import RazorVisualizationHook


def get_data_info(idx):
    root_path = dirname(dirname(dirname(dirname(__file__))))
    return {
        'img_path': os.path.join(root_path, 'tools/visualizations/demo.jpg')
    }


class ToyModel(BaseModel):

    def __init__(self):
        data_preprocessor = dict(
            type='mmcls.ClsDataPreprocessor',
            # RGB format normalization parameters
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            # convert image from BGR to RGB
            to_rgb=True,
        )
        super().__init__(data_preprocessor=data_preprocessor)
        self.op = nn.Conv2d(3, 3, 1)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        out = self.op(inputs)
        return out


class TestVisualizationHook(TestCase):

    def setUp(self) -> None:
        # TODO: The argument `out_file` has not been supported in MMEngine yet.
        #  Temporarily, we use `ClsVisualizer` here
        ClsVisualizer.get_instance('visualizer')

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='mmcls.PackClsInputs')
        ]

        self.runner = Mock()
        self.runner.val_loop.dataloader.dataset.get_data_info = get_data_info
        self.runner.cfg = ConfigDict(
            test_dataloader=dict(dataset=dict(pipeline=test_pipeline)))
        self.runner.model = ToyModel()

        self.recorders = ConfigDict(
            out=dict(_scope_='mmrazor', type='ModuleOutputs', source='op'))
        self.mappings = ConfigDict(out=dict(recorder='out'))

    def test_before_run(self):
        hook = RazorVisualizationHook(self.recorders, self.mappings)
        hook.before_run(self.runner)

    def test_before_train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        out_dir = timestamp + '1'
        self.runner.work_dir = timestamp
        self.runner.timestamp = '1'
        self.runner.epoch = 0

        hook = RazorVisualizationHook(
            self.recorders, self.mappings, out_dir=out_dir, enabled=False)
        # initialize recorders
        hook.before_run(self.runner)
        hook.before_train(self.runner)
        self.assertTrue(not osp.exists(f'{timestamp}/1/{out_dir}'))

        hook = RazorVisualizationHook(
            self.recorders, self.mappings, out_dir=out_dir, enabled=True)
        # initialize recorders
        hook.before_run(self.runner)
        hook.before_train(self.runner)
        self.assertTrue(osp.exists(f'{timestamp}/1/{out_dir}'))
        shutil.rmtree(f'{timestamp}')

    def test_after_train_epoch(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        out_dir = timestamp + '1'
        self.runner.work_dir = timestamp
        self.runner.timestamp = '1'

        hook = RazorVisualizationHook(
            self.recorders, self.mappings, out_dir=out_dir, enabled=False)
        # initialize recorders
        hook.before_run(self.runner)
        self.runner.epoch = 0
        hook.after_train_epoch(self.runner)
        self.assertTrue(not osp.exists(f'{timestamp}/1/{out_dir}'))

        self.runner.epoch = 1
        hook = RazorVisualizationHook(
            self.recorders,
            self.mappings,
            out_dir=out_dir,
            enabled=True,
            interval=2)
        # initialize recorders
        hook.before_run(self.runner)
        hook.after_train_epoch(self.runner)
        self.assertTrue(not osp.exists(f'{timestamp}/1/{out_dir}'))

        self.runner.epoch = 2
        hook.after_train_epoch(self.runner)
        self.assertTrue(osp.exists(f'{timestamp}/1/{out_dir}'))
        shutil.rmtree(f'{timestamp}')
