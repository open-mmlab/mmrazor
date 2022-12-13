# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import subprocess
from unittest import TestCase

import torch

from mmrazor import digit_version

TEST_TOOLS = os.getenv('TEST_TOOLS') == 'true'


class TestTools(TestCase):
    _config_path = None

    def setUp(self) -> None:
        if not TEST_TOOLS:
            self.skipTest('disabled')

    @property
    def config_path(self):
        if self._config_path is None:
            self._config_path = self._get_config_path()
        return self._config_path

    def _setUp(self) -> None:
        self.workdir = os.path.dirname(__file__) + '/tmp/'
        if not os.path.exists(self.workdir):
            os.mkdir(self.workdir)

    def save_to_config(self, name, content):
        with open(self.workdir + f'/{name}', 'w') as f:
            f.write(content)

    def test_get_channel_unit(self):
        if digit_version(torch.__version__) < digit_version('1.12.0'):
            self.skipTest('version of torch < 1.12.0')

        for path in self.config_path:
            with self.subTest(path=path):
                self._setUp()
                self.save_to_config('pretrain.py', f"""_base_=['{path}']""")
                try:
                    subprocess.run([
                        'python', './tools/pruning/get_channel_units.py',
                        f'{self.workdir}/pretrain.py', '-o',
                        f'{self.workdir}/unit.json'
                    ])
                except Exception as e:
                    self.fail(f'{e}')
                self.assertTrue(os.path.exists(f'{self.workdir}/unit.json'))

                self._tearDown()

    def test_get_prune_config(self):
        if digit_version(torch.__version__) < digit_version('1.12.0'):
            self.skipTest('version of torch < 1.12.0')
        for path in self.config_path:
            with self.subTest(path=path):
                self._setUp()
                self.save_to_config('pretrain.py', f"""_base_=['{path}']""")
                try:
                    subprocess.run([
                        'python',
                        './tools/pruning/get_l1_prune_config.py',
                        f'{self.workdir}/pretrain.py',
                        '-o',
                        f'{self.workdir}/prune.py',
                    ])
                    pass
                except Exception as e:
                    self.fail(f'{e}')
                self.assertTrue(os.path.exists(f'{self.workdir}/prune.py'))

                self._tearDown()

    def _tearDown(self) -> None:
        print('delete')
        shutil.rmtree(self.workdir)
        pass

    def _get_config_path(self):
        config_paths = []
        paths = [
            ('mmcls', 'mmcls::resnet/resnet34_8xb32_in1k.py'),
            ('mmdet', 'mmdet::retinanet/retinanet_r18_fpn_1x_coco.py'),
            (
                'mmseg',
                'mmseg::deeplabv3plus/deeplabv3plus_r50-d8_4xb4-20k_voc12aug-512x512.py'  # noqa
            ),
            ('mmyolo',
             'mmyolo::yolov5/yolov5_m-p6-v62_syncbn_fast_8xb16-300e_coco.py')
        ]
        for repo_name, path in paths:
            try:
                __import__(repo_name)
                config_paths.append(path)
            except Exception:
                pass
        return config_paths
