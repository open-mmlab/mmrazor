# Copyright (c) OpenMMLab. All rights reserved.
import os
import shutil
import subprocess
from typing import List
from unittest import TestCase

config_paths: List = [
    'mmcls::resnet/resnet34_8xb32_in1k.py',
    'mmdet::retinanet/retinanet_r18_fpn_1x_coco.py',
    'mmseg::deeplabv3plus/deeplabv3plus_r50-d8_4xb4-20k_voc12aug-512x512.py',
]


class TestTools(TestCase):

    def _setUp(self) -> None:
        self.workdir = os.path.dirname(__file__) + '/tmp/'
        if not os.path.exists(self.workdir):
            os.mkdir(self.workdir)

    def save_to_config(self, name, content):
        with open(self.workdir + f'/{name}', 'w') as f:
            f.write(content)

    def test_get_channel_unit(self):
        for path in config_paths:
            with self.subTest(path=path):
                self._setUp()
                self.save_to_config('pretrain.py', f"""_base_=['{path}']""")
                try:
                    subprocess.run([
                        'python', './tools/get_channel_units.py',
                        f'{self.workdir}/pretrain.py', '-o',
                        f'{self.workdir}/unit.json'
                    ])
                    self.assertTrue(
                        os.path.exists(f'{self.workdir}/unit.json'))

                except Exception as e:
                    self.fail(f'{e}')
                self._tearDown()

    def test_get_search_config(self):
        for path in config_paths:
            with self.subTest(path=path):
                self._setUp()
                self.save_to_config('pretrain.py', f"""_base_=['{path}']""")
                try:
                    subprocess.run([
                        'python',
                        './tools/get_search_config.py',
                        f'{self.workdir}/pretrain.py',
                        '-o',
                        f'{self.workdir}/search.py',
                    ])
                    self.assertTrue(
                        os.path.exists(f'{self.workdir}/search.py'))
                except Exception as e:
                    self.fail(f'{e}')
                self._tearDown()

    def test_get_prune_config(self):
        for path in config_paths:
            with self.subTest(path=path):
                self._setUp()
                self.save_to_config('pretrain.py', f"""_base_=['{path}']""")
                try:
                    subprocess.run([
                        'python',
                        './tools/get_prune_config.py',
                        f'{self.workdir}/pretrain.py',
                        '-o',
                        f'{self.workdir}/prune.py',
                    ])
                    pass
                    self.assertTrue(os.path.exists(f'{self.workdir}/prune.py'))
                except Exception as e:
                    self.fail(f'{e}')
                self._tearDown()

    def _tearDown(self) -> None:
        print('delete')
        shutil.rmtree(self.workdir)
        pass
