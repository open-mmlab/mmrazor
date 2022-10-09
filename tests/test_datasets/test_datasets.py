# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import pickle
import tempfile
from unittest import TestCase

import numpy as np
from mmcls.registry import DATASETS as CLS_DATASETS

from mmrazor.registry import DATASETS
from mmrazor.utils import register_all_modules

register_all_modules()
ASSETS_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../data/dataset'))


class Test_CRD_CIFAR10(TestCase):
    ORI_DATASET_TYPE = 'CIFAR10'
    DATASET_TYPE = 'CRDDataset'

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        tmpdir = tempfile.TemporaryDirectory()
        cls.tmpdir = tmpdir
        data_prefix = tmpdir.name
        cls.ORI_DEFAULT_ARGS = dict(
            data_prefix=data_prefix, pipeline=[], test_mode=False)
        cls.DEFAULT_ARGS = dict(neg_num=1, percent=0.5)

        dataset_class = CLS_DATASETS.get(cls.ORI_DATASET_TYPE)
        base_folder = osp.join(data_prefix, dataset_class.base_folder)
        os.mkdir(base_folder)

        cls.fake_imgs = np.random.randint(
            0, 255, size=(6, 3 * 32 * 32), dtype=np.uint8)
        cls.fake_labels = np.random.randint(0, 10, size=(6, ))
        cls.fake_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        batch1 = dict(
            data=cls.fake_imgs[:2], labels=cls.fake_labels[:2].tolist())
        with open(osp.join(base_folder, 'data_batch_1'), 'wb') as f:
            f.write(pickle.dumps(batch1))

        batch2 = dict(
            data=cls.fake_imgs[2:4], labels=cls.fake_labels[2:4].tolist())
        with open(osp.join(base_folder, 'data_batch_2'), 'wb') as f:
            f.write(pickle.dumps(batch2))

        test_batch = dict(
            data=cls.fake_imgs[4:], fine_labels=cls.fake_labels[4:].tolist())
        with open(osp.join(base_folder, 'test_batch'), 'wb') as f:
            f.write(pickle.dumps(test_batch))

        meta = {dataset_class.meta['key']: cls.fake_classes}
        meta_filename = dataset_class.meta['filename']
        with open(osp.join(base_folder, meta_filename), 'wb') as f:
            f.write(pickle.dumps(meta))

        dataset_class.train_list = [['data_batch_1', None],
                                    ['data_batch_2', None]]
        dataset_class.test_list = [['test_batch', None]]
        dataset_class.meta['md5'] = None

    def test_initialize(self):
        dataset_class = DATASETS.get(self.DATASET_TYPE)

        # Test overriding metainfo by `metainfo` argument
        ori_cfg = {
            **self.ORI_DEFAULT_ARGS, 'metainfo': {
                'classes': ('bus', 'car')
            },
            'type': self.ORI_DATASET_TYPE,
            '_scope_': 'mmcls'
        }
        cfg = {'dataset': ori_cfg, **self.DEFAULT_ARGS}
        dataset = dataset_class(**cfg)
        self.assertEqual(dataset.dataset.CLASSES, ('bus', 'car'))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()


class Test_CRD_CIFAR100(Test_CRD_CIFAR10):
    ORI_DATASET_TYPE = 'CIFAR100'
