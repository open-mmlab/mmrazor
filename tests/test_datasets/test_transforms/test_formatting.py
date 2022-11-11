# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import unittest

import numpy as np
import torch
from mmcls.structures import ClsDataSample
from mmengine.structures import LabelData

from mmrazor.datasets.transforms import PackCRDClsInputs


class TestPackClsInputs(unittest.TestCase):

    def setUp(self):
        """Setup the model and optimizer which are used in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        data_prefix = osp.join(osp.dirname(__file__), '../../data')
        img_path = osp.join(data_prefix, 'color.jpg')
        rng = np.random.RandomState(0)
        self.results1 = {
            'sample_idx': 1,
            'img_path': img_path,
            'ori_height': 300,
            'ori_width': 400,
            'height': 600,
            'width': 800,
            'scale_factor': 2.0,
            'flip': False,
            'img': rng.rand(300, 400),
            'gt_label': rng.randint(3, ),
            # TODO.
            'contrast_sample_idxs': rng.randint(3, )
        }
        self.meta_keys = ('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                          'scale_factor', 'flip')

    def test_transform(self):
        transform = PackCRDClsInputs(meta_keys=self.meta_keys)
        results = transform(copy.deepcopy(self.results1))
        self.assertIn('inputs', results)
        self.assertIsInstance(results['inputs'], torch.Tensor)
        self.assertIn('data_samples', results)
        self.assertIsInstance(results['data_samples'], ClsDataSample)

        data_sample = results['data_samples']
        self.assertIsInstance(data_sample.gt_label, LabelData)

    def test_repr(self):
        transform = PackCRDClsInputs(meta_keys=self.meta_keys)
        self.assertEqual(
            repr(transform), f'PackCRDClsInputs(meta_keys={self.meta_keys})')
