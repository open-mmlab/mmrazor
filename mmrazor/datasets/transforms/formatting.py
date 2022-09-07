# Copyright (c) OpenMMLab. All rights reserved.
try:
    from mmcls.datasets.transforms.formatting import PackClsInputs, to_tensor
    from mmcls.structures import ClsDataSample
except ImportError:
    from mmrazor.utils import get_placeholder
    PackClsInputs = get_placeholder('mmcls')
    to_tensor = get_placeholder('mmcls')
    ClsDataSample = get_placeholder('mmcls')

import warnings
from typing import Any, Dict, Generator

import numpy as np
import torch

from mmrazor.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackCRDClsInputs(PackClsInputs):

    def transform(self, results: Dict) -> Dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:
            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`ClsDataSample`): The annotation info of the
              sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)
        else:
            warnings.warn(
                'Cannot get "img" in the input dict of `PackClsInputs`,'
                'please make sure `LoadImageFromFile` has been added '
                'in the data pipeline or images have been loaded in '
                'the dataset.')

        data_sample = ClsDataSample()
        if 'gt_label' in results:
            gt_label = results['gt_label']
            data_sample.set_gt_label(gt_label)

        if 'sample_idx' in results:
            # transfer `sample_idx` to Tensor
            self.meta_keys: Generator[Any, None, None] = (
                key for key in self.meta_keys if key != 'sample_idx')
            value = results['sample_idx']
            if isinstance(value, int):
                value = torch.tensor(value).to(torch.long)
            data_sample.set_data(dict(sample_idx=value))

        if 'contrast_sample_idxs' in results:
            value = results['contrast_sample_idxs']
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).to(torch.long)
            data_sample.set_data(dict(contrast_sample_idxs=value))

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results
