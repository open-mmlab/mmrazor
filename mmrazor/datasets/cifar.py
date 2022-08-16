# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, List, Optional

import numpy as np

from mmrazor.registry import DATASETS

try:
    from mmcls.datasets.cifar import CIFAR10
except ImportError:
    from ..utils import get_placeholder
    CIFAR10 = get_placeholder('mmcls')


@DATASETS.register_module()
class CRD_CIFAR10(CIFAR10):
    """The reconstructed dataset for crd distillation reconstructed dataset
    adds another four parameters to control the resampling of data. If
    is_train, the dataset will return one positive sample and index of neg_num
    negative sample; else the dataset only return one positive sample like the
    dataset in mmcls library.

    Args:
        is_train: If True, `` __getitem__`` method will return the positive
            sample and index of `neg_num` negative sample, usually used in
            the training of crd distillation. And 'False' is used in the
            eval and test dataset of crd distillation.
        mode: Controls how negative samples are generated.
        percent: Control the cutting ratio of negative samples.
        neg_num: The index length of negative samples.

    Returns:
        dict: Dict of dataset sample. The following fields are contained.
            - img and gt_label: Same as the dataset in mmcls library.
            - contrast_sample_idx: the indexes of contrasted
                samples(neg_num + 1).
            - idx: The index of sample.
    """
    num_classes = 10

    def __init__(self,
                 data_prefix: str,
                 test_mode: bool,
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 download: bool = True,
                 neg_num=16384,
                 sample_mode='exact',
                 percent=1.0,
                 serialize_data=False,
                 **kwargs):
        assert serialize_data is False, NotImplementedError(
            '`serialize_data` is not supported for now.')
        kwargs['serialize_data'] = serialize_data
        super().__init__(data_prefix, test_mode, metainfo, data_root, download,
                         **kwargs)

        self.full_init()

        # parse contrast info.
        assert sample_mode in ['exact', 'random']
        if not self.test_mode:
            self.gt_labels = [data['gt_label'] for data in self.data_list]
            self.neg_num = neg_num
            self.sample_mode = sample_mode
            self.num_samples = self.__len__()

            self.cls_positive: List[List[int]] = [
                [] for i in range(self.num_classes)
            ]
            for i in range(self.num_samples):
                self.cls_positive[self.gt_labels[i]].append(i)

            self.cls_negative: List[List[int]] = [
                [] for i in range(self.num_classes)
            ]
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [
                np.asarray(self.cls_positive[i])
                for i in range(self.num_classes)
            ]
            self.cls_negative = [
                np.asarray(self.cls_negative[i])
                for i in range(self.num_classes)
            ]

            if 0 < percent < 1:
                n = int(len(self.cls_negative[0]) * percent)
                self.cls_negative = [
                    np.random.permutation(self.cls_negative[i])[0:n]
                    for i in range(self.num_classes)
                ]

            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    def _get_contrast_info(self, data, idx):
        if self.sample_mode == 'exact':
            pos_idx = idx
        elif self.sample_mode == 'random':
            pos_idx = np.random.choice(self.cls_positive[self.gt_labels[idx]],
                                       1)
            pos_idx = pos_idx[0]
        else:
            raise NotImplementedError(self.sample_mode)
        replace = True if self.neg_num > \
            len(self.cls_negative[self.gt_labels[idx]]) else False
        neg_idx = np.random.choice(
            self.cls_negative[self.gt_labels[idx]],
            self.neg_num,
            replace=replace)
        contrast_sample_idxs = np.hstack((np.asarray([pos_idx]), neg_idx))
        data['contrast_sample_idxs'] = contrast_sample_idxs
        return data

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.
        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        if not self.test_mode:
            data_info = self._get_contrast_info(data_info, idx)
        return self.pipeline(data_info)


@DATASETS.register_module()
class CRD_CIFAR100(CRD_CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset."""

    base_folder = 'cifar-100-python'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    filename = 'cifar-100-python.tar.gz'
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    num_classes = 100
