# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import Any, Dict, List, Union

import numpy as np
from mmengine.dataset.base_dataset import BaseDataset, force_full_init

from mmrazor.registry import DATASETS


@DATASETS.register_module()
class CRDDataset:
    """A wrapper of `CRD` dataset.

    Suitable for image classification datasets like CIFAR. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, each data sample has contrast information.
    Contrast information for an image is indices of negetive data samples.
    Note:
        ``CRDDataset`` should not inherit from ``BaseDataset``
        since ``get_subset`` and ``get_subset_`` could  produce ambiguous
        meaning sub-dataset which conflicts with original dataset. If you
        want to use a sub-dataset of ``CRDDataset``, you should set
        ``indices`` arguments for wrapped dataset which inherit from
        ``BaseDataset``.
    Args:
        dataset (BaseDataset or dict): The dataset to be repeated.
        neg_num (int): number of negetive data samples.
        percent (float): sampling percentage.
        lazy_init (bool, optional): whether to load annotation during
            instantiation. Defaults to False
        num_classes (int, optional): Number of classes. Defaults to None.
        sample_mode (str, optional): Data sampling mode. Defaults to 'exact'.
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 neg_num: int,
                 percent: float,
                 lazy_init: bool = False,
                 num_classes: int = None,
                 sample_mode: str = 'exact') -> None:
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif isinstance(dataset, BaseDataset):
            self.dataset = dataset
        else:
            raise TypeError(
                'elements in datasets sequence should be config or '
                f'`BaseDataset` instance, but got {type(dataset)}')
        self._metainfo = self.dataset.metainfo

        self._fully_initialized = False

        # CRD unique attributes.
        self.num_classes = num_classes
        self.neg_num = neg_num
        self.sample_mode = sample_mode
        self.percent = percent

        if not lazy_init:
            self.full_init()

    def _parse_fullset_contrast_info(self) -> None:
        """parse contrast information of the whole dataset."""
        assert self.sample_mode in [
            'exact', 'random'
        ], ('`sample_mode` must in [`exact`, `random`], '
            f'but get `{self.sample_mode}`')

        # Handle special occasion:
        #   if dataset's ``CLASSES`` is not list of consecutive integers,
        #   e.g. [2, 3, 5].
        num_classes: int = self.num_classes  # type: ignore
        if num_classes is None:
            num_classes = max(self.dataset.get_gt_labels()) + 1

        if not self.dataset.test_mode:  # type: ignore
            # Parse info.
            self.gt_labels = self.dataset.get_gt_labels()
            self.num_samples: int = self.dataset.__len__()

            self.cls_positive: List[List[int]] = [[]
                                                  for _ in range(num_classes)
                                                  ]  # type: ignore
            for i in range(self.num_samples):
                self.cls_positive[self.gt_labels[i]].append(i)

            self.cls_negative: List[List[int]] = [[]
                                                  for i in range(num_classes)
                                                  ]  # type: ignore
            for i in range(num_classes):  # type: ignore
                for j in range(num_classes):  # type: ignore
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [
                np.asarray(self.cls_positive[i])
                for i in range(num_classes)  # type: ignore
            ]
            self.cls_negative = [
                np.asarray(self.cls_negative[i])
                for i in range(num_classes)  # type: ignore
            ]

            if 0 < self.percent < 1:
                n = int(len(self.cls_negative[0]) * self.percent)
                self.cls_negative = [
                    np.random.permutation(self.cls_negative[i])[0:n]
                    for i in range(num_classes)  # type: ignore
                ]

            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

    @property
    def metainfo(self) -> dict:
        """Get the meta information of the repeated dataset.

        Returns:
            dict: The meta information of repeated dataset.
        """
        return copy.deepcopy(self._metainfo)

    def _get_contrast_info(self, data: Dict, idx: int) -> Dict:
        """Get contrast information for each data sample."""
        if self.sample_mode == 'exact':
            pos_idx = idx
        elif self.sample_mode == 'random':
            pos_idx = np.random.choice(self.cls_positive[self.gt_labels[idx]],
                                       1)
            pos_idx = pos_idx[0]  # type: ignore
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

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        self._parse_fullset_contrast_info()

        self._fully_initialized = True

    @force_full_init
    def get_data_info(self, idx: int) -> Dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.
        Returns:
            dict: The idx-th annotation of the dataset.
        """
        data_info = self.dataset.get_data_info(idx)  # type: ignore
        if not self.dataset.test_mode:  # type: ignore
            data_info = self._get_contrast_info(data_info, idx)
        return data_info

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        return self.dataset.pipeline(data_info)

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
            warnings.warn(
                'Please call `full_init()` method manually to accelerate '
                'the speed.')
            self.full_init()

        if self.dataset.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.dataset.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self.dataset._rand_another()
                continue
            return data

        raise Exception(
            f'Cannot find valid image after {self.dataset.max_refetch}! '
            'Please check your image path and pipeline')

    @force_full_init
    def __len__(self):
        return len(self.dataset)

    def get_subset_(self, indices: Union[List[int], int]) -> None:
        """Not supported in ``ClassBalancedDataset`` for the ambiguous meaning
        of sub-dataset."""
        raise NotImplementedError(
            '`ClassBalancedDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `ClassBalancedDataset`.')

    def get_subset(self, indices: Union[List[int], int]) -> 'BaseDataset':
        """Not supported in ``ClassBalancedDataset`` for the ambiguous meaning
        of sub-dataset."""
        raise NotImplementedError(
            '`ClassBalancedDataset` dose not support `get_subset` and '
            '`get_subset_` interfaces because this will lead to ambiguous '
            'implementation of some methods. If you want to use `get_subset` '
            'or `get_subset_` interfaces, please use them in the wrapped '
            'dataset first and then use `ClassBalancedDataset`.')
