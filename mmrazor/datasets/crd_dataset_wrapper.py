# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
from mmengine.dataset.base_dataset import BaseDataset, force_full_init

from mmrazor.registry import DATASETS
from .crd_dataset_mixin import CRD_ClsDatasetMixin


@DATASETS.register_module()
class CRDDataset:
    """A wrapper of class balanced dataset.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :meth:`get_cat_ids` to support
    ClassBalancedDataset.
    The repeat factor is computed as followed.
    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`
    Note:
        ``ClassBalancedDataset`` should not inherit from ``BaseDataset``
        since ``get_subset`` and ``get_subset_`` could  produce ambiguous
        meaning sub-dataset which conflicts with original dataset. If you
        want to use a sub-dataset of ``ClassBalancedDataset``, you should set
        ``indices`` arguments for wrapped dataset which inherit from
        ``BaseDataset``.
    Args:
        dataset (BaseDataset or dict): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        lazy_init (bool, optional): whether to load annotation during
            instantiation. Defaults to False
    """

    def __init__(self,
                 dataset: Union[BaseDataset, dict],
                 oversample_thr: float,
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
        self.oversample_thr = oversample_thr
        self._metainfo = self.dataset.metainfo

        self._fully_initialized = False
        self.num_classes = num_classes
        if not lazy_init:
            self.full_init()

    def _parse_fullset_contrast_info(self, neg_num: int, sample_mode: str,
                                     percent: float) -> None:
        """parse contrast information of the whole dataset.

        Args:
            neg_num (int): negative sample number.
            sample_mode (str): sample mode.
            percent (float): sampling percentage.
        """
        assert sample_mode in [
            'exact', 'random'
        ], ('`sample_mode` must in [`exact`, `random`], '
            f'but get `{sample_mode}`')

        # Handle special occasion:
        #   if dataset's ``CLASSES`` is not list of consecutive integers,
        #   e.g. [2, 3, 5].
        num_classes = self.num_classes
        if num_classes is None:
            num_classes: int = len(self.dataset.CLASSES)  # type: ignore

        if not self.dataset.test_mode:  # type: ignore
            # Must fully initialize dataset first.
            self.full_init()  # type: ignore

            # Parse info.
            self.gt_labels = [
                data['gt_label'] for data in self.dataset.data_list
            ]  # type: ignore
            self.neg_num = neg_num
            self.sample_mode = sample_mode
            self.num_samples = self.dataset.__len__()  # type: ignore

            self.cls_positive: List[List[int]] = [[]
                                                  for i in range(num_classes)]
            for i in range(self.num_samples):
                self.cls_positive[self.gt_labels[i]].append(i)

            self.cls_negative: List[List[int]] = [[]
                                                  for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [
                np.asarray(self.cls_positive[i]) for i in range(num_classes)
            ]
            self.cls_negative = [
                np.asarray(self.cls_negative[i]) for i in range(num_classes)
            ]

            if 0 < percent < 1:
                n = int(len(self.cls_negative[0]) * percent)
                self.cls_negative = [
                    np.random.permutation(self.cls_negative[i])[0:n]
                    for i in range(num_classes)
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

    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        self.dataset.full_init()
        # Get repeat factors for each image.
        repeat_factors = self._get_repeat_factors(self.dataset,
                                                  self.oversample_thr)
        # Repeat dataset's indices according to repeat_factors. For example,
        # if `repeat_factors = [1, 2, 3]`, and the `len(dataset) == 3`,
        # the repeated indices will be [1, 2, 2, 3, 3, 3].
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        self._fully_initialized = True

    def _get_repeat_factors(self, dataset: BaseDataset,
                            repeat_thr: float) -> List[float]:
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (BaseDataset): The dataset.
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.
        Returns:
            List[float]: The repeat factors for each images in the dataset.
        """
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq: defaultdict = defaultdict(float)
        num_images = len(dataset)
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        for k, v in category_freq.items():
            assert v > 0, f'caterogy {k} does not contain any images'
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I and its labels L(I), compute the image-level
        # repeat factor:
        #    r(I) = max_{c in L(I)} r(c)
        repeat_factors = []
        for idx in range(num_images):
            cat_ids = set(self.dataset.get_cat_ids(idx))
            if len(cat_ids) != 0:
                repeat_factor = max(
                    {category_repeat[cat_id]
                     for cat_id in cat_ids})
                repeat_factors.append(repeat_factor)

        return repeat_factors

    @force_full_init
    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.
        Returns:
            int: Local index of data.
        """
        return self.repeat_indices[idx]

    @force_full_init
    def get_cat_ids(self, idx: int) -> List[int]:
        """Get category ids of class balanced dataset by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_cat_ids(sample_idx)

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.
        Returns:
            dict: The idx-th annotation of the dataset.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)

    def __getitem__(self, idx):
        warnings.warn('Please call `full_init` method manually to '
                      'accelerate the speed.')
        if not self._fully_initialized:
            self.full_init()

        ori_index = self._get_ori_dataset_idx(idx)
        return self.dataset[ori_index]

    @force_full_init
    def __len__(self):
        return len(self.repeat_indices)

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


@DATASETS.register_module()
class CRD_CIFAR10(CIFAR10, CRD_ClsDatasetMixin):
    """The reconstructed dataset for crd distillation reconstructed dataset
    adds another four parameters to control the resampling of data. If is
    training mode, the dataset will return one positive sample and index of
    ``neg_num`` negative sample; else the dataset only return one positive
    sample like the dataset in ``mmcls`` library.

    Args:
        data_prefix (str): Prefix for data.
        test_mode (bool): ``test_mode=True`` means in test phase.
            It determines to use the training set or test set.
        metainfo (dict, optional): Meta information for dataset, such as
            categories information. Defaults to None.
        data_root (str): The root directory for ``data_prefix``.
            Defaults to ''.
        download (bool): Whether to download the dataset if not exists.
            Defaults to True.
        neg_num (int, optional): Number of negative samples. Defaults to 16384.
        sample_mode (str, optional): Sample mode. Defaults to 'exact'.
        percent (float, optional): Sampling percentage. Defaults to 1.0.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
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
        CIFAR10.__init__(self, data_prefix, test_mode, metainfo, data_root,
                         download, **kwargs)
        CRD_ClsDatasetMixin.__init__(self, neg_num, percent, sample_mode)

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.
        Returns:
            Any: Depends on ``self.pipeline``.
        """
        return CRD_ClsDatasetMixin.prepare_data(self, idx)


@DATASETS.register_module()
class CRD_CIFAR100(CRD_CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        data_prefix (str): Prefix for data.
        test_mode (bool): ``test_mode=True`` means in test phase.
            It determines to use the training set or test set.
        metainfo (dict, optional): Meta information for dataset, such as
            categories information. Defaults to None.
        data_root (str): The root directory for ``data_prefix``.
            Defaults to ''.
        download (bool): Whether to download the dataset if not exists.
            Defaults to True.
        **kwargs: Other keyword arguments in :class:`BaseDataset`.
    """

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
    METAINFO = {'classes': CIFAR100_CATEGORIES}
    num_classes = 100


class CRD_ClsDatasetMixin(object):
    """Dataset mixin for CRD algorithm on classification datasets.

    Args:
        neg_num (int): Number of negative samples.
        percent (float): Sampling percentage.
        sample_mode (str): Sample mode. Defaults to 'exact'.
    """

    def __init__(self,
                 neg_num: int,
                 percent: float,
                 sample_mode: str = 'exact') -> None:

        self._parse_fullset_contrast_info(neg_num, sample_mode, percent)

    def _parse_fullset_contrast_info(self, neg_num: int, sample_mode: str,
                                     percent: float) -> None:
        """parse contrast information of the whole dataset.

        Args:
            neg_num (int): negative sample number.
            sample_mode (str): sample mode.
            percent (float): sampling percentage.
        """
        assert sample_mode in [
            'exact', 'random'
        ], ('`sample_mode` must in [`exact`, `random`], '
            f'but get `{sample_mode}`')

        # Handle special occasion:
        #   if dataset's ``CLASSES`` is not list of consecutive integers,
        #   e.g. [2, 3, 5].
        num_classes = getattr(self, 'num_classes', None)
        if num_classes is None:
            num_classes: int = len(self.CLASSES)  # type: ignore

        if not self.test_mode:  # type: ignore
            # Must fully initialize dataset first.
            self.full_init()  # type: ignore

            # Parse info.
            self.gt_labels = [data['gt_label']
                              for data in self.data_list]  # type: ignore
            self.neg_num = neg_num
            self.sample_mode = sample_mode
            self.num_samples = self.__len__()  # type: ignore

            self.cls_positive: List[List[int]] = [[]
                                                  for i in range(num_classes)]
            for i in range(self.num_samples):
                self.cls_positive[self.gt_labels[i]].append(i)

            self.cls_negative: List[List[int]] = [[]
                                                  for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [
                np.asarray(self.cls_positive[i]) for i in range(num_classes)
            ]
            self.cls_negative = [
                np.asarray(self.cls_negative[i]) for i in range(num_classes)
            ]

            if 0 < percent < 1:
                n = int(len(self.cls_negative[0]) * percent)
                self.cls_negative = [
                    np.random.permutation(self.cls_negative[i])[0:n]
                    for i in range(num_classes)
                ]

            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)

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

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.
        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)  # type: ignore
        if not self.test_mode:  # type: ignore
            data_info = self._get_contrast_info(data_info, idx)
        return self.pipeline(data_info)  # type: ignore
