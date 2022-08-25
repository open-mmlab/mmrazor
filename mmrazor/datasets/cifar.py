# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional

from mmrazor.registry import DATASETS
from .crd_dataset_mixin import CRD_ClsDatasetMixin

try:
    from mmcls.datasets.categories import CIFAR100_CATEGORIES
    from mmcls.datasets.cifar import CIFAR10
except ImportError:
    from ..utils import get_placeholder
    CIFAR10 = get_placeholder('mmcls')
    CIFAR100_CATEGORIES = get_placeholder('mmcls')


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
