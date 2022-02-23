# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import random_split


def split_dataset(dataset):
    dset_length = len(dataset)

    first_dset_length = dset_length // 2
    second_dset_length = dset_length - first_dset_length
    split_tuple = (first_dset_length, second_dset_length)
    first_dset, second_dset = random_split(dataset, split_tuple)

    first_dset.CLASSES = dataset.CLASSES
    second_dset.CLASSES = dataset.CLASSES

    return [first_dset, second_dset]
