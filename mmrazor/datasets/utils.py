# Copyright (c) OpenMMLab. All rights reserved.
from torch.utils.data import random_split


def split_dataset(dataset, proportion=0.5, num=0):
    dset_length = len(dataset)
    if num > 0:
        second_dset_length = num
    else:
        second_dset_length = int(dset_length * proportion)

    first_dset_length = dset_length - second_dset_length
    split_tuple = (first_dset_length, second_dset_length)
    first_dset, second_dset = random_split(dataset, split_tuple)

    first_dset.CLASSES = dataset.CLASSES
    second_dset.CLASSES = dataset.CLASSES
    return [first_dset, second_dset]
