# Copyright (c) OpenMMLab. All rights reserved.
import os
import random

from torch.utils.data import random_split


def split_dataset_txt(txt_path, save_dir):
    index_pool = dict()
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            (filename, index) = line.split(' ')
            if index not in index_pool:
                index_pool[index] = []
            index_pool[index].append(filename)

    assert len(index_pool) == 1000

    train_txt = list()
    val_txt = list()
    for index, filenames in index_pool.items():
        filenames_val = random.sample(filenames, 50)
        for name in filenames_val:
            filenames.remove(name)
        filenames_train = filenames
        for name in filenames_train:
            train_txt.append(f'{name} {index}')
        for name in filenames_val:
            val_txt.append(f'{name} {index}')

    assert len(val_txt) == 50000

    with open(os.path.join(save_dir, 'greedynas_train.txt'), 'w') as f:
        for line in train_txt:
            f.writelines(f'{line}\n')
    with open(os.path.join(save_dir, 'greedynas_val.txt'), 'w') as f:
        for line in val_txt:
            f.writelines(f'{line}\n')

    print(f'Split txt finished, greedynas_train.txt and '
          f'greedynas_val.txt were saved in {save_dir}')


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


if __name__ == '__main__':
    import sys

    split_dataset_txt(sys.argv[1], sys.argv[2])

