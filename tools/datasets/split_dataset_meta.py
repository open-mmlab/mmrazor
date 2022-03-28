# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description='Split imagenet dataset \
                                                  by .txt')
    parser.add_argument('txt_path', help='.txt path to split')
    parser.add_argument('save_dir', help='the dir to save splited results')
    parser.add_argument(
        '--test_size',
        type=float,
        default=None,
        help='If float, should be between 0.0 and 1.0 and represent the \
            proportion of the dataset to include in the test split. If int, \
            represents the absolute number of test samples. If None, \
            the value is set to the complement of the train size. \
            If train_size is also None, it will be set to 0.25.')
    parser.add_argument(
        '--train_size',
        type=float,
        default=None,
        help='If float, should be between 0.0 and 1.0 and represent the \
            proportion of the dataset to include in the train split. If int, \
            represents the absolute number of train samples. If None, the \
            value is automatically set to the complement of the test size.')
    parser.add_argument(
        '--random_state',
        type=int,
        default=None,
        help='Controls the shuffling applied to the data before applying \
            the split. Pass an int for reproducible output across multiple \
            function calls.')
    parser.add_argument(
        '--shuffle',
        type=bool,
        default=True,
        help='Whether or not to shuffle the data before splitting. \
            If shuffle=False then is_stratified must be False.')
    parser.add_argument(
        '--is_stratified',
        type=bool,
        default=True,
        help='If True, data is split in a stratified fashion, \
            else it is randomly splited.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    filenames = []
    labels = []
    with open(args.txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            (filename, label) = line.split(' ')
            filenames.append(filename)
            labels.append(label)

    if args.test_size is None:
        test_size = args.test_size
    elif args.test_size >= 1:
        test_size = int(args.test_size)
    else:
        test_size = float(args.test_size)

    if args.train_size is None:
        train_size = args.train_size
    elif args.train_size >= 1:
        train_size = int(args.train_size)
    else:
        train_size = float(args.train_size)

    stratify = None
    if args.is_stratified:
        stratify = labels

    X_train, X_test, y_train, y_test = train_test_split(
        filenames,
        labels,
        test_size=test_size,
        train_size=train_size,
        random_state=args.random_state,
        shuffle=args.shuffle,
        stratify=stratify)

    with open(os.path.join(args.save_dir, 'splited_train.txt'), 'w') as f:
        for i in range(len(X_train)):
            f.writelines(f'{X_train[i]} {y_train[i]}\n')
    with open(os.path.join(args.save_dir, 'splited_test.txt'), 'w') as f:
        for i in range(len(X_test)):
            f.writelines(f'{X_test[i]} {y_test[i]}\n')

    print(f'nums of splited_train_txt: {len(X_train)}; '
          f'nums of splited_test_txt: {len(X_test)}')
    print(f'Split txt finished, splited_train.txt and '
          f'splited_test.txt were saved in {args.save_dir}')


if __name__ == '__main__':
    main()
