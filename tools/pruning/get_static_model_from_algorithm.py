# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import os

import torch
from mmengine import Config, fileio
from mmengine.runner.checkpoint import load_checkpoint

from mmrazor.models import BaseAlgorithm
from mmrazor.registry import MODELS
from mmrazor.utils import print_log


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export a pruned model checkpoint.')
    parser.add_argument('config', help='config of the model')
    parser.add_argument(
        'checkpoint',
        default=None,
        type=str,
        help='checkpoint path of the model')
    parser.add_argument(
        '-o',
        type=str,
        default='',
        help='output path to store the pruned checkpoint.')
    args = parser.parse_args()
    return args


def get_save_path(config_path, checkpoint_path, target_path):
    if target_path != '':
        work_dir = target_path
    else:
        work_dir = 'work_dirs/' + os.path.basename(config_path).split('.')[0]

    checkpoint_name = os.path.basename(checkpoint_path).split(
        '.')[0] + '_pruned'

    return work_dir, checkpoint_name


def get_static_model(algorithm):
    from mmrazor.structures.subnet import export_fix_subnet, load_fix_subnet
    pruning_structure = algorithm.mutator.choice_template

    # to static model
    fix_mutable = export_fix_subnet(algorithm.architecture)[0]
    load_fix_subnet(algorithm.architecture, fix_mutable)
    model = algorithm.architecture
    return model, pruning_structure


if __name__ == '__main__':
    # init
    args = parse_args()
    config_path = args.config
    checkpoint_path = args.checkpoint
    target_path = args.o

    work_dir, checkpoint_name = get_save_path(config_path, checkpoint_path,
                                              target_path)
    os.makedirs(work_dir, exist_ok=True)

    # build model
    config = Config.fromfile(config_path)
    model = MODELS.build(config.model)
    assert isinstance(model, BaseAlgorithm), 'Model must be a BaseAlgorithm'
    load_checkpoint(model, checkpoint_path, map_location='cpu')

    pruned_model, structure = get_static_model(model)

    # save
    torch.save(pruned_model.state_dict(),
               os.path.join(work_dir, checkpoint_name + '.pth'))
    fileio.dump(
        structure, os.path.join(work_dir, checkpoint_name + '.json'), indent=4)

    print_log('Save pruned model to {}'.format(work_dir))
    print_log('Pruning Structure: {}'.format(json.dumps(structure, indent=4)))
