# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json

import torch.nn as nn
from mmengine import MODELS
from mmengine.config import Config

from mmrazor.models import BaseAlgorithm
from mmrazor.models.mutators import BaseChannelMutator


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get channel group of a model.')
    parser.add_argument('config', help='config of the model')
    parser.add_argument(
        '-c',
        '--with-channel',
        action='store_true',
        help='output with channel config')
    parser.add_argument(
        '-i',
        '--with-init-args',
        action='store_true',
        help='output with init args')
    parser.add_argument(
        '-o',
        '--output-path',
        default='',
        help='the file path to store channel group info')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    model = MODELS.build(config['model'])
    if isinstance(model, BaseAlgorithm):
        mutator = model.mutator
    elif isinstance(model, nn.module):
        mutator = BaseChannelMutator()
        mutator.prepare_from_supernet(model)
    config = mutator.config_template(
        with_channels=args.with_channel,
        with_group_init_args=args.with_init_args)
    json_config = json.dumps(config, indent=4, separators=(',', ':'))
    if args.output_path == '':
        print('=' * 100)
        print('config template')
        print('=' * 100)
        print(json_config)
    else:
        with open(args.output_path, 'w') as file:
            file.write(json_config)


if __name__ == '__main__':
    main()
