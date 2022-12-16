# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import sys

import torch.nn as nn
from mmengine import MODELS
from mmengine.config import Config

from mmrazor.models import BaseAlgorithm
from mmrazor.models.mutators import ChannelMutator

sys.setrecursionlimit(int(pow(2, 20)))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get channel unit of a model.')
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
        '--choice',
        action='store_true',
        help=('output choices template. When this flag is activated, '
              '-c and -i will be ignored'))
    parser.add_argument(
        '-o',
        '--output-path',
        default='',
        help='the file path to store channel unit info')
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    default_scope = config['default_scope']

    model = MODELS.build(config['model'])
    if isinstance(model, BaseAlgorithm):
        mutator = model.mutator
    elif isinstance(model, nn.Module):
        mutator: ChannelMutator = ChannelMutator(
            channel_unit_cfg=dict(
                type='L1MutableChannelUnit',
                default_args=dict(choice_mode='ratio'),
            ),
            parse_cfg={
                'type': 'ChannelAnalyzer',
                'demo_input': {
                    'type': 'DefaultDemoInput',
                    'scope': default_scope
                },
                'tracer_type': 'FxTracer'
            })
        mutator.prepare_from_supernet(model)
    if args.choice:
        config = mutator.choice_template
    else:
        config = mutator.config_template(
            with_channels=args.with_channel,
            with_unit_init_args=args.with_init_args)
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
