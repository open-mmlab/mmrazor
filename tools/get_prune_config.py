# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
from typing import Dict

from mmengine import Config

from mmrazor.models.mutators import ChannelMutator
from mmrazor.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser(
        description='Get the config to prune a model.')
    parser.add_argument('config', help='config of the model')
    parser.add_argument(
        '--checkpoint',
        default=None,
        type=str,
        help='checkpoint path of the model')
    parser.add_argument(
        '-o',
        type=str,
        default='./prune.py',
        help='output path to store the pruning config.')
    args = parser.parse_args()
    return args


def wrap_prune_config(config: Config, prune_target: Dict,
                      checkpoint_path: str):
    config = copy.deepcopy(config)

    arch_config: Dict = config['model']

    # update checkpoint_path
    if checkpoint_path is not None:
        arch_config.update({
            'init_cfg': {
                'type': 'Pretrained',
                'checkpoint': checkpoint_path  # noqa
            },
        })

    # deal with data_preprocessor
    if 'data_preprocessor' in config:
        data_preprocessor = config['data_preprocessor']
        arch_config.update({'data_preprocessor': data_preprocessor})
        config['data_preprocessor'] = None
    else:
        data_preprocessor = None

    # prepare algorithm
    algorithm_config = dict(
        _scope_='mmrazor',
        type='ItePruneAlgorithm',
        architecture=arch_config,
        target_pruning_ratio=prune_target,
        mutator_cfg=dict(
            type='ChannelMutator',
            channel_unit_cfg=dict(
                type='L1MutableChannelUnit',
                default_args=dict(choice_mode='ratio')),
            parse_cfg=dict(type='PruneTracer', tracer_type='FxTracer')))
    config['model'] = algorithm_config

    return config


if __name__ == '__main__':
    args = parse_args()
    config_path = args.config
    checkpoint_path = args.checkpoint
    target_path = args.o

    origin_config = Config.fromfile(config_path)

    # get subnet config
    model = MODELS.build(origin_config['model'])
    mutator: ChannelMutator = ChannelMutator(
        channel_unit_cfg=dict(
            type='L1MutableChannelUnit',
            default_args=dict(choice_mode='ratio'),
        ),
        parse_cfg={
            'type': 'PruneTracer',
            'tracer_type': 'FxTracer'
        })
    mutator.prepare_from_supernet(model)
    choice_template = mutator.choice_template

    # prune and finetune

    prune_config: Config = wrap_prune_config(origin_config, choice_template,
                                             checkpoint_path)
    prune_config.dump(target_path)
