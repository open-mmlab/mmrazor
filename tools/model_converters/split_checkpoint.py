# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp

from mmcv import Config
from mmcv.runner import load_checkpoint, save_checkpoint

from mmrazor.models import build_algorithm
from mmrazor.models.pruners.utils import SwitchableBatchNorm2d


def parse_args():
    parser = argparse.ArgumentParser(description='Split a slimmable trained'
                                     'model checkpoint')
    parser.add_argument('config', type=str, help='path of train config file')
    parser.add_argument('checkpoint', type=str, help='checkpoint path')
    parser.add_argument(
        '--channel-cfgs',
        nargs='+',
        help='The path of the channel configs. '
        'The order should be the same as that of train.')
    parser.add_argument('--output-dir', type=str, default='')
    args = parser.parse_args()

    return args


def convert_bn(module, bn_ind):

    def traverse(module):
        for name, child in module.named_children():
            if isinstance(child, SwitchableBatchNorm2d):
                setattr(module, name, child.bns[bn_ind])
            else:
                traverse(child)

    traverse(module)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(dict(algorithm=dict(channel_cfg=args.channel_cfgs)))

    for i, channel_cfg in enumerate(args.channel_cfgs):
        algorithm = build_algorithm(cfg.algorithm)
        load_checkpoint(algorithm, args.checkpoint, map_location='cpu')
        convert_bn(algorithm, i)
        for module in algorithm.modules():
            if hasattr(module, 'out_mask'):
                del module.out_mask
            if hasattr(module, 'in_mask'):
                del module.in_mask
        assert algorithm.with_pruner, \
            'The algorithm should has attr pruner. Please check your ' \
            'config file.'
        algorithm.pruner.deploy_subnet(algorithm.architecture,
                                       algorithm.channel_cfg[i])
        filename = osp.join(args.output_dir, f'checkpoint_{i + 1}.pth')
        save_checkpoint(algorithm, filename)

    print(f'Successfully split the original checkpoint `{args.checkpoint}` to '
          f'{len(args.channel_cfgs)} different checkpoints.')


if __name__ == '__main__':
    main()
