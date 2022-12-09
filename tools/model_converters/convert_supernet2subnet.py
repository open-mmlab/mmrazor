# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

from mmengine.config import Config
from mmengine.runner import Runner

from mmrazor.models.architectures.utils import export_subnet_checkpoint
from mmrazor.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a NAS supernet checkpoint to be converted')
    parser.add_argument('config', help='NAS model config file path')
    parser.add_argument('checkpoint', help='supernet checkpoint file path')
    parser.add_argument('yaml', help='YAML with subnet settings file path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    args = parser.parse_args()
    return args


def main():
    register_all_modules(False)
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    cfg.load_from = args.checkpoint
    cfg.work_dir = '/'.join(args.checkpoint.split('/')[:-1])

    runner = Runner.from_cfg(cfg)

    export_subnet_checkpoint(
        runner.model, fix_subnet=args.yaml, prefix=cfg.work_dir)


if __name__ == '__main__':
    main()
