import argparse
import logging
import os
import os.path as osp
import re
from collections import OrderedDict
from pathlib import Path

import mmcv
import mmengine
from mmengine.logging import print_log
from modelindex.load_model_index import load
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from mmrazor.testing import FastStopTrainingHook  # noqa: F401

os.environ['MKL_THREADING_LAYER'] = 'GNU'

console = Console()
MMRAZOR_ROOT = Path(__file__).absolute().parents[1]

METRIC_MAPPINGS = {
    'accuracy/top1': 'Top 1 Accuracy',
    'accuracy/top5': 'Top 5 Accuracy'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test all models' accuracy in model-index.yml")
    parser.add_argument(
        'partition', type=str, help='Cluster partition to use.')
    parser.add_argument(
        '--job-name',
        type=str,
        default='razor-train-benchmark',
        help='Slurm job name prefix')
    parser.add_argument('--port', type=int, default=29666, help='dist port')
    parser.add_argument(
        '--models', nargs='+', type=str, help='Specify model names to run.')
    parser.add_argument('--gpus', type=int, default=8, help='num gpus')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/benchmark_train',
        help='the dir to save metric')
    parser.add_argument('--amp', action='store_true', help='use amp')
    parser.add_argument(
        '--auto-scale-lr', action='store_true', help='use auto scale lr')
    parser.add_argument(
        '--auto-resume', action='store_true', help='use auto resume')
    parser.add_argument(
        '--replace-ceph', action='store_true', help='load data from ceph')
    parser.add_argument(
        '--early-stop', action='store_true', help='early stop training')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--summary', action='store_true', help='collect results')
    parser.add_argument(
        '--local',
        action='store_true',
        help='run at local instead of cluster.')
    parser.add_argument(
        '--mail', type=str, help='Mail address to watch test status.')
    parser.add_argument(
        '--mail-type',
        nargs='+',
        default=['BEGIN'],
        choices=['NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'],
        help='Mail address to watch test status.')
    parser.add_argument(
        '--quotatype',
        default=None,
        choices=['reserved', 'auto', 'spot'],
        help='Quota type, only available for phoenix-slurm>=0.2')

    args = parser.parse_args()
    return args


def replace_to_ceph(cfg):

    file_client_args = dict(
        backend='petrel',
        path_mapping=dict({
            './data/coco':
            's3://openmmlab/datasets/detection/coco',
            'data/coco':
            's3://openmmlab/datasets/detection/coco',
            './data/cityscapes':
            's3://openmmlab/datasets/segmentation/cityscapes',
            'data/cityscapes':
            's3://openmmlab/datasets/segmentation/cityscapes',
            './data/imagenet':
            's3://openmmlab/datasets/classification/imagenet',
            'data/imagenet':
            's3://openmmlab/datasets/classification/imagenet',
        }))

    def _process_pipeline(dataset, name):

        def replace_img(pipeline):
            if pipeline['type'] == 'LoadImageFromFile':
                pipeline['file_client_args'] = file_client_args

        def replace_ann(pipeline):
            if pipeline['type'] == 'LoadAnnotations' or pipeline[
                    'type'] == 'LoadPanopticAnnotations':
                pipeline['file_client_args'] = file_client_args

        if 'pipeline' in dataset:
            replace_img(dataset.pipeline[0])
            replace_ann(dataset.pipeline[1])
            if 'dataset' in dataset:
                # dataset wrapper
                replace_img(dataset.dataset.pipeline[0])
                replace_ann(dataset.dataset.pipeline[1])
        else:
            # dataset wrapper
            replace_img(dataset.dataset.pipeline[0])
            replace_ann(dataset.dataset.pipeline[1])

    def _process_evaluator(evaluator, name):
        if evaluator['type'] == 'CocoPanopticMetric':
            evaluator['file_client_args'] = file_client_args

    # half ceph
    _process_pipeline(cfg.train_dataloader.dataset, cfg.filename)
    _process_pipeline(cfg.val_dataloader.dataset, cfg.filename)
    _process_pipeline(cfg.test_dataloader.dataset, cfg.filename)
    _process_evaluator(cfg.val_evaluator, cfg.filename)
    _process_evaluator(cfg.test_evaluator, cfg.filename)


def create_train_job_batch(commands, model_info, args, port):

    fname = model_info.name

    cfg_path = Path(model_info.config)

    cfg = mmengine.Config.fromfile(cfg_path)

    if args.replace_ceph:
        replace_to_ceph(cfg)

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.auto_resume:
        cfg.resume = True

    if args.early_stop:
        if 'custom_hooks' in cfg:
            cfg.custom_hooks.append(dict(type='mmrazor.FastStopTrainingHook'))
        else:
            custom_hooks = [dict(type='mmrazor.FastStopTrainingHook')]
            cfg.custom_hooks = custom_hooks

    job_name = f'{args.job_name}_{fname}'
    work_dir = Path(args.work_dir) / fname
    work_dir.mkdir(parents=True, exist_ok=True)

    train_cfg_path = work_dir / 'config.py'
    cfg.dump(train_cfg_path)

    if args.quotatype is not None:
        quota_cfg = f'#SBATCH --quotatype {args.quotatype}\n'
    else:
        quota_cfg = ''

    launcher = 'none' if args.local else 'slurm'
    runner = 'python' if args.local else 'srun python'
    master_port = f'MASTER_PORT={port}'

    script_name = osp.join('tools', 'train.py')
    job_script = (f'#!/bin/bash\n'
                  f'#SBATCH --output {work_dir}/job.%j.out\n'
                  f'#SBATCH --partition={args.partition}\n'
                  f'#SBATCH --job-name {job_name}\n'
                  f'#SBATCH --gres=gpu:{args.gpus}\n'
                  f'{quota_cfg}'
                  f'#SBATCH --ntasks-per-node={args.gpus}\n'
                  f'#SBATCH --ntasks={args.gpus}\n'
                  f'#SBATCH --cpus-per-task=5\n\n'
                  f'{master_port} {runner} -u {script_name} {train_cfg_path} '
                  f'--work-dir {work_dir} '
                  f'--launcher={launcher}\n')

    with open(work_dir / 'job.sh', 'w') as f:
        f.write(job_script)

    commands.append(f'echo "{train_cfg_path}"')
    if args.local:
        commands.append(f'bash {work_dir}/job.sh')
    else:
        commands.append(f'sbatch {work_dir}/job.sh')

    return work_dir / 'job.sh'


def summary(args):
    # parse model-index.yml
    model_index_file = MMRAZOR_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    if args.models:
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    model_results = dict()
    for model_info in models.values():
        model_name = model_info.name
        work_dir = Path(args.work_dir) / model_name
        sub_dirs = [p.name for p in work_dir.iterdir() if p.is_dir()]

        if len(sub_dirs) == 0:
            print(f'{model_name} has no results.')
            continue

        latest_time = sub_dirs[-1]
        latest_json = work_dir / latest_time / f'{latest_time}.json'

        if not latest_json.exists():
            print(f'{model_name} has no results.')
            continue
        latest_result = mmcv.load(latest_json, 'json')

        expect_result = model_info.results[0].metrics
        summary_result = {
            'expect': expect_result,
            'actual':
            {METRIC_MAPPINGS[k]: v
             for k, v in latest_result.items()}
        }
        model_results[model_name] = summary_result

    mmengine.fileio.dump(model_results,
                         Path(args.work_dir) / 'summary.yml', 'yaml')
    print(f'Summary results saved in {Path(args.work_dir)}/summary.yml')


def train(args):
    # parse model-index.yml
    model_index_file = MMRAZOR_ROOT / 'model-index.yml'
    model_index = load(str(model_index_file))
    model_index.build_models_with_collections()
    models = OrderedDict({model.name: model for model in model_index.models})

    commands = []
    if args.models:
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    preview_script = ''
    port = args.port
    for model_info in models.values():
        script_path = create_train_job_batch(commands, model_info, args, port)
        preview_script = script_path or preview_script
        port += 1
    command_str = '\n'.join(commands)

    preview = Table()
    preview.add_column(str(preview_script))
    preview.add_column('Shell command preview')
    preview.add_row(
        Syntax.from_path(
            preview_script,
            background_color='default',
            line_numbers=True,
            word_wrap=True),
        Syntax(
            command_str,
            'bash',
            background_color='default',
            line_numbers=True,
            word_wrap=True))
    console.print(preview)

    if args.run:
        os.system(command_str)
    else:
        console.print('Please set "--run" to start the job')


def main():
    args = parse_args()
    if args.summary:
        summary(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
