import argparse
import os
import os.path as osp
import re
from collections import OrderedDict
from pathlib import Path

import mmcv
from modelindex.load_model_index import load
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

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
        default='work_dirs/benchmark_test',
        help='the dir to save metric')
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


def create_train_job_batch(commands, model_info, args, port):

    fname = model_info.name

    config = Path(model_info.config)
    # assert config.exists(), f'{fname}: {config} not found.'

    job_name = f'{args.job_name}_{fname}'
    work_dir = Path(args.work_dir) / fname
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.quotatype is not None:
        quota_cfg = f'#SBATCH --quotatype {args.quotatype}\n'
    else:
        quota_cfg = ''

    launcher = 'none' if args.local else 'slurm'
    runner = 'python' if args.local else 'srun python'
    master_port = f'NASTER_PORT={port}'

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
                  f'{master_port} {runner} -u {script_name} {config} '
                  f'--work-dir {work_dir} '
                  f'--launcher={launcher}\n')

    with open(work_dir / 'job.sh', 'w') as f:
        f.write(job_script)

    commands.append(f'echo "{config}"')
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

    mmcv.fileio.dump(model_results,
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
