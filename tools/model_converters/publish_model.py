# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import shutil
from pathlib import Path
from typing import Union

import torch
from mmengine import digit_version


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('ckpt', help='input checkpoint filename', type=str)
    parser.add_argument('--model-name', help='model(config) name', type=str)
    parser.add_argument('--timestamp', help='training timestamp', type=str)
    parser.add_argument('--out-dir', help='output dir', type=str)
    args = parser.parse_args()
    return args


def cal_file_sha256(file_path: Union[str, Path]) -> str:
    import hashlib

    BLOCKSIZE = 65536
    sha256_hash = hashlib.sha256()

    with open(file_path, 'rb') as f:
        block = f.read(BLOCKSIZE)
        while block:
            sha256_hash.update(block)
            block = f.read(BLOCKSIZE)

    return sha256_hash.hexdigest()


def process_checkpoint(ckpt_path_str: str, model_name: str, timestamp: str,
                       out_dir_str: str) -> None:

    ckpt_path = Path(ckpt_path_str)
    work_dir = ckpt_path.parent

    out_dir: Path = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_ckpt_path = out_dir / 'tmp.pth'

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # remove message_hub for smaller file size
    if 'message_hub' in checkpoint:
        del checkpoint['message_hub']
    # remove param_schedulers for smaller file size
    if 'param_schedulers' in checkpoint:
        del checkpoint['param_schedulers']

    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    if digit_version(torch.__version__) >= digit_version('1.6'):
        torch.save(
            checkpoint, tmp_ckpt_path, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, tmp_ckpt_path)

    sha = cal_file_sha256(tmp_ckpt_path)
    save_ckpt_path = f'{out_dir}/{model_name}_{timestamp}-{sha[:8]}.pth'
    tmp_ckpt_path.rename(save_ckpt_path)
    print(f'Successfully generated the publish-ckpt as {save_ckpt_path}.')

    log_path = work_dir / timestamp / f'{timestamp}.log'
    save_log_path = f'{out_dir}/{model_name}_{timestamp}-{sha[:8]}.log'
    shutil.copy(str(log_path), str(save_log_path))
    print(f'Successfully generated the publish-log as {save_log_path}.')

    log_path = work_dir / timestamp / f'{timestamp}.log'
    json_path = work_dir / timestamp / f'vis_data/{timestamp}.json'
    save_json_path = f'{out_dir}/{model_name}_{timestamp}-{sha[:8]}.json'
    shutil.copy(str(json_path), str(save_json_path))
    print(f'Successfully generated the publish-log as {save_json_path}.')


def main():
    args = parse_args()
    process_checkpoint(args.ckpt, args.model_name, args.timestamp,
                       args.out_dir)


if __name__ == '__main__':
    main()
