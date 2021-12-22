# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import datetime
from pathlib import Path

import mmcv
import torch
from mmcv import digit_version


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('in_file', help='input checkpoint filename')
    parser.add_argument('out_file', help='output checkpoint filename')
    parser.add_argument('--mutable-cfg', help='input mutable cfg filename')
    parser.add_argument('--channel-cfg', help='output channel cfg filename')
    args = parser.parse_args()
    return args


def cal_file_sha256(file_path: str) -> str:
    import hashlib

    BLOCKSIZE = 65536
    sha256_hash = hashlib.sha256()

    with open(file_path, 'rb') as f:
        block = f.read(BLOCKSIZE)
        while block:
            sha256_hash.update(block)
            block = f.read(BLOCKSIZE)

    return sha256_hash.hexdigest()


def process_checkpoint(in_file,
                       out_file,
                       mutable_cfg_file=None,
                       channel_cfg_file=None):
    checkpoint = torch.load(in_file, map_location='cpu')
    # remove optimizer for smaller file size
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    # if it is necessary to remove some sensitive data in checkpoint['meta'],
    # add the code here.
    if digit_version(torch.__version__) >= digit_version('1.6'):
        torch.save(checkpoint, out_file, _use_new_zipfile_serialization=False)
    else:
        torch.save(checkpoint, out_file)

    sha = cal_file_sha256(out_file)
    if out_file.endswith('.pth'):
        out_file_name = out_file[:-4]
    else:
        out_file_name = out_file

    current_date = datetime.datetime.now().strftime('%Y%m%d')
    final_file_prefix = out_file_name + f'_{current_date}-{sha[:8]}'
    final_ckpt_file = f'{final_file_prefix}.pth'
    Path(out_file).rename(final_ckpt_file)

    print(f'Successfully generated the publish-ckpt as {final_ckpt_file}.')

    if mutable_cfg_file:
        mutable_cfg = mmcv.fileio.load(mutable_cfg_file)
        final_mutable_cfg_file = f'{final_file_prefix}_mutable_cfg.yaml'
        mmcv.fileio.dump(mutable_cfg, final_mutable_cfg_file)
        print(f'Successfully generated the publish-mutable-cfg as \
                {final_mutable_cfg_file}.')

    if channel_cfg_file:
        channel_cfg = mmcv.fileio.load(channel_cfg_file)
        final_channel_cfg_file = f'{final_file_prefix}_channel_cfg.yaml'
        mmcv.fileio.dump(channel_cfg, final_channel_cfg_file)
        print(f'Successfully generated the publish-channel-cfg as \
                {final_channel_cfg_file}.')


def main():
    args = parse_args()
    out_dir = Path(args.out_file).parent
    if not out_dir.exists():
        raise ValueError(f'Directory {out_dir} does not exist, '
                         'please generate it manually.')
    process_checkpoint(args.in_file, args.out_file, args.mutable_cfg,
                       args.channel_cfg)


if __name__ == '__main__':
    main()
