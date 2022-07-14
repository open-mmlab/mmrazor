# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process a checkpoint to be published')
    parser.add_argument('checkpoint', help='input checkpoint filename')
    parser.add_argument(
        '--inplace', action='store_true', help='replace origin ckpt')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    new_state_dict = dict()

    for key, value in checkpoint['state_dict'].items():
        if key.startswith('architecture.model.distiller.teacher'):
            new_key = key.replace('architecture.model.distiller.teacher',
                                  'architecture.teacher')
        elif key.startswith('architecture.model'):
            new_key = key.replace('architecture.model', 'architecture')
        else:
            new_key = key

        new_state_dict[new_key] = value

    checkpoint['state_dict'] = new_state_dict

    if args.inplace:
        torch.save(checkpoint, args.checkpoint)
    else:
        ckpt_path = Path(args.checkpoint)
        ckpt_name = ckpt_path.stem
        ckpt_dir = ckpt_path.parent
        new_ckpt_path = ckpt_dir / f'{ckpt_name}_latest.pth'
        torch.save(checkpoint, new_ckpt_path)


if __name__ == '__main__':
    main()
