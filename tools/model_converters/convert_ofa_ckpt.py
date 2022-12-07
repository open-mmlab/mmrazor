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
        if 'blocks.0.' in key:
            new_key = key.replace('blocks.0', 'layer1.0')
        elif 'blocks' in key:
            for i in range(1, 21):
                if 'blocks.' + str(i) in key:
                    new_key = key.replace(
                        'blocks.' + str(i),
                        'layer' + str(i // 4 + 1) + '.' + str((i + 3) % 4))
        else:
            new_key = key

        if 'mobile_inverted_conv' in new_key:
            new_key = new_key.replace('mobile_inverted_conv.', '')
        if 'depth_conv' in key:
            new_key = new_key.replace('depth_conv', 'depthwise_conv')
        if 'point_linear' in key:
            new_key = new_key.replace('point_linear', 'linear_conv')
        if 'inverted_bottleneck' in key:
            new_key = new_key.replace('inverted_bottleneck', 'expand_conv')
        if '.conv.conv' in new_key:
            new_key = new_key.replace('.conv.conv', '.conv')
        if '.bn.bn' in new_key:
            new_key = new_key.replace('.bn.bn', '.bn')

        if 'layer1.0.depthwise_conv.weight' in new_key:
            new_key = new_key.replace('layer1.0.depthwise_conv.weight',
                                      'layer1.0.depthwise_conv.conv.weight')
        if 'layer1.0.linear_conv.weight' in new_key:
            new_key = new_key.replace('layer1.0.linear_conv.weight',
                                      'layer1.0.linear_conv.conv.weight')

        if 'depthwise_conv.se.fc.reduce' in new_key:
            new_key = new_key.replace('depthwise_conv.se.fc.reduce',
                                      'se.conv1.conv')
        if 'depthwise_conv.se.fc.expand' in new_key:
            new_key = new_key.replace('depthwise_conv.se.fc.expand',
                                      'se.conv2.conv')

        if 'final_expand_layer' in new_key:
            new_key = new_key.replace('final_expand_layer',
                                      'last_conv.final_expand_layer')
        if 'feature_mix_layer' in new_key:
            new_key = new_key.replace('feature_mix_layer',
                                      'last_conv.feature_mix_layer')

        if '5to3_matrix' in new_key:
            new_key = new_key.replace('5to3_matrix', 'trans_matrix_5to3')
        if '7to5_matrix' in new_key:
            new_key = new_key.replace('7to5_matrix', 'trans_matrix_7to5')

        new_key = 'architecture.backbone.' + new_key

        if 'classifier.linear' in new_key:
            new_key = new_key.replace('classifier.linear', 'head.fc')
            new_key = new_key.replace('backbone.', '')

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
