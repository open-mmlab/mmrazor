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
        key = key.replace('module.', 'architecture.backbone.')
        if 'blocks.10' in key:
            new_key = key.replace('blocks.10', 'layer3.3')
        elif 'blocks.11' in key:
            new_key = key.replace('blocks.11', 'layer3.4')
        elif 'blocks.12' in key:
            new_key = key.replace('blocks.12', 'layer3.5')
        elif 'blocks.13' in key:
            new_key = key.replace('blocks.13', 'layer4.0')
        elif 'blocks.14' in key:
            new_key = key.replace('blocks.14', 'layer4.1')
        elif 'blocks.15' in key:
            new_key = key.replace('blocks.15', 'layer4.2')
        elif 'blocks.16' in key:
            new_key = key.replace('blocks.16', 'layer4.3')
        elif 'blocks.17' in key:
            new_key = key.replace('blocks.17', 'layer4.4')
        elif 'blocks.18' in key:
            new_key = key.replace('blocks.18', 'layer4.5')
        elif 'blocks.19' in key:
            new_key = key.replace('blocks.19', 'layer5.0')
        elif 'blocks.20' in key:
            new_key = key.replace('blocks.20', 'layer5.1')
        elif 'blocks.21' in key:
            new_key = key.replace('blocks.21', 'layer5.2')
        elif 'blocks.22' in key:
            new_key = key.replace('blocks.22', 'layer5.3')
        elif 'blocks.23' in key:
            new_key = key.replace('blocks.23', 'layer5.4')
        elif 'blocks.24' in key:
            new_key = key.replace('blocks.24', 'layer5.5')
        elif 'blocks.25' in key:
            new_key = key.replace('blocks.25', 'layer5.6')
        elif 'blocks.26' in key:
            new_key = key.replace('blocks.26', 'layer5.7')
        elif 'blocks.27' in key:
            new_key = key.replace('blocks.27', 'layer6.0')
        elif 'blocks.28' in key:
            new_key = key.replace('blocks.28', 'layer6.1')
        elif 'blocks.29' in key:
            new_key = key.replace('blocks.29', 'layer6.2')
        elif 'blocks.30' in key:
            new_key = key.replace('blocks.30', 'layer6.3')
        elif 'blocks.31' in key:
            new_key = key.replace('blocks.31', 'layer6.4')
        elif 'blocks.32' in key:
            new_key = key.replace('blocks.32', 'layer6.5')
        elif 'blocks.33' in key:
            new_key = key.replace('blocks.33', 'layer6.6')
        elif 'blocks.34' in key:
            new_key = key.replace('blocks.34', 'layer6.7')
        elif 'blocks.35' in key:
            new_key = key.replace('blocks.35', 'layer7.0')
        elif 'blocks.36' in key:
            new_key = key.replace('blocks.36', 'layer7.1')
        elif 'blocks.0' in key:
            new_key = key.replace('blocks.0', 'layer1.0')
        elif 'blocks.1' in key:
            new_key = key.replace('blocks.1', 'layer1.1')
        elif 'blocks.2' in key:
            new_key = key.replace('blocks.2', 'layer2.0')
        elif 'blocks.3' in key:
            new_key = key.replace('blocks.3', 'layer2.1')
        elif 'blocks.4' in key:
            new_key = key.replace('blocks.4', 'layer2.2')
        elif 'blocks.5' in key:
            new_key = key.replace('blocks.5', 'layer2.3')
        elif 'blocks.6' in key:
            new_key = key.replace('blocks.6', 'layer2.4')
        elif 'blocks.7' in key:
            new_key = key.replace('blocks.7', 'layer3.0')
        elif 'blocks.8' in key:
            new_key = key.replace('blocks.8', 'layer3.1')
        elif 'blocks.9' in key:
            new_key = key.replace('blocks.9', 'layer3.2')
        else:
            new_key = key

        if 'mobile_inverted_conv.depth_conv.conv.conv' in new_key:
            final_new_key = new_key.replace(
                'mobile_inverted_conv.depth_conv.conv.conv',
                'depthwise_conv.conv')
        elif 'mobile_inverted_conv.depth_conv.bn.bn' in new_key:
            final_new_key = new_key.replace(
                'mobile_inverted_conv.depth_conv.bn.bn', 'depthwise_conv.bn')
        elif 'mobile_inverted_conv.point_linear.conv.conv' in new_key:
            final_new_key = new_key.replace(
                'mobile_inverted_conv.point_linear.conv.conv',
                'linear_conv.conv')
        elif 'mobile_inverted_conv.point_linear.bn.bn' in new_key:
            final_new_key = new_key.replace(
                'mobile_inverted_conv.point_linear.bn.bn', 'linear_conv.bn')
        elif 'shortcut.conv.conv' in new_key:
            final_new_key = new_key.replace('shortcut.conv.conv',
                                            'shortcut.conv')
        elif 'mobile_inverted_conv.inverted_bottleneck.conv.conv' in new_key:
            final_new_key = new_key.replace(
                'mobile_inverted_conv.inverted_bottleneck.conv.conv',
                'expand_conv.conv')
        elif 'mobile_inverted_conv.inverted_bottleneck.bn.bn' in new_key:
            final_new_key = new_key.replace(
                'mobile_inverted_conv.inverted_bottleneck.bn.bn',
                'expand_conv.bn')
        elif 'mobile_inverted_conv.depth_conv.se.fc.reduce' in new_key:
            final_new_key = new_key.replace(
                'mobile_inverted_conv.depth_conv.se.fc.reduce',
                'se.conv1.conv')
        elif 'mobile_inverted_conv.depth_conv.se.fc.expand' in new_key:
            final_new_key = new_key.replace(
                'mobile_inverted_conv.depth_conv.se.fc.expand',
                'se.conv2.conv')
        elif 'first_conv.conv.conv' in new_key:
            final_new_key = new_key.replace('first_conv.conv.conv',
                                            'first_conv.conv')
        elif 'first_conv.bn.bn' in new_key:
            final_new_key = new_key.replace('first_conv.bn.bn',
                                            'first_conv.bn')
        elif 'final_expand_layer.conv.conv' in new_key:
            final_new_key = new_key.replace('final_expand_layer.conv.conv',
                                            'final_expand_layer.conv')
        elif 'final_expand_layer.bn.bn' in new_key:
            final_new_key = new_key.replace('final_expand_layer.bn.bn',
                                            'final_expand_layer.bn')
        elif 'feature_mix_layer.conv.conv' in new_key:
            final_new_key = new_key.replace('feature_mix_layer.conv.conv',
                                            'feature_mix_layer.conv')
        elif 'classifier.linear.linear' in new_key:
            final_new_key = new_key.replace(
                'backbone.classifier.linear.linear', 'head.fc')
        else:
            final_new_key = new_key

        new_state_dict[final_new_key] = value

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
