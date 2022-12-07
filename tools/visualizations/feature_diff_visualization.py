# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch
from mmengine.config import Config
from mmengine.registry import VISUALIZERS
from mmengine.utils import import_modules_from_strings

from mmrazor.models.task_modules import RecorderManager
from mmrazor.utils import register_all_modules
from mmrazor.visualization.local_visualizer import modify


def parse_args():
    parser = argparse.ArgumentParser(description='Feature map visualization')
    parser.add_argument('img', help='Image file')
    parser.add_argument(
        'config1', help='train config file path for the first model')
    parser.add_argument(
        'config2', help='train config file path for the second model')
    parser.add_argument('vis_config', help='visualization config file path')
    parser.add_argument(
        'checkpoint1', help='Checkpoint file for the first model')
    parser.add_argument(
        'checkpoint2', help='Checkpoint file for the second model')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument('--repo', help='the corresponding repo name')
    parser.add_argument(
        '--use-norm',
        action='store_true',
        help='normalize the featmap before visualization')
    parser.add_argument(
        '--overlaid', action='store_true', help='overlaid image')
    parser.add_argument(
        '--channel-reduction',
        help='Reduce multiple channels to a single channel. The optional value'
        ' is \'squeeze_mean\', \'select_max\' or \'pixel_wise_max\'.',
        default=None)
    parser.add_argument(
        '--topk',
        help='If channel_reduction is not None and topk > 0, it will select '
        'topk channel to show by the sum of each channel. If topk <= 0, '
        'tensor_chw is assert to be one or three.',
        type=int,
        default=20)
    parser.add_argument(
        '--arrangement',
        nargs='+',
        type=int,
        help='the arrangement of featmap when channel_reduction is not None '
        'and topk > 0.',
        default=[4, 5])
    parser.add_argument(
        '--resize-shape',
        nargs='+',
        type=int,
        help='the shape to scale the feature map',
        default=None)
    parser.add_argument(
        '--alpha', help='the transparency of featmap', default=0.5)

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def norm(feat):
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    centered = (feat - mean) / (std + 1e-6)
    centered = centered.reshape(C, N, H, W).permute(1, 0, 2, 3)
    return centered


def main(args):
    register_all_modules(False)
    mod = import_modules_from_strings(f'{args.repo}.utils')
    mod.register_all_modules()

    apis = import_modules_from_strings(f'{args.repo}.apis')
    inference_model, init_model = None, None
    for attr_name in dir(apis):
        if 'inference_' in attr_name:
            inference_model = getattr(apis, attr_name)
        if 'init_' in attr_name:
            init_model = getattr(apis, attr_name)
    assert inference_model and init_model

    model1 = init_model(args.config1, args.checkpoint1, device=args.device)
    # init visualizer
    visualizer = VISUALIZERS.build(model1.cfg.visualizer)
    visualizer.draw_featmap = modify

    model2 = init_model(args.config2, args.checkpoint2, device=args.device)

    visualization_cfg = Config.fromfile(args.vis_config)
    recorder_cfg1 = visualization_cfg.recorders1
    mappings1 = visualization_cfg.mappings1
    recorder_cfg2 = visualization_cfg.recorders2
    mappings2 = visualization_cfg.mappings2

    recorder_manager1 = RecorderManager(recorder_cfg1)
    recorder_manager1.initialize(model1)

    recorder_manager2 = RecorderManager(recorder_cfg2)
    recorder_manager2.initialize(model2)

    with recorder_manager1:
        # test a single image
        _ = inference_model(model1, args.img)

    with recorder_manager2:
        # test a single image
        _ = inference_model(model2, args.img)

    overlaid_image = mmcv.imread(
        args.img, channel_order='rgb') if args.overlaid else None

    for name1, name2 in zip(mappings1.keys(), mappings2.keys()):
        record1 = mappings1[name1]
        recorder1 = recorder_manager1.get_recorder(record1.recorder)
        record_idx = getattr(record1, 'record_idx', 0)
        data_idx = getattr(record1, 'data_idx')
        feats1 = recorder1.get_record_data(record_idx, data_idx)
        if isinstance(feats1, torch.Tensor):
            feats1 = (feats1, )

        record2 = mappings2[name2]
        recorder2 = recorder_manager2.get_recorder(record2.recorder)
        record_idx = getattr(record2, 'record_idx', 0)
        data_idx = getattr(record2, 'data_idx')
        feats2 = recorder2.get_record_data(record_idx, data_idx)
        if isinstance(feats2, torch.Tensor):
            feats2 = (feats2, )

        for i, (feat1, feat2) in enumerate(zip(feats1, feats2)):
            diff = torch.abs(feat1 - feat2)
            if args.use_norm:
                diff = norm(diff)
            drawn_img = visualizer.draw_featmap(
                diff[0],
                overlaid_image,
                args.channel_reduction,
                topk=args.topk,
                arrangement=tuple(args.arrangement),
                resize_shape=tuple(args.resize_shape)
                if args.resize_shape else None,
                alpha=args.alpha)
            visualizer.add_datasample(
                f'model1_{name1}_model2_{name2}_{i}',
                drawn_img,
                show=args.out_file is None,
                wait_time=0.1,
                out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
