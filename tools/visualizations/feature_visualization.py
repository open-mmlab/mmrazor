# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch
from mmengine.config import Config, DictAction
from mmengine.registry import VISUALIZERS
from mmengine.utils import import_modules_from_strings

from mmrazor.models.task_modules import RecorderManager
from mmrazor.utils import register_all_modules
from mmrazor.visualization.local_visualizer import modify


def parse_args():
    parser = argparse.ArgumentParser(description='Feature map visualization')
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('vis_config', help='visualization config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
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
        type=int,
        help='If channel_reduction is not None and topk > 0, it will select '
        'topk channel to show by the sum of each channel. If topk <= 0, '
        'tensor_chw is assert to be one or three.',
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
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.',
        default={})

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

    model = init_model(args.config, args.checkpoint, device=args.device)
    # init visualizer
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.draw_featmap = modify

    visualization_cfg = Config.fromfile(args.vis_config)
    recorder_cfg = visualization_cfg.recorders
    mappings = visualization_cfg.mappings
    recorder_manager = RecorderManager(recorder_cfg)
    recorder_manager.initialize(model)

    with recorder_manager:
        # test a single image
        result = inference_model(model, args.img)

    overlaid_image = mmcv.imread(
        args.img, channel_order='rgb') if args.overlaid else None

    for name, record in mappings.items():
        recorder = recorder_manager.get_recorder(record.recorder)
        record_idx = getattr(record, 'record_idx', 0)
        data_idx = getattr(record, 'data_idx')
        feats = recorder.get_record_data(record_idx, data_idx)
        if isinstance(feats, torch.Tensor):
            feats = (feats, )

        for i, feat in enumerate(feats):
            if args.use_norm:
                feat = norm(feat)
            drawn_img = visualizer.draw_featmap(
                feat[0],
                overlaid_image,
                args.channel_reduction,
                topk=args.topk,
                arrangement=tuple(args.arrangement),
                resize_shape=tuple(args.resize_shape)
                if args.resize_shape else None,
                alpha=args.alpha)
            visualizer.add_datasample(
                f'{name}_{i}',
                drawn_img,
                data_sample=result,
                draw_gt=False,
                show=args.out_file is None,
                wait_time=0.1,
                out_file=args.out_file,
                **args.cfg_options)


if __name__ == '__main__':
    args = parse_args()
    main(args)
