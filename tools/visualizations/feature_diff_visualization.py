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
    parser = argparse.ArgumentParser(description='Train an algorithm')
    parser.add_argument('img', help='Image file')
    parser.add_argument(
        'tea_config', help='train config file path for the teacher model')
    parser.add_argument(
        'stu_config', help='train config file path for the student model')
    parser.add_argument('vis_config', help='visualization config file path')
    parser.add_argument(
        'tea_checkpoint', help='Checkpoint file for the teacher model')
    parser.add_argument(
        'stu_checkpoint', help='Checkpoint file for the student model')
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

    student = init_model(
        args.stu_config, args.stu_checkpoint, device=args.device)
    # init visualizer
    visualizer = VISUALIZERS.build(student.cfg.visualizer)
    visualizer.draw_featmap = modify

    teacher = init_model(
        args.tea_config, args.tea_checkpoint, device=args.device)

    visualization_cfg = Config.fromfile(args.vis_config)
    student_recorder_cfg = visualization_cfg.student_recorders
    student_mappings = visualization_cfg.student_mappings
    teacher_recorder_cfg = visualization_cfg.teacher_recorders
    teacher_mappings = visualization_cfg.teacher_mappings

    student_recorder_manager = RecorderManager(student_recorder_cfg)
    student_recorder_manager.initialize(student)

    teacher_recorder_manager = RecorderManager(teacher_recorder_cfg)
    teacher_recorder_manager.initialize(teacher)

    with student_recorder_manager:
        # test a single image
        _ = inference_model(student, args.img)

    with teacher_recorder_manager:
        # test a single image
        _ = inference_model(teacher, args.img)

    overlaid_image = mmcv.imread(
        args.img, channel_order='rgb') if args.overlaid else None

    for teacher_name, student_name in zip(teacher_mappings.keys(),
                                          student_mappings.keys()):
        teacher_record = teacher_mappings[teacher_name]
        teacher_recorder = teacher_recorder_manager.get_recorder(
            teacher_record.recorder)
        record_idx = getattr(teacher_record, 'record_idx', 0)
        data_idx = getattr(teacher_record, 'data_idx')
        teacher_feats = teacher_recorder.get_record_data(record_idx, data_idx)
        if isinstance(teacher_feats, torch.Tensor):
            teacher_feats = (teacher_feats, )

        student_record = student_mappings[student_name]
        student_recorder = student_recorder_manager.get_recorder(
            student_record.recorder)
        record_idx = getattr(student_record, 'record_idx', 0)
        data_idx = getattr(student_record, 'data_idx')
        student_feats = student_recorder.get_record_data(record_idx, data_idx)
        if isinstance(student_feats, torch.Tensor):
            student_feats = (student_feats, )

        for i, (teacher_feat,
                student_feat) in enumerate(zip(teacher_feats, student_feats)):
            diff = torch.abs(teacher_feat - student_feat)
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
                f'tea_{teacher_name}_stu_{student_name}_{i}',
                drawn_img,
                show=args.out_file is None,
                wait_time=0.1,
                out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
