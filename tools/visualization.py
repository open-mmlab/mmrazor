import argparse
import os
import torch
import torch.nn.functional as F
import mmcv
import numpy as np
import cv2

from mmengine.config import Config, DictAction, ConfigDict
from mmengine.utils import import_modules_from_strings

from mmrazor.models.task_modules import RecorderManager
from mmrazor.utils import register_all_modules
from mmrazor.visualization import RazorLocalVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train an algorithm')
    # parser.add_argument('img', help='Image file')
    # parser.add_argument('config', help='train config file path')
    # parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument('--repo', help='the corresponding repo name')
    parser.add_argument('--use-norm', action='store_true', help='normalize the featmap before visualization')
    parser.add_argument('--overlaid', action='store_true', help='overlaid image')
    parser.add_argument(
        '--channel-reduction',
        help='Reduce multiple channels to a single channel. The optional value'
             ' is \'squeeze_mean\' or \'select_max\'.', default='squeeze_mean')
    parser.add_argument(
        '--topk',
        help='If channel_reduction is not None and topk > 0, it will select '
             'topk channel to show by the sum of each channel. If topk <= 0, '
             'tensor_chw is assert to be one or three.', default=20)
    parser.add_argument(
        '--arrangement',
        help='the arrangement of featmap when channel_reduction is not None '
             'and topk > 0.', default=(4, 5))
    parser.add_argument(
        '--resize-shape', help='the shape to scale the feature map')
    parser.add_argument('--alpha', help='the transparency of featmap', default=0.5)

    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def resize(ori_shape, feat):
    ori_h, ori_w = ori_shape
    # img_h, img_w = img_shape
    feat_h, feat_w = feat.shape[-2:]
    if feat_h / feat_w > ori_h / ori_w:
        feat_h = round(feat_w * ori_h / ori_w)
    else:
        feat_w = round(feat_h * ori_w / ori_h)
    # h, w = round(ori_h * feat_h / img_h), round(ori_w * feat_w / img_w)
    feat = feat[..., :feat_h, :feat_w]
    return F.interpolate(feat, ori_shape, mode='bilinear')

def norm(feat):
    N, C, H, W = feat.shape
    feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
    mean = feat.mean(dim=-1, keepdim=True)
    std = feat.std(dim=-1, keepdim=True)
    centered = (feat - mean) / std
    centered = centered.reshape(C, N, H, W).permute(1, 0, 2, 3)
    return centered

def to255(feat, mmin=None, mmax=None):
    if mmin is None:
        mmax = np.max(feat)
        mmin = np.min(feat)
    # mmax, mmin = 10, -10
    k = (255 - 0) / (mmax - mmin)
    normed = 0 + k * (feat - mmin)
    return np.clip(normed.astype(int), 0, 255)
    # return torch.clamp(normed.int(), 0, 255).cpu().numpy()


def convert_overlay_heatmap(feat_map, img, alpha = 0.5, mmin=None, mmax=None):
    """Convert feat_map to heatmap and overlay on image, if image is not None.

    Args:
        feat_map (np.ndarray, torch.Tensor): The feat_map to convert
            with of shape (H, W), where H is the image height and W is
            the image width.
        img (np.ndarray, optional): The origin image. The format
            should be RGB. Defaults to None.
        alpha (float): The transparency of featmap. Defaults to 0.5.

    Returns:
        np.ndarray: heatmap
    """
    assert feat_map.ndim == 2 or (feat_map.ndim == 3
                                  and feat_map.shape[0] in [1, 3])
    if isinstance(feat_map, torch.Tensor):
        feat_map = feat_map.detach().cpu().numpy()

    if feat_map.ndim == 3:
        feat_map = feat_map.transpose(1, 2, 0)

    if mmax is None:
        norm_img = np.zeros(feat_map.shape)
        norm_img = cv2.normalize(feat_map, norm_img, 0, 255, cv2.NORM_MINMAX)
        # print(norm_img)
        # print(feat_map.min(), feat_map.max())
    else:
        norm_img = to255(feat_map, mmin, mmax)
        # print(norm_img)
    print(norm_img.max())
    norm_img = np.asarray(norm_img, dtype=np.uint8)
    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
    if img is not None:
        heat_img = cv2.addWeighted(img, 1 - alpha, heat_img, alpha, 0)
    return heat_img


# def resize(feat, ori_size, img_size, pad_size):
#     ph, pw = pad_size[:2]
#     ih, iw = img_size[:2]
#     fh, fw = feat.shape[-2:]
#     rh, rw = ih / ph, iw / pw
#     h, w = round(fh * rh), round(fw * rw)
#     feat = feat[..., :h, :w]
#     return F.interpolate(feat, ori_size[:2], mode='bilinear')
gfl_cfg = r'D:\projects\openmmlab\mmdetection\configs\gfl\gfl_r101_fpn_ms-2x_coco.py'
gfl_ckpt = r'D:\projects\checkpoints\gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'
retina_cfg = r'G:\projects\openmmlab\mmdetection\configs\retinanet\retinanet_r101_fpn_2x_coco.py'
retina_ckpt = r'G:\projects\research\checkpoint\retinanet_r101_fpn_2x_coco.pth'

def main(args):
    args.config = gfl_cfg
    args.repo = 'mmdet'
    args.checkpoint = gfl_ckpt
    args.img = r'D:\projects\openmmlab\mmdetection\demo\demo.jpg'
    args.overlaid = True
    args.channel_reduction = 'squeeze_mean'
    use_norm = False
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
    visualizer = RazorLocalVisualizer()

    recorders = ConfigDict(neck=dict(_scope_='mmrazor', type='ModuleOutputs', source='neck'))
    mapping = ConfigDict(p3=dict(recorder='neck', data_idx=0),
                         p4=dict(recorder='neck', data_idx=1),
                         p5=dict(recorder='neck', data_idx=2),
                         p6=dict(recorder='neck', data_idx=3))
    recorder_manager = RecorderManager(recorders)
    recorder_manager.initialize(model)

    with recorder_manager:
        # test a single image
        _ = inference_model(model, args.img)

    overlaid_image = mmcv.imread(args.img, channel_order='rgb') if args.overlaid else None

    for name, record in mapping.items():
        recorder = recorder_manager.get_recorder(record.recorder)
        record_idx = getattr(record, 'record_idx', 0)
        data_idx = getattr(record, 'data_idx')
        feats = recorder.get_record_data(record_idx, data_idx)
        if isinstance(feats, torch.Tensor):
            feats = (feats,)

        for i, feat in enumerate(feats):
            if use_norm:
                feat = norm(feat)
            drawn_img = visualizer.draw_featmap(feat[0], overlaid_image, 'pixel_wise_max')
            visualizer.add_datasample(
                f'{name}_{i}',
                drawn_img,
                show=args.out_file is None,
                wait_time=0.1,
                out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    main(args)
