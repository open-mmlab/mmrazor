# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.dist import master_only
from mmengine.visualization.utils import (convert_overlay_heatmap,
                                          img_from_canvas)


@master_only
def modify(featmap: torch.Tensor,
           overlaid_image: Optional[np.ndarray] = None,
           channel_reduction: Optional[str] = 'pixel_wise_max',
           topk: int = 20,
           arrangement: Tuple[int, int] = (4, 5),
           resize_shape: Optional[tuple] = None,
           alpha: float = 0.5):
    assert isinstance(featmap,
                      torch.Tensor), (f'`featmap` should be torch.Tensor,'
                                      f' but got {type(featmap)}')
    assert featmap.ndim == 3, f'Input dimension must be 3, ' \
                              f'but got {featmap.ndim}'
    featmap = featmap.detach().cpu()

    if overlaid_image is not None:
        if overlaid_image.ndim == 2:
            overlaid_image = cv2.cvtColor(overlaid_image, cv2.COLOR_GRAY2RGB)

        if overlaid_image.shape[:2] != featmap.shape[1:]:
            warnings.warn(f'Since the spatial dimensions of '
                          f'overlaid_image: {overlaid_image.shape[:2]} and '
                          f'featmap: {featmap.shape[1:]} are not same, '
                          f'the feature map will be interpolated. '
                          f'This may cause mismatch problems ！')
            if resize_shape is None:
                overlaid_image_h, overlaid_image_w = overlaid_image.shape[:2]
                feat_h, feat_w = featmap.shape[-2:]
                if feat_h / feat_w > overlaid_image_h / overlaid_image_w:
                    feat_h = round(feat_w * overlaid_image_h /
                                   overlaid_image_w)
                else:
                    feat_w = round(feat_h * overlaid_image_w /
                                   overlaid_image_h)
                featmap = featmap[..., :feat_h, :feat_w]
                featmap = F.interpolate(
                    featmap[None], overlaid_image.shape[:2],
                    mode='bilinear')[0]

    if resize_shape is not None:
        featmap = F.interpolate(
            featmap[None], resize_shape, mode='bilinear',
            align_corners=False)[0]
        if overlaid_image is not None:
            overlaid_image = cv2.resize(overlaid_image, resize_shape[::-1])

    if channel_reduction is not None:
        assert channel_reduction in [
            'squeeze_mean', 'select_max', 'pixel_wise_max'], \
            f'Mode only support "squeeze_mean", "select_max", ' \
            f'"pixel_wise_max", but got {channel_reduction}'
        if channel_reduction == 'select_max':
            sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
            _, indices = torch.topk(sum_channel_featmap, 1)
            feat_map = featmap[indices]
        elif channel_reduction == 'squeeze_mean':
            feat_map = torch.mean(featmap, dim=0)
        else:
            feat_map = torch.max(featmap, dim=0)[0]
        return convert_overlay_heatmap(feat_map, overlaid_image, alpha)
    elif topk <= 0:
        featmap_channel = featmap.shape[0]
        assert featmap_channel in [
            1, 3
        ], ('The input tensor channel dimension must be 1 or 3 '
            'when topk is less than 1, but the channel '
            f'dimension you input is {featmap_channel}, you can use the'
            ' channel_reduction parameter or set topk greater than '
            '0 to solve the error')
        return convert_overlay_heatmap(featmap, overlaid_image, alpha)
    else:
        row, col = arrangement
        channel, height, width = featmap.shape
        assert row * col >= topk, 'The product of row and col in ' \
                                  'the `arrangement` is less than ' \
                                  'topk, please set the ' \
                                  '`arrangement` correctly'

        # Extract the feature map of topk
        topk = min(channel, topk)
        sum_channel_featmap = torch.sum(featmap, dim=(1, 2))
        _, indices = torch.topk(sum_channel_featmap, topk)
        topk_featmap = featmap[indices]

        fig = plt.figure(frameon=False)
        # Set the window layout
        fig.subplots_adjust(
            left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        dpi = fig.get_dpi()
        fig.set_size_inches((width * col + 1e-2) / dpi,
                            (height * row + 1e-2) / dpi)
        for i in range(topk):
            axes = fig.add_subplot(row, col, i + 1)
            axes.axis('off')
            axes.text(2, 15, f'channel: {indices[i]}', fontsize=10)
            axes.imshow(
                convert_overlay_heatmap(topk_featmap[i], overlaid_image,
                                        alpha))
        image = img_from_canvas(fig.canvas)
        plt.close(fig)
        return image
