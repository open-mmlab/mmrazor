# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union
import warnings

import cv2
import mmcv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData, BaseDataElement
from mmengine.visualization import Visualizer
from mmengine.visualization.utils import (check_type, check_type_and_length,
                                          color_str2rgb, color_val_matplotlib, convert_overlay_heatmap,
                                          img_from_canvas, tensor2ndarray,
                                          value2list, wait_continue)

from ..registry import VISUALIZERS
from types import MethodType

def modify_draw_featmap(visualizer):
    @master_only
    def modify(self, featmap: torch.Tensor,
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
                overlaid_image = cv2.cvtColor(overlaid_image,
                                              cv2.COLOR_GRAY2RGB)

            if overlaid_image.shape[:2] != featmap.shape[1:]:
                warnings.warn(
                    f'Since the spatial dimensions of '
                    f'overlaid_image: {overlaid_image.shape[:2]} and '
                    f'featmap: {featmap.shape[1:]} are not same, '
                    f'the feature map will be interpolated. '
                    f'This may cause mismatch problems ！')
                if resize_shape is None:
                    overlaid_image_h, overlaid_image_w = overlaid_image.shape[
                                                         :2]
                    feat_h, feat_w = featmap.shape[-2:]
                    if feat_h / feat_w > overlaid_image_h / overlaid_image_w:
                        feat_h = round(
                            feat_w * overlaid_image_h / overlaid_image_w)
                    else:
                        feat_w = round(
                            feat_h * overlaid_image_w / overlaid_image_h)
                    featmap = featmap[..., :feat_h, :feat_w]
                    featmap = \
                    F.interpolate(featmap[None], overlaid_image.shape[:2],
                                  mode='bilinear')[0]

        if resize_shape is not None:
            featmap = F.interpolate(
                featmap[None],
                resize_shape,
                mode='bilinear',
                align_corners=False)[0]
            if overlaid_image is not None:
                overlaid_image = cv2.resize(overlaid_image, resize_shape[::-1])

        if channel_reduction is not None:
            assert channel_reduction in [
                'squeeze_mean', 'select_max', 'pixel_wise_max'], \
                f'Mode only support "squeeze_mean", "select_max", "pixel_wise_max", ' \
                f'but got {channel_reduction}'
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
    return MethodType(modify, visualizer)


@VISUALIZERS.register_module()
class RazorLocalVisualizer(Visualizer):

    @staticmethod
    @master_only
    def draw_featmap(featmap: torch.Tensor,
                     overlaid_image: Optional[np.ndarray] = None,
                     channel_reduction: Optional[str] = 'pixel_wise_max',
                     topk: int = 20,
                     arrangement: Tuple[int, int] = (4, 5),
                     resize_shape: Optional[tuple] = None,
                     alpha: float = 0.5) -> np.ndarray:
        """Draw featmap.

        - If `overlaid_image` is not None, the final output image will be the
        weighted sum of img and featmap.

        - If `resize_shape` is specified, `featmap` and `overlaid_image`
        are interpolated.

        - If `resize_shape` is None and `overlaid_image` is not None,
        the feature map will be interpolated to the spatial size of the image
        in the case where the spatial dimensions of `overlaid_image` and
        `featmap` are different.

        - If `channel_reduction` is "squeeze_mean" and "select_max",
        it will compress featmap to single channel image and weighted
        sum to `overlaid_image`.

        -  if `channel_reduction` is None

          - If topk <= 0, featmap is assert to be one or three
          channel and treated as image and will be weighted sum
          to ``overlaid_image``.
          - If topk > 0, it will select topk channel to show by the sum of
          each channel. At the same time, you can specify the `arrangement`
          to set the window layout.

        Args:
            featmap (torch.Tensor): The featmap to draw which format is
                (C, H, W).
            overlaid_image (np.ndarray, optional): The overlaid image.
                Default to None.
            channel_reduction (str, optional): Reduce multiple channels to a
                single channel. The optional value is 'squeeze_mean'
                or 'select_max'. Defaults to 'squeeze_mean'.
            topk (int): If channel_reduction is not None and topk > 0,
                it will select topk channel to show by the sum of each channel.
                if topk <= 0, tensor_chw is assert to be one or three.
                Defaults to 20.
            arrangement (Tuple[int, int]): The arrangement of featmap when
                channel_reduction is not None and topk > 0. Defaults to (4, 5).
            resize_shape (tuple, optional): The shape to scale the feature map.
                Default to None.
            alpha (Union[int, List[int]]): The transparency of featmap.
                Defaults to 0.5.

        Returns:
            np.ndarray: RGB image.
        """
        assert isinstance(featmap,
                          torch.Tensor), (f'`featmap` should be torch.Tensor,'
                                          f' but got {type(featmap)}')
        assert featmap.ndim == 3, f'Input dimension must be 3, ' \
                                  f'but got {featmap.ndim}'
        featmap = featmap.detach().cpu()

        if overlaid_image is not None:
            if overlaid_image.ndim == 2:
                overlaid_image = cv2.cvtColor(overlaid_image,
                                              cv2.COLOR_GRAY2RGB)

            if overlaid_image.shape[:2] != featmap.shape[1:]:
                warnings.warn(
                    f'Since the spatial dimensions of '
                    f'overlaid_image: {overlaid_image.shape[:2]} and '
                    f'featmap: {featmap.shape[1:]} are not same, '
                    f'the feature map will be interpolated. '
                    f'This may cause mismatch problems ！')
                if resize_shape is None:
                    overlaid_image_h, overlaid_image_w = overlaid_image.shape[:2]
                    feat_h, feat_w = featmap.shape[-2:]
                    if feat_h / feat_w > overlaid_image_h / overlaid_image_w:
                        feat_h = round(feat_w * overlaid_image_h / overlaid_image_w)
                    else:
                        feat_w = round(feat_h * overlaid_image_w / overlaid_image_h)
                    featmap = featmap[..., :feat_h, :feat_w]
                    featmap = F.interpolate(featmap[None], overlaid_image.shape[:2], mode='bilinear')[0]

        if resize_shape is not None:
            featmap = F.interpolate(
                featmap[None],
                resize_shape,
                mode='bilinear',
                align_corners=False)[0]
            if overlaid_image is not None:
                overlaid_image = cv2.resize(overlaid_image, resize_shape[::-1])

        if channel_reduction is not None:
            assert channel_reduction in [
                'squeeze_mean', 'select_max', 'pixel_wise_max'], \
                f'Mode only support "squeeze_mean", "select_max", "pixel_wise_max", ' \
                f'but got {channel_reduction}'
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

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['BaseDataElement'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)
