# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init

from ..builder import LOSSES


@LOSSES.register_module()
class FGDLoss(nn.Module):
    """PyTorch version of 'Focal and Global Knowledge Distillation for
    Detectors'.

    <https://arxiv.org/abs/2111.11837>

    Args:
        in_channels (int): Channels of the input feature map.
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001.
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005.
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001.
        lambda_fgd (float, optional): Weight of relation_loss.
            Defaults to 0.000005.
    """

    def __init__(
        self,
        in_channels,
        temp=0.5,
        alpha_fgd=0.001,
        beta_fgd=0.0005,
        gamma_fgd=0.001,
        lambda_fgd=0.000005,
    ):
        super(FGDLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        self.conv_mask_s = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.LayerNorm([in_channels // 2, 1, 1]), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.LayerNorm([in_channels // 2, 1, 1]), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1))

        self.reset_parameters()

    def forward(self, preds_S, preds_T):
        """Forward function.

        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:]
        N, C, H, W = preds_S.shape
        # Bs*[nt*4], (tl_x, tl_y, br_x, br_y)
        gt_bboxes = self.current_data['gt_bboxes']
        # Meta information of each image, e.g., image size, scaling factor.
        metas = self.current_data['img_metas']  # list[dict]

        spatial_attention_t, channel_attention_t = self.get_attention(
            preds_T, self.temp)
        spatial_attention_s, channel_attention_s = self.get_attention(
            preds_S, self.temp)

        mask_fg = torch.zeros_like(spatial_attention_t)
        mask_bg = torch.ones_like(spatial_attention_t)
        wmin, wmax, hmin, hmax = [], [], [], []
        for i in range(N):
            new_boxx = torch.ones_like(gt_bboxes[i])
            new_boxx[:, 0] = gt_bboxes[i][:, 0] / metas[i]['img_shape'][1] * W
            new_boxx[:, 2] = gt_bboxes[i][:, 2] / metas[i]['img_shape'][1] * W
            new_boxx[:, 1] = gt_bboxes[i][:, 1] / metas[i]['img_shape'][0] * H
            new_boxx[:, 3] = gt_bboxes[i][:, 3] / metas[i]['img_shape'][0] * H

            wmin.append(torch.floor(new_boxx[:, 0]).int())
            wmax.append(torch.ceil(new_boxx[:, 2]).int())
            hmin.append(torch.floor(new_boxx[:, 1]).int())
            hmax.append(torch.ceil(new_boxx[:, 3]).int())

            height = hmax[i].view(1, -1) + 1 - hmin[i].view(1, -1)
            width = wmax[i].view(1, -1) + 1 - wmin[i].view(1, -1)
            area = 1.0 / height / width

            for j in range(len(gt_bboxes[i])):
                mask_fg[i][hmin[i][j]:hmax[i][j]+1,
                           wmin[i][j]:wmax[i][j]+1] = \
                        torch.maximum(mask_fg[i][hmin[i][j]:hmax[i][j]+1,
                                      wmin[i][j]:wmax[i][j]+1], area[0][j])

            mask_bg[i] = torch.where(mask_fg[i] > 0, 0, 1)
            if torch.sum(mask_bg[i]):
                mask_bg[i] /= torch.sum(mask_bg[i])

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, mask_fg,
                                             mask_bg, channel_attention_t,
                                             spatial_attention_t)
        mask_loss = self.get_mask_loss(channel_attention_s,
                                       channel_attention_t,
                                       spatial_attention_s,
                                       spatial_attention_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)

        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
            + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss

        return loss

    def get_attention(self, preds, temp):
        """Calculate spatial and channel attention.

        Args:
            preds (Tensor): Model prediction with shape (N, C, H, W).
            temp (float): Temperature coefficient.
        """
        N, C, H, W = preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        spatial_attention = (H * W * F.softmax(
            (fea_map / temp).view(N, -1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(
            axis=2, keepdim=False).mean(
                axis=2, keepdim=False)
        channel_attention = C * F.softmax(channel_map / temp, dim=1)

        return spatial_attention, channel_attention

    def get_fea_loss(self, preds_S, preds_T, mask_fg, mask_bg,
                     channel_attention_t, spatial_attention_t):
        loss_mse = nn.MSELoss(reduction='sum')

        mask_fg = mask_fg.unsqueeze(dim=1)
        mask_bg = mask_bg.unsqueeze(dim=1)

        channel_attention_t = channel_attention_t.unsqueeze(dim=-1).unsqueeze(
            dim=-1)
        spatial_attention_t = spatial_attention_t.unsqueeze(dim=1)

        fea_t = torch.mul(preds_T, torch.sqrt(spatial_attention_t))
        fea_t = torch.mul(fea_t, torch.sqrt(channel_attention_t))
        fea_t_fg = torch.mul(fea_t, torch.sqrt(mask_fg))
        fea_t_bg = torch.mul(fea_t, torch.sqrt(mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(spatial_attention_t))
        fea_s = torch.mul(fea_s, torch.sqrt(channel_attention_t))
        fea_s_fg = torch.mul(fea_s, torch.sqrt(mask_fg))
        fea_s_bg = torch.mul(fea_s, torch.sqrt(mask_bg))

        loss_fg = loss_mse(fea_s_fg, fea_t_fg) / len(mask_fg)
        loss_bg = loss_mse(fea_s_bg, fea_t_bg) / len(mask_bg)

        return loss_fg, loss_bg

    def get_mask_loss(self, channel_attention_s, channel_attention_t,
                      spatial_attention_s, spatial_attention_t):

        mask_loss = torch.sum(
            torch.abs(
                (channel_attention_s -
                 channel_attention_t))) / len(channel_attention_s) + torch.sum(
                     torch.abs(
                         (spatial_attention_s -
                          spatial_attention_t))) / len(spatial_attention_s)

        return mask_loss

    def spatial_pool(self, x, is_student_input):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if is_student_input:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, is_student_input=True)
        context_t = self.spatial_pool(preds_T, is_student_input=False)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t) / len(out_s)

        return rela_loss

    def last_zero_init(self, m):
        if isinstance(m, nn.Sequential):
            constant_init(m[-1], val=0)
        else:
            constant_init(m, val=0)

    def reset_parameters(self):
        kaiming_init(self.conv_mask_s, mode='fan_in')
        kaiming_init(self.conv_mask_t, mode='fan_in')
        self.conv_mask_s.inited = True
        self.conv_mask_t.inited = True

        self.last_zero_init(self.channel_add_conv_s)
        self.last_zero_init(self.channel_add_conv_t)
