# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import multi_apply, unmap

from ..builder import LOSSES


def levels_to_images(mlvl_tensor, num_classes=80):
    """Concat multi-level feature maps by image.

    [feature_level0, feature_level1...] -> [feature_image0, feature_image1...]
    Convert the shape of each element in mlvl_tensor from (N, C, H, W) to
    (N, H*W , C), then split the element to N elements with shape (H*W, C), and
    concat elements in same image of all level along first dimension.
    Args:
        mlvl_tensor (list[torch.Tensor]): list of Tensor which collect from
            corresponding level. Each element is of shape (N, C, H, W)
    Returns:
        list[torch.Tensor]: A list that contains N tensors and each tensor is
            of shape (num_elements, C)
    """
    batch_size = mlvl_tensor[0].size(0)
    batch_list = [[] for _ in range(batch_size)]
    for t in mlvl_tensor:
        t = t.permute(0, 2, 3, 1)
        t = t.reshape(batch_size, -1, num_classes)
        for img in range(batch_size):
            batch_list[img].append(t[img])
    return [torch.cat(item, 0) for item in batch_list]


@LOSSES.register_module()
class RankMimicLoss(nn.Module):

    def __init__(self, cls_out_channels=80, loss_weight=1.0):
        super(RankMimicLoss, self).__init__()
        self.cls_out_channels = cls_out_channels
        self.loss_weight = loss_weight

    def forward(self, student_assgin_results, student_inside_flags,
                student_cls_scores, teacher_assgin_results,
                teacher_inside_flags, teacher_cls_scores):

        student_cls_scores = levels_to_images(student_cls_scores)
        teacher_cls_scores = levels_to_images(teacher_cls_scores)

        loss_list = multi_apply(self.loss_single, student_assgin_results,
                                student_inside_flags, student_cls_scores,
                                teacher_assgin_results, teacher_inside_flags,
                                teacher_cls_scores)[0]

        return sum(loss_list) / len(loss_list) * self.loss_weight

    def loss_single(self, student_assign_result, student_inside_flag,
                    student_cls_score, teacher_assign_result,
                    teacher_inside_flag, teacher_cls_score):

        student_gt_inds = student_assign_result.gt_inds
        teacher_gt_inds = teacher_assign_result.gt_inds

        # assert student_gt_inds == teacher_gt_inds

        student_num_gts = student_assign_result.num_gts
        teacher_num_gts = teacher_assign_result.num_gts

        assert student_num_gts == teacher_num_gts

        student_labels = student_assign_result.labels
        teacher_labels = teacher_assign_result.labels

        student_num_anchors = student_inside_flag.size(0)
        teacher_num_anchors = teacher_inside_flag.size(0)

        student_unmap_gt_inds = unmap(
            student_gt_inds, student_num_anchors, student_inside_flag, fill=0)
        teacher_unmap_gt_inds = unmap(
            teacher_gt_inds, teacher_num_anchors, teacher_inside_flag, fill=0)

        student_unmap_labels = unmap(
            student_labels, student_num_anchors, student_inside_flag, fill=-1)
        teacher_unmap_labels = unmap(
            teacher_labels, teacher_num_anchors, teacher_inside_flag, fill=-1)

        loss = 0
        for i in range(1, student_num_gts + 1):
            student_anchor_inds = student_unmap_gt_inds == i
            teacher_anchor_inds = teacher_unmap_gt_inds == i

            student_label_inds = student_unmap_labels[student_anchor_inds]
            teacher_label_inds = teacher_unmap_labels[teacher_anchor_inds]

            student_anchor_scores = student_cls_score[
                teacher_anchor_inds, student_label_inds].sigmoid()
            teacher_anchor_scores = teacher_cls_score[
                teacher_anchor_inds, teacher_label_inds].sigmoid()

            student_rank_scores, _ = torch.sort(student_anchor_scores)
            teacher_rank_scores, _ = torch.sort(teacher_anchor_scores)

            q = F.softmax(student_rank_scores)
            p = F.softmax(teacher_rank_scores)

            kl_loss = (p * p.log() - p * q.log()).sum()
            loss += kl_loss

        return loss / student_num_gts,
