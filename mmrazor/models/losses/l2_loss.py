# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrazor.registry import MODELS


@MODELS.register_module()
class L2Loss(nn.Module):
    """Calculate the two-norm loss between the two features.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        normalize (bool): Whether to normalize the feature. Defaults to True.
        mult (float): Multiplier for feature normalization. Defaults to 1.0.
        div_element (bool): Whether to divide the loss by element-wise.
            Defaults to False.
        dist (bool): Whether to conduct two-norm dist as torch.dist(p=2).
            Defaults to False.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        normalize: bool = True,
        mult: float = 1.0,
        div_element: bool = False,
        dist: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.normalize = normalize
        self.mult = mult
        self.div_element = div_element
        self.dist = dist

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation.

        Args:
            s_feature (torch.Tensor): The student model feature with
                shape (N, C, H, W) or shape (N, C).
            t_feature (torch.Tensor): The teacher model feature with
                shape (N, C, H, W) or shape (N, C).
        """
        if self.normalize:
            s_feature = self.normalize_feature(s_feature)
            t_feature = self.normalize_feature(t_feature)

        loss = torch.sum(torch.pow(torch.sub(s_feature, t_feature), 2))

        # Calculate l2_loss as dist.
        if self.dist:
            loss = torch.sqrt(loss)
        else:
            if self.div_element:
                loss = loss / s_feature.numel()
            else:
                loss = loss / s_feature.size(0)

        return self.loss_weight * loss

    def normalize_feature(self, feature: torch.Tensor) -> torch.Tensor:
        """Normalize the input feature.

        Args:
            feature (torch.Tensor): The student model feature with
                shape (N, C, H, W) or shape (N, C).
        """
        feature = feature.view(feature.size(0), -1)
        return feature / feature.norm(2, dim=1, keepdim=True) * self.mult
