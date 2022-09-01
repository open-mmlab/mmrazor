# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.dist import get_dist_info

from mmrazor.registry import MODELS
from ..architectures.ops import GatherTensors


class DAFLLoss(nn.Module):
    """Base class for DAFL losses.

    paper link: https://arxiv.org/pdf/1904.01186.pdf

    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, preds_T: torch.Tensor) -> torch.Tensor:
        """Forward function for the DAFLLoss.

        Args:
            preds_T (torch.Tensor): The predictions of teacher.
        """
        return self.loss_weight * self.forward_train(preds_T)

    def forward_train(self, preds_T: torch.Tensor) -> torch.Tensor:
        """Forward function during training.

        Args:
            preds_T (torch.Tensor): The predictions of teacher.
        """
        raise NotImplementedError


@MODELS.register_module()
class OnehotLikeLoss(DAFLLoss):
    """The loss function for measuring the one-hot-likeness of the target
    logits."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward_train(self, preds_T: torch.Tensor) -> torch.Tensor:
        """Forward function in training for the OnehotLikeLoss.

        Args:
            preds_T (torch.Tensor): The predictions of teacher.
        """
        fake_label = preds_T.data.max(1)[1]
        return F.cross_entropy(preds_T, fake_label)


@MODELS.register_module()
class InformationEntropyLoss(DAFLLoss):
    """The loss function for measuring the class balance of the target logits.

    Args:
        gather (bool, optional): The switch controlling whether
            collecting tensors from multiple gpus. Defaults to True.
    """

    def __init__(self, gather=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gather = gather
        _, self.world_size = get_dist_info()

    def forward_train(self, preds_T: torch.Tensor) -> torch.Tensor:
        """Forward function in training for the InformationEntropyLoss.

        Args:
            preds_T (torch.Tensor): The predictions of teacher.
        """
        # Gather predictions from all GPUS to calibrate the loss function.
        if self.gather and self.world_size > 1:
            preds_T = torch.cat(GatherTensors.apply(preds_T), dim=0)
        class_prob = F.softmax(preds_T, dim=1).mean(dim=0)
        info_entropy = class_prob * torch.log10(class_prob)
        return info_entropy.sum()


@MODELS.register_module()
class ActivationLoss(nn.Module):
    """The loss function for measuring the activation of the target featuremap.
    It is negative of the norm of the target featuremap.

    Args:
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        norm_type (str, optional):The type of the norm. Defaults to 'abs'.
    """

    def __init__(self, loss_weight=1.0, norm_type='abs') -> None:
        super().__init__()
        self.loss_weight = loss_weight
        assert norm_type in ['norm', 'abs'], \
            '"norm_type" must be "norm" or "abs"'
        self.norm_type = norm_type

        if self.norm_type == 'norm':
            self.norm_fn = lambda x: -x.norm()
        elif self.norm_type == 'abs':
            self.norm_fn = lambda x: -x.abs().mean()

    def forward(self, feat_T: torch.Tensor) -> torch.Tensor:
        """Forward function for the ActivationLoss.

        Args:
            feat_T (torch.Tensor): The featuremap of teacher.
        """
        return self.loss_weight * self.forward_train(feat_T)

    def forward_train(self, feat_T: torch.Tensor) -> torch.Tensor:
        """Forward function in training for the ActivationLoss.

        Args:
            feat_T (torch.Tensor): The featuremap of teacher.
        """
        feat_T = feat_T.view(feat_T.size(0), -1)
        return self.norm_fn(feat_T)
