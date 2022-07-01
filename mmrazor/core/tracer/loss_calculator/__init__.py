# Copyright (c) OpenMMLab. All rights reserved.
from .image_classifier_loss_calculator import ImageClassifierPseudoLoss
from .single_stage_detector_loss_calculator import \
    SingleStageDetectorPseudoLoss

__all__ = ['ImageClassifierPseudoLoss', 'SingleStageDetectorPseudoLoss']
