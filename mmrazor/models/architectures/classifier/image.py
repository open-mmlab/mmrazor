# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from mmcls.models import ImageClassifier

from mmrazor.registry import MODELS


@MODELS.register_module()
class SearchableImageClassifier(ImageClassifier):

    def __init__(self,
                 backbone: Dict,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[Dict] = None,
                 data_preprocessor: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None):
        super().__init__(backbone, neck, head, pretrained, train_cfg,
                         data_preprocessor, init_cfg)

        if self.with_head:
            self.head.connect_with_backbone(self.backbone.last_mutable)
