# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from mmcls.models import ImageClassifier
from torch import Tensor

# bignas PR

try:
    from mmrazor.models.architectures.dynamic_ops import DynamicInputResizer
except ImportError:
    from mmrazor.utils import get_placeholder
    DynamicInputResizer = get_placeholder('DynamicInputResizer')

from mmrazor.models.mutables import OneShotMutableValue
from mmrazor.registry import MODELS


@MODELS.register_module()
class SearchableImageClassifier(ImageClassifier):
    """SearchableImageClassifier for sliceable networks.

    Args:
        backbone (dict): The same as ImageClassifier.
        neck (dict, optional): The same as ImageClassifier. Defaults to None.
        head (dict, optional): The same as ImageClassifier. Defaults to None.
        pretrained (dict, optional): The same as ImageClassifier. Defaults to
            None.
        train_cfg (dict, optional): The same as ImageClassifier. Defaults to
            None.
        data_preprocessor (dict, optional): The same as ImageClassifier.
            Defaults to None.
        init_cfg (dict, optional): The same as ImageClassifier. Defaults to
            None.
        input_resizer_cfg (dict, optional): Resize Process for the input data.
            Defaults to None.
    """

    def __init__(self,
                 backbone: Dict,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[Dict] = None,
                 data_preprocessor: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None,
                 input_resizer_cfg: Optional[Dict] = None):
        super().__init__(backbone, neck, head, pretrained, train_cfg,
                         data_preprocessor, init_cfg)

        if self.with_head:
            self.head.connect_with_backbone(self.backbone.last_mutable)

        if input_resizer_cfg is not None:
            input_resizer: Optional[DynamicInputResizer] = \
                self._build_input_resizer(input_resizer_cfg)
        else:
            input_resizer = None
        self.input_resizer = input_resizer

    def extract_feat(self, batch_inputs: Tensor, stage='neck') -> Tensor:
        if self.input_resizer is not None:
            batch_inputs = self.input_resizer(batch_inputs)

        return super().extract_feat(batch_inputs, stage)

    def _build_input_resizer(self,
                             input_resizer_cfg: Dict) -> DynamicInputResizer:
        input_resizer_cfg_ = input_resizer_cfg['input_resizer']
        input_resizer = MODELS.build(input_resizer_cfg_)
        if not isinstance(input_resizer, DynamicInputResizer):
            raise TypeError('input_resizer should be a `dict` or '
                            '`DynamicInputResizer` instance, but got '
                            f'{type(input_resizer)}')

        mutable_shape_cfg = input_resizer_cfg['mutable_shape']
        mutable_shape = MODELS.build(mutable_shape_cfg)
        if not isinstance(mutable_shape, OneShotMutableValue):
            raise ValueError('`mutable_shape` should be instance of '
                             'OneShotMutableValue')
        input_resizer.mutate_shape(mutable_shape)

        return input_resizer
