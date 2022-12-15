# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

try:
    from mmcls.models import ImageClassifier
except ImportError:
    from mmrazor.utils import get_placeholder
    ImageClassifier = get_placeholder('mmcls')
from torch import Tensor

from mmrazor.models.architectures.dynamic_ops import DynamicInputResizer
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
        input_resizer_cfg (dict, optional): Configs for a input resizer, which
            is designed for dynamically changing the input size, making the
            input size as a searchable part. Defaults to None.
        connect_head (dict, optional): Dimensions are aligned in head will be
            substitute to it's `str type` value, so that search_space of the
            first components can be connets to the next. e.g:
            {'connect_with_backbone': 'backbone.last_mutable'} means that
            func:`connect_with_backbone` will be substitute to backbones
            last_mutable. Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 input_resizer_cfg: Optional[dict] = None,
                 connect_head: Optional[dict] = None):
        super().__init__(backbone, neck, head, pretrained, train_cfg,
                         data_preprocessor, init_cfg)

        if self.with_head and connect_head is not None:
            for kh, vh in connect_head.items():
                component, attr = vh.split('.')
                value = getattr(getattr(self, component), attr)
                getattr(self.head, kh)(value)

        if input_resizer_cfg is not None:
            input_resizer: Optional[DynamicInputResizer] = \
                self._build_input_resizer(input_resizer_cfg)
        else:
            input_resizer = None
        self.input_resizer = input_resizer

    def extract_feat(self,
                     batch_inputs: Tensor,
                     stage: str = 'neck',
                     input_resizer: bool = True) -> Tensor:
        """Extract features with resizing inputs first."""
        if self.input_resizer is not None and input_resizer:
            batch_inputs = self.input_resizer(batch_inputs)

        return super().extract_feat(batch_inputs, stage)

    def _build_input_resizer(self,
                             input_resizer_cfg: Dict) -> DynamicInputResizer:
        """Build a input resizer."""
        mutable_shape_cfg = dict(type='OneShotMutableValue')

        mutable_shape_cfg['alias'] = \
            input_resizer_cfg.get('alias', 'input_shape')

        assert 'input_sizes' in input_resizer_cfg and \
            isinstance(input_resizer_cfg['input_sizes'][0], list), (
                'input_resizer_cfg[`input_sizes`] should be List[list].')
        mutable_shape_cfg['value_list'] = \
            input_resizer_cfg.get('input_sizes')  # type: ignore

        mutable_shape = MODELS.build(mutable_shape_cfg)

        input_resizer = MODELS.build(dict(type='DynamicInputResizer'))
        input_resizer.register_mutable_attr('shape', mutable_shape)

        return input_resizer

    def simple_test(self, img, img_metas=None, **kwargs):
        """Test without augmentation."""
        x = self.extract_feat(img, input_resizer=False)
        res = self.head.simple_test(x, **kwargs)

        return res
