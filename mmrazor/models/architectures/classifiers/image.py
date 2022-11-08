# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmrazor.registry import MODELS

try:
    from mmcls.models import ImageClassifier
except ImportError:
    from mmrazor.utils import get_placeholder
    ImageClassifier = get_placeholder('mmcls')


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
                 connect_head: Optional[dict] = None):
        super().__init__(backbone, neck, head, pretrained, train_cfg,
                         data_preprocessor, init_cfg)

        if self.with_head and connect_head is not None:
            for kh, vh in connect_head.items():
                component, attr = vh.split('.')
                value = getattr(getattr(self, component), attr)
                getattr(self.head, kh)(value)
