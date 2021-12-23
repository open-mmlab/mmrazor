# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import ARCHITECTURES
from .base import BaseArchitecture


@ARCHITECTURES.register_module()
class MMClsArchitecture(BaseArchitecture):
    """Architecture based on MMCls."""

    def __init__(self, **kwargs):
        super(MMClsArchitecture, self).__init__(**kwargs)

    def forward_dummy(self, img):
        """Used for calculating network flops."""
        output = img
        for name, child in self.model.named_children():
            if name == 'head':
                output = child.fc(output[0])
            else:
                output = child(output)
        return output

    def cal_pseudo_loss(self, pseudo_img):
        """Used for executing ``forward`` with pseudo_img."""
        return sum(pseudo_img)
