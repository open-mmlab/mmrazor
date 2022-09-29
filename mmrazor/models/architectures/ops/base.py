# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.model import BaseModule


class BaseOP(BaseModule):
    """Base class for searchable operations.

    Args:
        in_channels (int): The input channels of the operation.
        out_channels (int): The output channels of the operation.
        stride (int): Stride of the operation. Defaults to 1.
    """

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super(BaseOP, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
