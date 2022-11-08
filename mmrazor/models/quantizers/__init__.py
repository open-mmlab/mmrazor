# Copyright (c) OpenMMLab. All rights reserved.
from .base import CustomQuantizer
from .qat_quantizer import QATQuantizer

__all__ = ['CustomQuantizer', 'QATQuantizer']
