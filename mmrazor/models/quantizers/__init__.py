# Copyright (c) OpenMMLab. All rights reserved.
from .base import CustomQuantizer
from .openvino_quantizer import OpenvinoQuantizer
from .trt_quantizer import TensorRTQuantizer

__all__ = ['CustomQuantizer', 'TensorRTQuantizer', 'OpenvinoQuantizer']
