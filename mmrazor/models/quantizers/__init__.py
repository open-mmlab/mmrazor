# Copyright (c) OpenMMLab. All rights reserved.
from .base import WithoutDeployQuantizer, WithDeployQuantizer
from .openvino_quantizer import OpenVINOQuantizer
from .tensorrt_quantizer import TensorRTQuantizer

__all__ = [
    'WithoutDeployQuantizer',
    'WithDeployQuantizer',
    'TensorRTQuantizer', 
    'OpenVINOQuantizer']
