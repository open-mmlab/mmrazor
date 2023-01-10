# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor import digit_version

if digit_version(torch.__version__) >= digit_version('1.13.0'):
    from .academic_quantizer import AcademicQuantizer
    from .base import BaseQuantizer
    from .native_quantizer import NativeQuantizer
    from .openvino_quantizer import OpenVINOQuantizer
    from .tensorrt_quantizer import TensorRTQuantizer

    __all__ = [
        'BaseQuantizer', 'AcademicQuantizer', 'NativeQuantizer',
        'TensorRTQuantizer', 'OpenVINOQuantizer'
    ]
