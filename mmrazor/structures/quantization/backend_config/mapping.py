# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor import digit_version
from .academic import get_academic_backend_config
from .native import get_native_backend_config
from .openvino import get_openvino_backend_config
from .tensorrt import get_tensorrt_backend_config

if digit_version(
        torch.__version__) >= digit_version('1.13.0') and digit_version(
            torch.__version__) <= digit_version('1.13.1'):
    BackendConfigs = {
        'academic': get_academic_backend_config(),
        'native': get_native_backend_config(),
        'tensorrt': get_tensorrt_backend_config(),
        'openvino': get_openvino_backend_config()
    }
else:
    BackendConfigs = {
        'academic': None,
        'native': None,
        'tensorrt': None,
        'openvino': None
    }
