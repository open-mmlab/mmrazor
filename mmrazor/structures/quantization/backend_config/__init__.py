# Copyright (c) OpenMMLab. All rights reserved.
from .academic import (get_academic_backend_config,
                       get_academic_backend_config_dict)
from .mapping import BackendConfigs
from .native import get_native_backend_config, get_native_backend_config_dict
from .openvino import (get_openvino_backend_config,
                       get_openvino_backend_config_dict)
from .tensorrt import (get_tensorrt_backend_config,
                       get_tensorrt_backend_config_dict)

__all__ = [
    'BackendConfigs',
    'get_native_backend_config',
    'get_native_backend_config_dict',
    'get_academic_backend_config',
    'get_academic_backend_config_dict',
    'get_openvino_backend_config',
    'get_openvino_backend_config_dict',
    'get_tensorrt_backend_config',
    'get_tensorrt_backend_config_dict',
]
