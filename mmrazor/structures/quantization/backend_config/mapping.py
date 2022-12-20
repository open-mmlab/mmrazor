# Copyright (c) OpenMMLab. All rights reserved.
from .academic import get_academic_backend_config
from .native import get_native_backend_config
from .openvino import get_openvino_backend_config
from .tensorrt import get_tensorrt_backend_config

BackendConfigs = {
    'academic': get_academic_backend_config(),
    'native': get_native_backend_config(),
    'tensorrt': get_tensorrt_backend_config(),
    'openvino': get_openvino_backend_config()
}
