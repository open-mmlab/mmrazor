from enum import Enum
from .native import get_native_backend_config
from .tensorrt import get_tensorrt_backend_config
from .openvino import get_openvino_backend_config

BackendConfigs = {
    'native': get_native_backend_config(),
    'tensorrt': get_tensorrt_backend_config(),
    'openvino': get_openvino_backend_config()
}