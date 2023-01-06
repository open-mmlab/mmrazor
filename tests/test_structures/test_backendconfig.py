# Copyright (c) OpenMMLab. All rights reserved.
from torch.ao.quantization.backend_config import BackendConfig

from mmrazor.structures.quantization.backend_config import (
    BackendConfigs, get_academic_backend_config,
    get_academic_backend_config_dict, get_native_backend_config,
    get_native_backend_config_dict, get_openvino_backend_config,
    get_openvino_backend_config_dict, get_tensorrt_backend_config,
    get_tensorrt_backend_config_dict)


def test_get_backend_config():
    # test get_native_backend_config
    native_backend_config = get_native_backend_config()
    assert isinstance(native_backend_config, BackendConfig)
    assert native_backend_config.name == 'native'
    native_backend_config_dict = get_native_backend_config_dict()
    assert isinstance(native_backend_config_dict, dict)

    # test get_academic_backend_config
    academic_backend_config = get_academic_backend_config()
    assert isinstance(academic_backend_config, BackendConfig)
    assert academic_backend_config.name == 'academic'
    academic_backend_config_dict = get_academic_backend_config_dict()
    assert isinstance(academic_backend_config_dict, dict)

    # test get_openvino_backend_config
    openvino_backend_config = get_openvino_backend_config()
    assert isinstance(openvino_backend_config, BackendConfig)
    assert openvino_backend_config.name == 'openvino'
    openvino_backend_config_dict = get_openvino_backend_config_dict()
    assert isinstance(openvino_backend_config_dict, dict)

    # test get_tensorrt_backend_config
    tensorrt_backend_config = get_tensorrt_backend_config()
    assert isinstance(tensorrt_backend_config, BackendConfig)
    assert tensorrt_backend_config.name == 'tensorrt'
    tensorrt_backend_config_dict = get_tensorrt_backend_config_dict()
    assert isinstance(tensorrt_backend_config_dict, dict)


def test_backendconfigs_mapping():
    mapping = BackendConfigs
    assert isinstance(mapping, dict)
    assert 'academic' in mapping.keys()
    assert isinstance(mapping['academic'], BackendConfig)
