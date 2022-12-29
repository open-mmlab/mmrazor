# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.ao.quantization.backend_config import BackendConfig, DTypeConfig

from .common_operator_config_utils import (_get_conv_configs,
                                           _get_linear_configs)

# ===================
# |  DTYPE CONFIGS  |
# ===================

# weighted op int8 dtype config
# this is config for ops that has quantized weights, like linear, conv
weighted_op_int8_dtype_config = DTypeConfig(
    input_dtype=torch.quint8,
    output_dtype=torch.quint8,
    weight_dtype=torch.qint8,
    bias_dtype=torch.float,
)

# =====================
# |  BACKEND CONFIGS  |
# =====================


def get_academic_backend_config() -> BackendConfig:
    """Return the `BackendConfig` for academic reseaching."""
    conv_dtype_configs = [weighted_op_int8_dtype_config]
    linear_dtype_configs = [weighted_op_int8_dtype_config]

    return BackendConfig('native') \
        .set_backend_pattern_configs(_get_conv_configs(conv_dtype_configs)) \
        .set_backend_pattern_configs(_get_linear_configs(linear_dtype_configs))


def get_academic_backend_config_dict():
    """Return the `BackendConfig` for academic reseaching in dictionary
    form."""
    return get_academic_backend_config().to_dict()


__all__ = [
    'get_academic_backend_config',
    'get_academic_backend_config_dict',
]
