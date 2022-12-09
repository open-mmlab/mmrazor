# Copyright (c) OpenMMLab. All rights reserved.
from .make_divisible import make_divisible
from .misc import add_prefix
from .optim_wrapper import reinitialize_optim_wrapper_count_status
# yapf:disable
from .quantization_util import (PerChannelLoadHook, _is_float_qparams,
                                _is_per_channel, _is_per_tensor,
                                _is_symmetric_quant,
                                check_is_valid_convert_custom_config_dict,
                                check_is_valid_fuse_custom_config_dict,
                                check_is_valid_prepare_custom_config_dict,
                                check_is_valid_qconfig_dict,
                                get_custom_module_class_keys, is_tracing_state,
                                pot_quantization, sync_tensor)
# yapf:enable
from .utils import get_module_device, set_requires_grad
from .quantization_util import str2class

__all__ = [
    'add_prefix', 'check_is_valid_convert_custom_config_dict',
    'check_is_valid_fuse_custom_config_dict',
    'check_is_valid_prepare_custom_config_dict', 'check_is_valid_qconfig_dict',
    'get_module_device', 'get_custom_module_class_keys', 'make_divisible',
    'pot_quantization', 'reinitialize_optim_wrapper_count_status',
    'set_requires_grad', 'sync_tensor', '_is_per_channel', '_is_per_tensor',
    '_is_symmetric_quant', '_is_float_qparams', 'is_tracing_state',
    'PerChannelLoadHook', 'str2class'
]
