# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Set

import torch

USE_LINK = False
USE_DDP = False

try:
    import spring.linklink as link
    assert link.is_initialized()
    USE_LINK = True
except (ModuleNotFoundError, AssertionError):
    import torch.distributed as dist
    if torch.distributed.is_initialized():
        USE_DDP = True


def sync_tensor(tensor):
    if USE_LINK:
        if tensor.is_cuda is True:
            tensor.data = tensor.data / link.get_world_size()
            link.allreduce(tensor.data)
    elif USE_DDP:
        tensor.data = tensor.data / dist.get_world_size()
        dist.all_reduce(tensor.data)
    return tensor


def pot_quantization(tensor: torch.Tensor, mode='round'):
    log2t = torch.log2(tensor)
    if mode == 'round':
        log2t = (torch.round(log2t) - log2t).detach() + log2t
    else:
        assert mode == 'floor'
        log2t = (torch.floor(log2t) - log2t).detach() + log2t
    return 2**log2t


def _is_per_channel(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [
        torch.per_channel_symmetric, torch.per_channel_affine,
        torch.per_channel_affine_float_qparams
    ]


def _is_per_tensor(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]


def _is_symmetric_quant(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [torch.per_tensor_symmetric, torch.per_channel_symmetric]


def _is_float_qparams(qscheme: 'torch.qscheme') -> bool:
    return qscheme in [
        torch.per_channel_affine_float_qparams,
    ]


def check_is_valid_config_dict(config_dict: Any, allowed_keys: Set[str],
                               dict_name: str) -> None:
    r""" Checks if the given config_dict has the correct keys
    Args:
      `config_dict`: dictionary whose keys we want to check
    """

    for k in config_dict.keys():
        if k not in allowed_keys:
            raise ValueError('Expected ' + dict_name +
                             ' to have the following keys: ' +
                             str(allowed_keys) + '. But found \'' + k +
                             '\' instead.')


def check_is_valid_qconfig_dict(qconfig_dict: Any) -> None:
    r""" Checks if the given qconfig_dict has the correct keys
    Args:
      `qconfig_dict`: dictionary whose keys we want to check
    """

    qconfig_dict_allowed_keys = {
        '', 'object_type', 'module_name_regex', 'module_name',
        'module_name_object_type_order'
    }
    check_is_valid_config_dict(qconfig_dict, qconfig_dict_allowed_keys,
                               'qconfig_dict')


def check_is_valid_prepare_custom_config_dict(
        prepare_custom_config_dict: Optional[Dict[str, Any]] = None) -> None:
    r""" Checks if the given prepare_custom_config_dict has the correct keys
    Args:
      `prepare_custom_config_dict`: customization configuration dictionary for
      quantization tool
    """
    if not prepare_custom_config_dict:
        return

    prepare_custom_config_dict_allowed_keys = {
        'standalone_module_name', 'standalone_module_class',
        'float_to_observed_custom_module_class', 'non_traceable_module_name',
        'non_traceable_module_class', 'input_quantized_idxs',
        'output_quantized_idxs', 'preserved_attributes'
    }
    check_is_valid_config_dict(prepare_custom_config_dict,
                               prepare_custom_config_dict_allowed_keys,
                               'prepare_custom_config_dict')


def check_is_valid_convert_custom_config_dict(
        convert_custom_config_dict: Optional[Dict[str, Any]] = None) -> None:
    r""" Checks if the given convert_custom_config_dict has the correct keys
    Args:
      `convert_custom_config_dict`: dictionary for custom configurations for
      convert function
    """
    if not convert_custom_config_dict:
        return

    convert_custom_config_dict_allowed_keys = {
        'observed_to_quantized_custom_module_class', 'preserved_attributes'
    }
    check_is_valid_config_dict(convert_custom_config_dict,
                               convert_custom_config_dict_allowed_keys,
                               'convert_custom_config_dict')


def check_is_valid_fuse_custom_config_dict(
        fuse_custom_config_dict: Optional[Dict[str, Any]] = None) -> None:
    r""" Checks if the given fuse_custom_config_dict has the correct keys
    Args:
      `fuse_custom_config_dict`: dictionary for custom configurations for
      fuse_fx
    """
    if not fuse_custom_config_dict:
        return

    fuse_custom_config_dict_allowed_keys = {'preserved_attributes'}
    check_is_valid_config_dict(fuse_custom_config_dict,
                               fuse_custom_config_dict_allowed_keys,
                               'fuse_custom_config_dict')


def get_custom_module_class_keys(custom_config_dict,
                                 custom_config_dict_key) -> List[Any]:
    r""" Get all the unique custom module keys in the custom config dict
    e.g.
    Input:
    custom_config_dict = {
        "float_to_observed_custom_module_class": {
           "static": {
               CustomModule1: ObservedCustomModule
           },
           "dynamic": {
               CustomModule2: DynamicObservedCustomModule
           },
           "weight_only": {
               CustomModule3: WeightOnlyObservedCustomModule
           },
        },
    }
    Output:
    # extract all the keys in "static", "dynamic" and "weight_only" dict
    [CustomModule1, CustomModule2, CustomModule3]
    """
    # using set to dedup
    float_custom_module_classes: Set[Any] = set()
    custom_module_mapping = custom_config_dict.get(custom_config_dict_key, {})
    for quant_mode in ['static', 'dynamic', 'weight_only']:
        quant_mode_custom_module_config = custom_module_mapping.get(
            quant_mode, {})
        quant_mode_custom_module_classes = set(
            quant_mode_custom_module_config.keys())
        float_custom_module_classes |= quant_mode_custom_module_classes
    return list(float_custom_module_classes)
