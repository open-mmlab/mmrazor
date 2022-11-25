# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.registry import MODELS
from mmrazor.structures.quantization import DefaultQconfigs
from .base import CustomQuantizer


@MODELS.register_module()
class TensorRTQuantizer(CustomQuantizer):
    """Quantizer for TensorRT backend."""

    def __init__(self,
                 qconfig=DefaultQconfigs['tensorrt'],
                 is_qat=True,
                 skipped_methods=None,
                 prepare_custom_config_dict=None,
                 convert_custom_config_dict=None,
                 equalization_qconfig_dict=None,
                 _remove_qconfig=True,
                 init_cfg=None):
        super().__init__(qconfig, is_qat, skipped_methods,
                         prepare_custom_config_dict,
                         convert_custom_config_dict, equalization_qconfig_dict,
                         _remove_qconfig, init_cfg)
