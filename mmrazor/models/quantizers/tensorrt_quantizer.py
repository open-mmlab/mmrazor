# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Tuple, Union

import torch

from mmrazor.registry import MODELS
from .native_quantizer import TorchNativeQuantizer


@MODELS.register_module()
class TensorRTQuantizer(TorchNativeQuantizer):
    """Quantizer for quantizing and deploying to TensorRT backend.

    Each backend has its own features, for reducing the gap of quantized
    performance between before and after deployment as possible, we should
    match the backend's features in quantization.

    TensorRT's some important features about quantization is as follows:
    * support_w_mode = ('per_tensor', 'per_channel')
    * support_a_mode = ('per_tensor')
    """

    @property
    def backend(self):
        """The backend to deploy, also the key of the corresponding backend
        config."""
        return 'tensorrt'

    @property
    def support_w_modes(self):
        """Supported quantization modes for weight about per_tensor or
        per_channel."""
        return ('per_tensor', 'per_channel')

    @property
    def support_a_modes(self):
        """Supported quantization modes for activation about per_tensor or
        per_channel."""
        return ('per_tensor')

    def export_onnx(self,
                    model: Union[torch.nn.Module, torch.jit.ScriptModule,
                                 torch.jit.ScriptFunction],
                    args: Union[Tuple[Any, ...], torch.Tensor],
                    output_path: str,
                    opset_version: Optional[int] = 13,
                    **kwargs):
        """Export the onnx model that can be deployed to OpenVino backend."""

        symbolic_output_path = output_path.replace('.onnx', '_symbolic.onnx')
        torch.onnx.export(
            model,
            args,
            symbolic_output_path,
            opset_version=opset_version,
            **kwargs)

        from .exporters import TensorRTExplicitExporter
        exporter = TensorRTExplicitExporter(symbolic_output_path, output_path)
        exporter.export()

    @property
    def module_prev_wo_fakequant(self):
        """Configurate the modules that their previous nodes are redundant
        fakequants."""
        return (torch.nn.ReLU6, torch.nn.Identity)

    @property
    def module_next_wo_fakequant(self):
        """Configurate the modules that their next nodes are redundant
        fakequants."""
        return (torch.nn.MaxPool2d, )

    @property
    def method_next_wo_fakequant(self):
        """Configurate the methods that their next nodes are redundant
        fakequants."""
        return ('flatten', )

    @property
    def op_prev_wo_fakequant(self):
        """Configurate the OPs that their previous nodes are redundant
        fakequants."""
        return ('output', )
