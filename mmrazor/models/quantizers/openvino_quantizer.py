# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Optional, Tuple, Union

import torch

try:
    from torch.ao.quantization import disable_observer
except ImportError:
    from mmrazor.utils import get_placeholder
    disable_observer = get_placeholder('torch>=1.13')

from mmrazor.registry import MODELS
from mmrazor.structures.quantization import QConfigHandler
from .native_quantizer import NativeQuantizer


@MODELS.register_module()
class OpenVINOQuantizer(NativeQuantizer):
    """Quantizer for quantizing and deploying to Openvino backend.

    Each backend has its own features, for reducing the gap of quantized
    performance between before and after deployment as possible, we should
    match the backend's features in quantization.

    Openvino's some important features about quantization is as follows:
    * support_w_mode = ('per_tensor', 'per_channel')
    * support_a_mode = ('per_tensor')
    * weight range should be symmetric, such as int 8 is [-127, 127] rather
    than [-128, 127]
    """

    @property
    def backend(self):
        """The backend to deploy, also the key of the corresponding backend
        config."""
        return 'openvino'

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

    def prepare_for_mmdeploy(self,
                             model: torch.nn.Module,
                             dummy_input: Tuple = (1, 3, 224, 224),
                             checkpoint: Optional[str] = None):
        """Prepare for deploy to the backend with mmdeploy, which will be used
        in mmdeploy, and usually includes as follows:

        1. prepare for the float model rewritten by mmdeploy.
        2. load checkpoint consists of float weight and quantized params in
        mmrazor.
        3. post process weight fakequant for exporting .onnx that meet
        the backend's requirement.
        """
        self.convert_batchnorm2d(model)
        observed_model = self.prepare(model)
        if dummy_input is not None:
            observed_model(torch.randn(dummy_input))
        if checkpoint is not None:
            observed_model.load_state_dict(
                torch.load(checkpoint)['state_dict'])
        self.post_process_for_deploy(observed_model, keep_w_fake_quant=True)

        observed_model.apply(disable_observer)

        return observed_model

    def export_onnx(self,
                    model: Union[torch.nn.Module, torch.jit.ScriptModule,
                                 torch.jit.ScriptFunction],
                    args: Union[Tuple[Any, ...], torch.Tensor],
                    output_path: str,
                    opset_version: Optional[int] = 11,
                    **kwargs):
        """Export the onnx model that can be deployed to OpenVino backend."""

        symbolic_output_path = output_path.replace('.onnx', '_symbolic.onnx')
        torch.onnx.export(
            model,
            args,
            symbolic_output_path,
            opset_version=opset_version,
            **kwargs)

        from .exporters import OpenVinoQuantizeExportor
        exporter = OpenVinoQuantizeExportor(symbolic_output_path, output_path)
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
