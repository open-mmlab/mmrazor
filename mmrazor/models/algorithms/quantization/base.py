# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.model import MMDistributedDataParallel
from mmengine.runner import load_checkpoint
from mmengine.structures import BaseDataElement
from torch import nn
from torch.ao.quantization import FakeQuantizeBase

from mmrazor.models.task_modules import build_graphmodule
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from ..base import BaseAlgorithm

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]


@MODELS.register_module()
class MMArchitectureQuant(BaseAlgorithm):
    """General quantization.

    Args:
        architecture (dict | :obj:`BaseModel`): The config of
            :class:`BaseModel` or built model.
        quantizer (dict | :obj:`BaseModel`): The config of
            :class:`BaseQuantizer` or built model.
        export_mode (str): The mode of the model to be exported. Defaults to
            predict.
        qmodel_modes (list): The available mode of runner.
        data_preprocessor (dict | torch.nn.Module | None): The pre-process
            config of :class:`BaseDataPreprocessor`. Defaults to None.
        pretrained_ckpt (str, Optional): The path of pretrained checkpoint.
            Defaults to None.
        init_cfg (dict): The weight initialized config for
            :class:`BaseModule`.
    """

    

    def __init__(self,
                 architecture,
                 quantizer,
                 data_preprocessor=None,
                 forward_modes = ('tensor', 'predict', 'loss'),
                 float_checkpoint: Optional[str] = None,
                 input_shapes=(1, 3, 224, 224),
                 init_cfg=None):

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')
        super().__init__(architecture, data_preprocessor, init_cfg)
        if float_checkpoint:
            _ = load_checkpoint(self.architecture, float_checkpoint)
            self.architecture._is_init = True
        self.quantizer = MODELS.build(quantizer)
        self.input_shapes = input_shapes
        self.forward_modes = forward_modes

        self.qmodels = self._build_qmodels(self.architecture)

        self.sync_qparams('predict')

    def sync_qparams(self, src_mode):

        def traverse(module, prefix):
            for name, child in module._modules.items():
                if module is None:
                    continue
                child_name = f'{prefix}{name}'
                if isinstance(child, FakeQuantizeBase):
                    for name, param in child.named_parameters():
                        param_name = f'{child_name}.{name}'
                        src_param = src_state_dict[param_name]
                        if src_param.shape == param.shape:
                            param.data.copy_(src_param)
                        else:
                            requirs_grad = param.requires_grad
                            param.requires_grad = False
                            param.resize_(src_param.shape)
                            param.requires_grad = requirs_grad
                            param.data.copy_(src_param)
                    for name, buffer in child.named_buffers():
                        buffer_name = f'{child_name}.{name}'
                        src_buffer = src_state_dict[buffer_name]
                        if src_buffer.shape == buffer.shape:
                            buffer.data.copy_(src_buffer)
                        else:
                            buffer.resize_(src_buffer.shape)
                            buffer.data.copy_(src_buffer)
                else:
                    traverse(child, f'{child_name}.')

        src_state_dict = self.qmodels[src_mode].state_dict()
        for mode in self.forward_modes:
            if mode == src_mode:
                continue
            traverse(self.qmodels[mode], '')

    def _build_qmodels(self, model):

        qmodels = nn.ModuleDict()

        self.quantizer.swap_ff_with_fxff(model)
        tracer = self.quantizer.tracer
        import pdb;pdb.set_trace()
        for mode in self.forward_modes:
            concrete_args = {'mode': mode}
            traced_graph = tracer.trace(model, concrete_args=concrete_args)

            graph_mopdule = build_graphmodule(model, traced_graph)
            observed_module = self.quantizer.prepare(model, graph_mopdule)

            qmodels[mode] = observed_module

        dummy_input = torch.randn(self.input_shapes)
        qmodels['predict'](dummy_input, None, 'predict')

        return qmodels

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:

        if mode in self.qmodels:
            qmodel = self.qmodels[mode]
            return qmodel(inputs, data_samples, mode)
        else:
            return self.architecture(inputs, data_samples, mode)

    def calibrate_step(self, data):
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='tensor')


@MODEL_WRAPPERS.register_module()
class MMArchitectureQuantDDP(MMDistributedDataParallel):
    """DDPwapper for GeneralQuant."""

    def __init__(self,
                 *,
                 device_ids: Optional[Union[List, int, torch.device]] = None,
                 **kwargs) -> None:
        if device_ids is None:
            if os.environ.get('LOCAL_RANK') is not None:
                device_ids = [int(os.environ['LOCAL_RANK'])]
        super().__init__(device_ids=device_ids, **kwargs)
        # After moving all model parameters and buffers to the GPU
        # (`model.cuda()`), the buffers in model are different.
        self.module.qmodels = self.module._build_qmodels(
            self.module.architecture)

    def calibrate_step(self, data):
        return self.module.calibrate_step(data)

    def sync_qparams(self, src):
        self.module.sync_qparams(src)
