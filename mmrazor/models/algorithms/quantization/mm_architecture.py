# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.model import MMDistributedDataParallel
from mmengine.runner import load_checkpoint
from mmengine.structures import BaseDataElement
from torch import nn

from mmrazor.models.task_modules.tracer import build_graphmodule
from mmrazor.registry import MODEL_WRAPPERS, MODELS
from ..base import BaseAlgorithm

try:
    from torch.ao.quantization import (FakeQuantizeBase, MinMaxObserver,
                                       PerChannelMinMaxObserver)
except ImportError:
    from mmrazor.utils import get_placeholder
    FakeQuantizeBase = get_placeholder('torch>=1.13')
    MinMaxObserver = get_placeholder('torch>=1.13')
    PerChannelMinMaxObserver = get_placeholder('torch>=1.13')

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
                 forward_modes=('tensor', 'predict', 'loss'),
                 float_checkpoint: Optional[str] = None,
                 input_shapes=(1, 3, 224, 224),
                 init_cfg=None):

        if data_preprocessor is None:
            data_preprocessor = getattr(architecture, 'data_preprocessor',
                                        dict())
        super().__init__(architecture, data_preprocessor, init_cfg)

        self.quantizer = MODELS.build(quantizer)
        self.input_shapes = input_shapes
        self.forward_modes = forward_modes

        # Replace syncbn and _BatchNormXd (in mmengine) with batchnorm2d
        self.quantizer.convert_batchnorm2d(self.architecture)

        if float_checkpoint:
            _ = load_checkpoint(self.architecture, float_checkpoint)
            self.architecture._is_init = True

        self.qmodels = self._build_qmodels(self.architecture)
        self.sync_qparams('tensor')
        self.reset_observer_and_fakequant_statistics(self)

    def reset_observer_and_fakequant_statistics(self, model):
        """Reset the statistics in observers and fake quantizers.

        The forward computation in `_build_qmodels` can modify the original
        statistics in observers and fake quantizers.
        """
        for module in model.modules():
            if isinstance(module, MinMaxObserver):
                module.reset_min_max_vals()
            elif isinstance(module, PerChannelMinMaxObserver):
                min_val = torch.rand(0, )
                max_val = torch.rand(0, )
                module.min_val.resize_(min_val.shape).copy_(min_val)
                module.max_val.resize_(max_val.shape).copy_(max_val)
            elif isinstance(module, FakeQuantizeBase):
                module.scale.data = torch.ones_like(module.scale)
                module.zero_point.data = torch.zeros_like(module.zero_point)

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

        for mode in self.forward_modes:
            concrete_args = {'mode': mode}
            traced_graph = tracer.trace(model, concrete_args=concrete_args)
            graph_module = build_graphmodule(model, traced_graph)
            observed_module = self.quantizer.prepare(model, graph_module)
            qmodels[mode] = observed_module

        is_training = qmodels['tensor'].training
        # Avoid random input changing bn's statistics
        qmodels['tensor'].eval()
        # Originally, the steps to train a qat model is as follows:
        # 1. build qmodels 2. convert the model to ddpmodel 3. forward backward
        # The shape of `scale` and `zero_point` can be modified during forward.
        # We initialize these parameters with per-tensor mode by default for
        # convenience. Their shape will be modified during forward if
        # per-channel mode is used. It's hacky. Hence we need to input a
        # dummy input to make sure the shape has been modified.
        device = next(qmodels.parameters()).device
        dummy_input = torch.randn(self.input_shapes).to(device)
        qmodels['tensor'](dummy_input, None, 'tensor')
        qmodels['tensor'].train(mode=is_training)

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
        return self._run_forward(data, mode='predict')


@MODEL_WRAPPERS.register_module()
class MMArchitectureQuantDDP(MMDistributedDataParallel):
    """DDPwapper for MMArchitectureQuant."""

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
        self.module.sync_qparams('tensor')
        self.module.reset_observer_and_fakequant_statistics(self)

    def calibrate_step(self, data):
        return self.module.calibrate_step(data)

    def sync_qparams(self, src):
        self.module.sync_qparams(src)
