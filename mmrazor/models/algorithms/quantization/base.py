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
class GeneralQuant(BaseAlgorithm):
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
                 export_mode: str = 'predict',
                 qmodel_modes: List[str] = ['tensor', 'predict', 'loss'],
                 data_preprocessor=None,
                 pretrained_ckpt: Optional[str] = None,
                 init_cfg=None):

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')
        super().__init__(architecture, data_preprocessor, init_cfg)
        if pretrained_ckpt:
            _ = load_checkpoint(self.architecture, pretrained_ckpt)
            self.architecture._is_init = True
        self.quantizer = MODELS.build(quantizer)
        self._observers_enabled = True
        self._fake_quants_enabled = True
        self.export_mode = export_mode
        self.qmodel_modes = qmodel_modes
        self.qmodels = self._build_qmodels(self.architecture)

    def sync_param(self):

        def traverse(module, prefix):
            for name, child in module._modules.items():
                if module is None:
                    continue
                module_name = f'{prefix}{name}'
                if isinstance(child, FakeQuantizeBase):
                    for name, param in child.named_parameters():
                        param.data.copy_(self.qmodels['loss'].state_dict()
                                         [f'{module_name}.{name}'])
                    for name, buffer in child.named_buffers():
                        buffer.data.copy_(self.qmodels['loss'].state_dict()
                                          [f'{module_name}.{name}'])
                else:
                    traverse(child, f'{module_name}.')

        for mode in self.qmodel_modes:
            if mode == 'loss':
                continue
            traverse(self.qmodels[mode], '')

    def _build_qmodels(self, model):

        qmodels = nn.ModuleDict()

        self.quantizer._swap_ff_with_fxff(model)
        tracer = self.quantizer.tracer

        for mode in self.qmodel_modes:
            concrete_args = {'mode': mode}
            traced_graph = tracer.trace(model, concrete_args=concrete_args)

            qmodel = build_graphmodule(model, traced_graph)
            qmodels[mode] = self.quantizer.prepare(model, qmodel)

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
        self.state = (1, 0)
        return self._run_forward(data, mode='tensor')

    def convert(self, mode='predict'):
        qmodel = self.qmodels[self.export_mode]
        self.qmodels[mode] = self.quantizer.convert(qmodel)

    @property
    def state(self):
        return (self._observers_enabled, self._fake_quants_enabled)

    @state.setter
    def state(self, state: Tuple[bool, bool]):
        observers_enabled, fake_quants_enabled = state
        qmodel = self.qmodels[self.export_mode]
        for submodule in qmodel.modules():
            if isinstance(submodule, torch.quantization.FakeQuantize):
                if observers_enabled:
                    submodule.enable_observer()
                else:
                    submodule.disable_observer()

                if fake_quants_enabled:
                    submodule.enable_fake_quant()
                else:
                    submodule.disable_fake_quant()

        self._observers_enabled = observers_enabled
        self._fake_quants_enabled = fake_quants_enabled


@MODEL_WRAPPERS.register_module()
class GeneralQuantDDP(MMDistributedDataParallel):
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

    @property
    def state(self):
        return (self.module._observers_enabled,
                self.module._fake_quants_enabled)

    @state.setter
    def state(self, state: Tuple[bool]):
        self.module.state = state

    def convert(self, mode='predict'):
        self.module.convert(mode)
        self.module.qmodels[mode].cuda()

    def sync_param(self):
        self.module.sync_param()
