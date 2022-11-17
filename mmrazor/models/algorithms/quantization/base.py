# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.model import MMDistributedDataParallel
from mmengine.structures import BaseDataElement
from torch import nn
from torch.ao.quantization import FakeQuantizeBase
from torch.fx import GraphModule

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
        Args:
        architecture (dict | :obj:`BaseModel`): The config of
            :class:`BaseModel` or built model.
        quantizer (dict | :obj:`BaseModel`): The config of
            :class:`BaseQuantizer` or built model.
        data_preprocessor (dict | torch.nn.Module | None): The pre-process
            config of :class:`BaseDataPreprocessor`. Defaults to None.
        init_cfg (dict): The weight initialized config for
            :class:`BaseModule`.
    """

    def __init__(self,
                 architecture,
                 quantizer,
                 export_mode: str = 'predict',
                 qmodel_modes: List[str] = ['tensor', 'predict', 'loss'],
                 data_preprocessor=None,
                 init_cfg=None, sync=True):

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')
        super().__init__(architecture, data_preprocessor, init_cfg)
        self.quantizer = MODELS.build(quantizer)
        self._observers_enabled = True
        self._fake_quants_enabled = True
        self.export_mode = export_mode
        self.sync = sync
        import copy
        self.qmodels = self._build_qmodels(copy.deepcopy(self.architecture), qmodel_modes)

    def _build_qmodels(self, model, trace_modes):

        def _get_shared_fake_quantized(prefix, module):
            for name, child in module._modules.items():
                if module is None:
                    continue
                module_name = f'{prefix}{name}'
                if isinstance(child, FakeQuantizeBase):
                    share_fake_quantizes[module_name] = child
                else:
                    _get_shared_fake_quantized(f'{module_name}.', child)

        def _replace_fake_quantizes(prefix, module):
            for name, child in module._modules.items():
                if module is None:
                    continue
                module_name = f'{prefix}{name}'
                if isinstance(child, FakeQuantizeBase):
                    new_module = share_fake_quantizes[module_name]
                    setattr(module, name, new_module)
                else:
                    _replace_fake_quantizes(f'{module_name}.', child)

        qmodels = nn.ModuleDict()

        self.quantizer._swap_ff_with_fxff(model)
        tracer = self.quantizer.tracer

        for mode in trace_modes:
            concrete_args = {'mode': mode}
            qmodel = GraphModule(
                model, tracer.trace(model, concrete_args=concrete_args))
            qmodels[mode] = self.quantizer.prepare(model, qmodel)

        #  different modes of qmodels share the same fake_quantizes, so that
        #  the scale and zero_point in fake_quantizes are the same among
        #  different mode.
        share_fake_quantizes = dict()
        _get_shared_fake_quantized('', qmodels[self.export_mode])

        for mode, qmodel in qmodels.items():
            if mode == self.export_mode or not self.sync:
                continue
            _replace_fake_quantizes('', qmodel)

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

    def convert(self):
        qmodel = self.qmodels[self.export_mode]
        return self.quantizer.convert(qmodel)

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
    """DDPwapper for autoslim."""

    def __init__(self,
                 *,
                 device_ids: Optional[Union[List, int, torch.device]] = None,
                 **kwargs) -> None:
        if device_ids is None:
            if os.environ.get('LOCAL_RANK') is not None:
                device_ids = [int(os.environ['LOCAL_RANK'])]
        super().__init__(device_ids=device_ids, **kwargs)

    def calibrate_step(self, data):
        return self.module.calibrate_step(data)

    @property
    def state(self):
        return (self.module._observers_enabled,
                self.module._fake_quants_enabled)

    @state.setter
    def state(self, state: Tuple[bool]):
        self.module.state = state

    def convert(self):
        return self.module.convert()
