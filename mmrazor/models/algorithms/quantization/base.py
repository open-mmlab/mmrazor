# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.structures import BaseDataElement
from torch.fx import GraphModule

from mmrazor.registry import MODELS
from ..base import BaseAlgorithm

LossResults = Dict[str, torch.Tensor]
TensorResults = Union[Tuple[torch.Tensor], torch.Tensor]
PredictResults = List[BaseDataElement]
ForwardResults = Union[LossResults, TensorResults, PredictResults]

GRAPH_MODES = ['tensor', 'loss', 'predict']


@MODELS.register_module()
class GeneralQuant(BaseAlgorithm):

    def __init__(self,
                 architecture,
                 quantizer,
                 data_preprocessor=None,
                 init_cfg=None,
                 modes=GRAPH_MODES):
        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')

        super().__init__(architecture, data_preprocessor, init_cfg)
        self.quantizer = MODELS.build(quantizer)
        self.observers_enabled = True
        self.fake_quants_enabled = True
        self.gen_graphs(self.architecture, GRAPH_MODES)

    def gen_graphs(self, model, modes=['tensor', 'loss', 'predict']):
        for mode in modes:
            assert mode in ['tensor', 'loss', 'predict']
        self.quantizer._swap_ff_with_fxff(model)
        tracer = self.quantizer.tracer
        for mode in modes:
            concrete_args = {'mode': mode}
            if mode == 'tensor':
                self.graph_tensor = GraphModule(
                    model, tracer.trace(model, concrete_args=concrete_args))
            if mode == 'loss':
                self.graph_loss = GraphModule(
                    model, tracer.trace(model, concrete_args=concrete_args))
            if mode == 'predict':
                self.graph_predict = GraphModule(
                    model, tracer.trace(model, concrete_args=concrete_args))

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[BaseDataElement]] = None,
                mode: str = 'tensor') -> ForwardResults:

        if mode == 'loss':
            return self.graph_loss(inputs, data_samples, mode)
        elif mode == 'tensor':
            return self.graph_tensor(inputs, data_samples, mode)
        elif mode == 'predict':
            return self.graph_predict(inputs, data_samples, mode)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def calib_step(self, data):
        data = self.data_preprocessor(data, False)
        return self._run_forward(data, mode='tensor')

    def prepare(self, mode='tensor'):
        assert mode in ['tensor', 'loss', 'predict']
        if mode == 'tensor':
            self.graph_tensor = self.quantizer.prepare(self, self.graph_tensor)
        elif mode == 'loss':
            self.graph_loss = self.quantizer.prepare(self, self.graph_loss)
        else:
            self.graph_predict = self.quantizer.prepare(
                self, self.graph_predict)

    def convert(self, mode='tensor'):
        assert mode in ['tensor', 'loss', 'predict']
        if mode == 'tensor':
            self.graph_tensor = self.quantizer.convert(self.graph_tensor)
        elif mode == 'loss':
            self.graph_loss = self.quantizer.convert(self.graph_loss)
        else:
            self.graph_predict = self.quantizer.convert(self.graph_predict)

    @property
    def state(self):
        return (self.observers_enabled, self.fake_quants_enabled)

    @state.setter
    def state(self, state):
        observers_enabled, fake_quants_enabled = state
        for name, submodule in self.architecture.named_modules():
            if isinstance(submodule, torch.quantization.FakeQuantize):
                if observers_enabled:
                    submodule.enable_observer()
                else:
                    submodule.disable_observer()

                if fake_quants_enabled:
                    submodule.enable_fake_quant()
                else:
                    submodule.disable_fake_quant()

        self.observers_enabled = observers_enabled
        self.fake_quants_enabled = fake_quants_enabled
