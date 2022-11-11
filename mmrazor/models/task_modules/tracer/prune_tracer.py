# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
from mmengine.model import BaseModel, revert_sync_batchnorm

from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin
from mmrazor.registry import TASK_UTILS
from mmrazor.structures.graph import BaseGraph, ModuleGraph
from mmrazor.structures.graph.channel_graph import (
    ChannelGraph, default_channel_node_converter)
from mmrazor.structures.graph.module_graph import (FxTracerToGraphConverter,
                                                   PathToGraphConverter)
from mmrazor.utils import get_placeholder
from .backward_tracer import BackwardTracer
from .fx_tracer import CustomFxTracer
from .loss_calculator.sum_loss_calculator import SumPseudoLoss
from .razor_tracer import FxBaseNode, RazorFxTracer

try:
    from mmdet.models import BaseDetector
except Exception:
    BaseDetector = get_placeholder('mmdet')

try:
    from mmcls.models import ImageClassifier
except Exception:
    ImageClassifier = get_placeholder('mmcls')
"""
- How to config PruneTracer using hard code
  - fxtracer
    - concrete args
      - PruneTracer.default_concrete_args_fun
    - leaf module
      - PruneTracer.default_leaf_modules
    - method
      - None
  - ChannelNode
    - channel_nodes.py
  - DynamicOp
        ChannelUnits
"""

# concrete args


def default_mm_concrete_args(model, input_shape):
    return {'mode': 'tensor'}


def default_concrete_args(model, input_shape):
    return {}


def det_concrete_args(model, input_shape):
    assert isinstance(model, BaseDetector)
    from mmdet.testing._utils import demo_mm_inputs
    data = demo_mm_inputs(1, [input_shape[1:]])
    data = model.data_preprocessor(data, False)
    data.pop('inputs')
    return data


@TASK_UTILS.register_module()
class PruneTracer:
    from mmcv.cnn.bricks import Scale

    default_leaf_modules = (
        # dynamic op
        DynamicChannelMixin,
        # torch
        nn.Conv2d,
        nn.Linear,
        nn.modules.batchnorm._BatchNorm,
        # mmcv
        Scale,
    )
    default_concrete_args_fun = {
        BaseDetector: det_concrete_args,
        ImageClassifier: default_mm_concrete_args,
        BaseModel: default_mm_concrete_args,
        nn.Module: default_concrete_args
    }

    def __init__(self,
                 input_shape=(1, 3, 224, 224),
                 tracer_type='BackwardTracer') -> None:

        self.input_shape = input_shape

        assert tracer_type in ['BackwardTracer', 'FxTracer']
        self.tracer_type = tracer_type
        if tracer_type == 'BackwardTracer':
            self.tracer = BackwardTracer(
                loss_calculator=SumPseudoLoss(input_shape=input_shape))
        elif tracer_type == 'FxTracer':
            self.tracer = CustomFxTracer(leaf_module=self.default_leaf_modules)
        else:
            raise NotImplementedError()

    def trace(self, model):
        # model = copy.deepcopy(model)
        model = revert_sync_batchnorm(model)
        model.eval()
        if self.tracer_type == 'BackwardTracer':
            path_list = self.tracer.trace(model)
            module_graph: ModuleGraph = PathToGraphConverter(path_list,
                                                             model).graph
        elif self.tracer_type == 'FxTracer':
            fx_graph = self._fx_trace(model)
            fx_graph.owning_module = model
            fx_graph.graph = BaseGraph[FxBaseNode]()
            base_graph = RazorFxTracer().parse_torch_graph(fx_graph)

            module_graph = FxTracerToGraphConverter(base_graph, model).graph
            module_graph._model = model
        else:
            raise NotImplementedError()

        module_graph.refresh_module_name()
        module_graph.check(fix=True)
        module_graph.check()

        channel_graph = ChannelGraph.copy_from(module_graph,
                                               default_channel_node_converter)
        channel_graph.check(fix=True)
        channel_graph.check()

        channel_graph.forward(self.input_shape[1])
        unit_configs = channel_graph.generate_units_config()

        return unit_configs

    def _fx_trace(self, model):
        args = self.get_concrete_args(model)
        return self.tracer.trace(model, concrete_args=args)

    def get_concrete_args(self, model):
        for module_type, concrete_args_fun in self.default_concrete_args_fun.items(  # noqa
        ):  # noqa
            if isinstance(model, module_type):
                return concrete_args_fun(model, self.input_shape)
        return {}

    def _det_input(self, model):
        assert isinstance(model, BaseDetector)
        from mmdet.testing._utils import demo_mm_inputs
        data = demo_mm_inputs(1, [self.input_shape[1:]])
        data = model.data_preprocessor(data, False)
        data['mode'] = 'tensor'
        data.pop('inputs')
        return data
