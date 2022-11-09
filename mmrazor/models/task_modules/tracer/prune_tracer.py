# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmengine.model import BaseModel

from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin
from mmrazor.registry import TASK_UTILS
from mmrazor.structures.graph import BaseGraph, ModuleGraph
from mmrazor.structures.graph.channel_graph import (
    ChannelGraph, default_channel_node_converter)
from mmrazor.structures.graph.module_graph import (FxTracerToGraphConverter,
                                                   PathToGraphConverter)
from mmrazor.utils import get_placeholder
from .backward_tracer import BackwardTracer
from .fx_tracer import CostumFxTracer
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


def is_dynamic_op_fx(module, name):
    from mmcv.cnn.bricks import Scale

    is_leaf = (
        isinstance(module, DynamicChannelMixin)
        or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        or isinstance(module, nn.modules.batchnorm._BatchNorm)
        or isinstance(module, Scale))

    return is_leaf


@TASK_UTILS.register_module()
class PruneTracer:

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
            self.tracer = CostumFxTracer(is_extra_leaf_module=is_dynamic_op_fx)
        else:
            raise NotImplementedError()

    def trace(self, model):
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
        if isinstance(model, BaseDetector):
            data = self._det_input(model)
        elif isinstance(model, ImageClassifier):
            data = {'mode': 'tensor'}
        elif isinstance(model, BaseModel):
            data = {'mode': 'tensor'}
        else:
            data = {}

        return self.tracer.trace(model, concrete_args=data)

    def _det_input(self, model):
        assert isinstance(model, BaseDetector)
        from mmdet.testing._utils import demo_mm_inputs
        data = demo_mm_inputs(1, [self.input_shape[1:]])
        data = model.data_preprocessor(data, False)
        data['mode'] = 'tensor'
        data.pop('inputs')
        return data
