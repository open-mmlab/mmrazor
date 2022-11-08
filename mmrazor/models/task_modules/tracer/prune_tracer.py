# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin
from mmrazor.registry import TASK_UTILS
from mmrazor.structures.graph import BaseGraph, ModuleGraph
from mmrazor.structures.graph.channel_graph import (
    ChannelGraph, default_channel_node_converter)
from mmrazor.structures.graph.module_graph import (FxTracerToGraphConverter,
                                                   PathToGraphConverter)
from .backward_tracer import BackwardTracer
from .fx_tracer import CostumFxTracer
from .loss_calculator.sum_loss_calculator import SumPseudoLoss
from .razor_tracer import FxBaseNode, RazorFxTracer


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
        if self.tracer_type == 'BackwardTracer':
            path_list = self.tracer.trace(model)
            module_graph: ModuleGraph = PathToGraphConverter(path_list,
                                                             model).graph
        elif self.tracer_type == 'FxTracer':
            fx_graph = self.tracer.trace(model)
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
