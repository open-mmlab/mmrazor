# Copyright (c) OpenMMLab. All rights reserved.

import copy
from typing import Dict, List

import torch
import torch.nn as nn
from mmengine.utils.dl_utils import mmcv_full_available

from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel import (
    MutableChannelUnit, SequentialMutableChannelUnit)
from mmrazor.models.mutables.mutable_channel.units.utils import find_mutable
from mmrazor.registry import TASK_UTILS
from mmrazor.structures.graph import BaseGraph, ModuleGraph
from mmrazor.structures.graph.channel_graph import (
    ChannelGraph, default_channel_node_converter)
from mmrazor.structures.graph.module_graph import (FxTracerToGraphConverter,
                                                   PathToGraphConverter)
from mmrazor.utils import demo_inputs
from .backward_tracer import BackwardTracer
from .fx_tracer import CustomFxTracer
from .loss_calculator.sum_loss_calculator import SumPseudoLoss
from .razor_tracer import FxBaseNode, RazorFxTracer
"""
- How to config PruneTracer using hard code
  - fxtracer
    - concrete args
      - demo_inputs
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


def revert_sync_batchnorm(module: nn.Module) -> nn.Module:
    """Helper function to convert all `SyncBatchNorm` (SyncBN) and
    `mmcv.ops.sync_bn.SyncBatchNorm`(MMSyncBN) layers in the model to
    `BatchNormXd` layers.

    Adapted from @kapily's work:
    (https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547)

    Args:
        module (nn.Module): The module containing `SyncBatchNorm` layers.

    Returns:
        module_output: The converted module with `BatchNormXd` layers.
    """
    module_output = module
    module_checklist = [torch.nn.modules.batchnorm.SyncBatchNorm]

    if mmcv_full_available():
        from mmcv.ops import SyncBatchNorm
        module_checklist.append(SyncBatchNorm)

    if isinstance(module, tuple(module_checklist)):
        module_output = nn.BatchNorm2d(module.num_features, module.eps,
                                       module.momentum, module.affine,
                                       module.track_running_stats)
        if module.affine:
            # no_grad() may not be needed here but
            # just to be consistent with `convert_sync_batchnorm()`
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        module_output.training = module.training
        # qconfig exists in quantized models
        if hasattr(module, 'qconfig'):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output


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

        return self._find_mutable_units(model, unit_configs)

    def _fx_trace(self, model):
        args = demo_inputs(model, self.input_shape)
        if isinstance(args, dict):
            args.pop('inputs')
            return self.tracer.trace(model, concrete_args=args)
        else:
            return self.tracer.trace(model)

    def _find_mutable_units(self, model, units_config: Dict):
        model = copy.deepcopy(model)
        units: List[SequentialMutableChannelUnit] = [
            SequentialMutableChannelUnit.init_from_cfg(model, cfg)
            for cfg in units_config.values()
        ]
        for unit in units:
            unit.prepare_for_pruning(model)
        mutable_units = [unit for unit in units if unit.is_mutable]
        inputs = demo_inputs(model, [1, 3, 224, 224])
        model.eval()

        if isinstance(inputs, dict):
            inputs['mode'] = 'loss'
            template_output = model(**inputs)
        else:
            template_output = model(inputs)

        mutable_units = find_mutable(model, mutable_units, units,
                                     template_output)
        mutable_unit_config = {}
        for unit in mutable_units:
            mutable_unit_config[
                unit.name] = MutableChannelUnit.config_template(
                    unit, with_channels=True, with_init_args=True)
        return mutable_unit_config
