# Copyright (c) OpenMMLab. All rights reserved.
"""
- How to config ChannelAnalyzer by hard code
  - fxtracer
    - demo_inputs
        ./mmrazor/models/task_modules/demo_inputs/default_demo_inputs.py
    - leaf module
      - ChannelAnalyzer.default_leaf_modules
    - method
      - ./mmrazor/models/task_modules/tracer/fx_tracer.py
  - ChannelNode
    - ./mmrazor/structures/graph/channel_nodes.py
  - DynamicOp
        ./mmrazor/models/architectures/dynamic_ops/bricks/dynamic_conv.py
"""
import copy
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
from mmcv.cnn.bricks import Scale
from mmengine.model.utils import revert_sync_batchnorm

from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel import (
    MutableChannelUnit, SequentialMutableChannelUnit)
from mmrazor.models.mutables.mutable_channel.units.utils import find_mutable
from mmrazor.registry import TASK_UTILS
from mmrazor.structures.graph import ModuleGraph
from mmrazor.structures.graph.channel_graph import (
    ChannelGraph, default_channel_node_converter)
from mmrazor.structures.graph.module_graph import (FxTracerToGraphConverter,
                                                   PathToGraphConverter)
from mmrazor.structures.graph.pseudo_fx_graph import parse_torch_graph
from mmrazor.utils import print_log
from ..demo_inputs import BaseDemoInput, DefaultDemoInput
from .backward_tracer import BackwardTracer
from .fx_tracer import MMFxTracer
from .loss_calculator.sum_loss_calculator import SumPseudoLoss


@TASK_UTILS.register_module()
class ChannelAnalyzer:
    """The tracer for pruning. It return the configs of MutableChannelUnits as
    result.

    Args:
        demo_input (Union[List, Dict, Tuple, BaseDemoInput], optional):
            The demo input for the model. demo_input can be one of
            input_shape(list), config of a demo input generator, a demoinput
            generator. Defaults to (1, 3, 224, 224).
        tracer_type (str, optional): str indicates which basic tracer to use.
            Defaults to 'BackwardTracer'.
    """
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
                 demo_input: Union[List, Dict, Tuple,
                                   BaseDemoInput] = (1, 3, 224, 224),
                 tracer_type='BackwardTracer') -> None:

        if isinstance(demo_input, dict):
            self.demo_input = TASK_UTILS.build(demo_input)
        elif isinstance(demo_input, list) or isinstance(demo_input, tuple):
            self.demo_input = DefaultDemoInput(demo_input, False)
        elif isinstance(demo_input, BaseDemoInput):
            self.demo_input = demo_input
        else:
            raise NotImplementedError(f'{type(demo_input)},{demo_input}')

        self.input_shape = demo_input

        assert tracer_type in ['BackwardTracer', 'FxTracer']
        self.tracer_type = tracer_type
        if tracer_type == 'BackwardTracer':
            self.tracer = BackwardTracer(
                loss_calculator=SumPseudoLoss(
                    input_shape=self.demo_input.input_shape))
        elif tracer_type == 'FxTracer':
            from mmrazor import digit_version
            assert digit_version(torch.__version__) >= digit_version(
                '1.12.0'
            ), 'Please install torch>=1.12.0, if you want to use fx tracer.'
            self.tracer = MMFxTracer(leaf_module=self.default_leaf_modules)
        else:
            raise NotImplementedError()

    def analyze(self, model):
        """Tracer the model, and return configs of channel dependency."""
        model = copy.deepcopy(model)
        model = revert_sync_batchnorm(model)
        model.eval()
        if self.tracer_type == 'BackwardTracer':
            path_list = self.tracer.trace(model)
            module_graph: ModuleGraph = PathToGraphConverter(path_list,
                                                             model).graph
        elif self.tracer_type == 'FxTracer':
            fx_graph = self._fx_trace(model)
            fx_graph.owning_module = model
            base_graph = parse_torch_graph(fx_graph)

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

        channel_graph.forward(self.demo_input.input_shape[1])
        unit_configs = channel_graph.generate_units_config()

        return self._find_mutable_units(model, unit_configs)

    def _fx_trace(self, model):
        """Tracer the model using fx tracer."""
        args = self.demo_input.get_data(model)
        if isinstance(args, dict):
            args.pop('inputs')
            args['mode'] = 'tensor'
            return self.tracer.trace(model, concrete_args=args)
        else:
            return self.tracer.trace(model)

    def _find_mutable_units(self, model: nn.Module, units_config: Dict):
        """Test the tracer result and filter unforwardable units."""
        model = copy.deepcopy(model).cpu()
        units: List[SequentialMutableChannelUnit] = [
            SequentialMutableChannelUnit.init_from_cfg(model, cfg)
            for cfg in units_config.values()
        ]
        for unit in units:
            unit.prepare_for_pruning(model)
        mutable_units = [unit for unit in units if unit.is_mutable]
        inputs = self.demo_input.get_data(model)
        model.eval()

        template_output = None
        if isinstance(inputs, dict):
            for mode in ['loss', 'tensor', 'predict']:
                try:
                    inputs['mode'] = mode
                    template_output = model(**inputs)
                    break
                except Exception as e:
                    print_log(f'Forward failed in {mode} mode as {e}')
        else:
            try:
                template_output = model(inputs)
            except Exception as e:
                print_log(f'Forward failed in as {e}')
        if template_output is None:
            raise Exception(
                'Forward failed, there may be an error in demo input.',
                f'{inputs}')
        mutable_units = find_mutable(model, mutable_units, units, inputs,
                                     template_output)
        mutable_unit_config = {}
        for unit in mutable_units:
            mutable_unit_config[
                unit.name] = MutableChannelUnit.config_template(
                    unit, with_channels=True, with_init_args=True)
        return mutable_unit_config
