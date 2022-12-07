# Copyright (c) OpenMMLab. All rights reserved.
import unittest
from functools import partial

from mmcls.models.classifiers.image import ImageClassifier

from mmrazor.utils import get_placeholder

try:
    from torch.fx.graph_module import GraphModule
except ImportError:
    GraphModule = get_placeholder('torch>=1.12')
import torch

from mmrazor import digit_version
from mmrazor.models.task_modules.demo_inputs import DefaultDemoInput
from mmrazor.models.task_modules.tracer.fx_tracer import FxTracer
from ...data.models import UntracableModel

MODELS = [
    UntracableModel,
    partial(
        ImageClassifier,
        backbone=dict(type='mmrazor.UntracableBackBone'),
        head=dict(
            type='mmrazor.LinearHeadForTest',
            in_channel=16,
        )),
]


class TestFxTracer(unittest.TestCase):

    def test_model(self):
        if digit_version(torch.__version__) < digit_version('1.12.0'):
            self.skipTest('version of torch < 1.12.0')

        for Model in MODELS:
            with self.subTest(model=Model):
                model = Model()

                tracer = FxTracer()
                demo_input = DefaultDemoInput()
                inputs = demo_input.get_data(model)

                if isinstance(inputs, dict):
                    # args = copy.copy(inputs)
                    # args.pop('inputs')
                    # args['mode'] = 'tensor'
                    args = {'mode': 'tensor'}
                    torch_graph = tracer.trace(model, concrete_args=args)
                else:
                    torch_graph = tracer.trace(model)
                print(model)

                print(torch_graph)

                graph_module = GraphModule(model, torch_graph)
                print(graph_module)
                print(graph_module.code)

                inputs = demo_input.get_data(model)
                if isinstance(inputs, dict):
                    inputs['mode_1'] = inputs['mode']
                    inputs.pop('mode')
                    graph_module(**inputs)
                else:
                    graph_module(inputs)
