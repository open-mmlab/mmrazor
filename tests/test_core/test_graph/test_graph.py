# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
from unittest import TestCase

import torch

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.structures.graph import ModuleGraph
from ...data.models import Icep  # noqa
from ...data.models import MultipleUseModel  # noqa
from ...data.models import Xmodel  # noqa
from ...data.models import (AddCatModel, ConcatModel, DwConvModel,
                            ExpandLineModel, GroupWiseConvModel, LineModel,
                            ModelLibrary, MultiBindModel, MultiConcatModel,
                            MultiConcatModel2, ResBlock)

FULL_TEST = os.getenv('FULL_TEST') == 'true'

sys.setrecursionlimit(int(1e8))


def is_dynamic_op_fx(module, name):
    return isinstance(module, DynamicChannelMixin)


class ToyCNNPseudoLoss:

    def __call__(self, model):
        pseudo_img = torch.rand(2, 3, 16, 16)
        pseudo_output = model(pseudo_img)
        return pseudo_output.sum()


class TestGraph(TestCase):

    @classmethod
    def backward_tracer_passed_models(cls):
        '''MultipleUseModel: backward tracer can't distinguish multiple use and
        first bind then use.'''
        default_models = [
            LineModel,
            ResBlock,
            AddCatModel,
            ConcatModel,
            MultiConcatModel,
            MultiConcatModel2,
            GroupWiseConvModel,
            Xmodel,
            # MultipleUseModel,  # bug
            # Icep, bug
            ExpandLineModel,
            MultiBindModel,
            DwConvModel
        ]
        """
        googlenet return a tuple when training, so it
            should trace in eval mode
        """

        torch_models_includes = [
            'alexnet',
            'densenet',
            'efficientnet',
            'googlenet',
            # 'inception',  bug
            'mnasnet',
            'mobilenet',
            'regnet',
            'resnet',
            'resnext',
            # 'shufflenet',     # bug
            'squeezenet',
            'vgg',
            'wide_resnet',
        ]
        model_library = ModelLibrary(torch_models_includes)

        models = default_models + model_library.export_models(
        ) if FULL_TEST else default_models
        return models

    def test_init_from_backward_tracer(self) -> None:
        TestData = self.backward_tracer_passed_models()

        for data in TestData:
            with self.subTest(data=data):
                model = data()
                model.eval()
                graph = ModuleGraph.init_from_backward_tracer(model)

                # check channels
                self._valid_graph(graph)

    def _valid_graph(self, graph: ModuleGraph):
        try:
            graph.check()
        except Exception as e:
            self.fail(str(e) + '\n' + str(graph))
