# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmrazor.models.task_modules.tracer.razor_tracer import CostumFxTracer
from ...data.models import UnTracableModel


class TestFxTracer(unittest.TestCase):

    def test_trace(self):
        tracer = CostumFxTracer(warp_method={torch: torch.arange})
        model = UnTracableModel()
        graph = tracer.trace(model)
        print(graph)

        print(torch.arange.__name__)
