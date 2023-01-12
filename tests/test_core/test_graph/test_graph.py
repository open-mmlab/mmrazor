# Copyright (c) OpenMMLab. All rights reserved.
import sys
from unittest import TestCase

import torch

sys.setrecursionlimit(int(1e8))

DEVICE = torch.device('cpu')


class TestGraph(TestCase):
    pass
    # def test_init_from_fx_tracer(self) -> None:
    #     TestData = BackwardPassedModelManager.include_models()
    #     with SetTorchThread(1):
    #         with mp.Pool() as p:
    #             result = p.map(_test_init_from_fx_tracer, TestData)
    #     for res, model in zip(result, TestData):
    #         with self.subTest(model=model):
    #             self.assertTrue(res[0], res[1])

    # def test_init_from_backward_tracer(self) -> None:
    #     TestData = FxPassedModelManager.include_models()
    #     with SetTorchThread(1) as _:
    #         with mp.Pool() as p:
    #             result = p.map(_test_init_from_backward_tracer, TestData)
    #     for res, model in zip(result, TestData):
    #         # test_init_from_backward_tracer(model)
    #         with self.subTest(model=model):
    #             self.assertTrue(res[0], res[1])
