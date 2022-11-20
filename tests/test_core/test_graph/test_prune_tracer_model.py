# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing as mp
import os
import signal
import sys
import time
from contextlib import contextmanager
from functools import partial
from typing import List
from unittest import TestCase

import torch
import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.mixins import DynamicChannelMixin
from mmrazor.models.mutables.mutable_channel.units import \
    SequentialMutableChannelUnit
from mmrazor.models.task_modules.tracer.backward_tracer import BackwardTracer
from mmrazor.models.task_modules.tracer.fx_tracer import CustomFxTracer
from mmrazor.models.task_modules.tracer.prune_tracer import PruneTracer
from mmrazor.models.task_modules.tracer.razor_tracer import (FxBaseNode,
                                                             RazorFxTracer)
from mmrazor.structures.graph import BaseGraph, ModuleGraph
from mmrazor.structures.graph.channel_graph import (
    ChannelGraph, default_channel_node_converter)
from mmrazor.structures.graph.module_graph import (FxTracerToGraphConverter,
                                                   PathToGraphConverter)
from ...data.model_library import ModelGenerator
from ...data.tracer_passed_models import (PassedModelManager,
                                          backward_passed_library,
                                          fx_passed_library)
from ...utils import SetTorchThread

sys.setrecursionlimit(int(pow(2, 20)))
# test config
from mmrazor.models.task_modules.tracer import PruneTracer

DEVICE = torch.device('cpu')
FULL_TEST = os.getenv('FULL_TEST') == 'true'
MP = os.getenv('MP') == 'true'

DEBUG = os.getenv('DEBUG') == 'true'

if MP:
    POOL_SIZE = mp.cpu_count()
    TORCH_THREAD_SIZE = 1
    # torch.set_num_interop_threads(1)
else:
    POOL_SIZE = 1
    TORCH_THREAD_SIZE = -1

print(f'DEBUG: {DEBUG}')
print(f'FULL_TEST: {FULL_TEST}')
print(f'POOL_SIZE: {POOL_SIZE}')
print(f'TORCH_THREAD_SIZE: {TORCH_THREAD_SIZE}')

# tools for tesing

# test functions for mp


def _test_a_model(Model, tracer_type='fx'):
    start = time.time()

    try:
        model = Model.init_model()
        model.eval()
        if tracer_type == 'fx':
            tracer_type = 'FxTracer'
        elif tracer_type == 'backward':
            tracer_type = 'BackwardTracer'
        else:
            raise NotImplementedError()

        unit_configs = PruneTracer(
            tracer_type=tracer_type,
            demo_input={
                'type': 'DefaultDemoInput',
                'scope': Model.scope
            }).trace(model)
        out = len(unit_configs)
        print(f'test {Model} successful.')
        return Model.name, True, '', time.time() - start, out
    except Exception as e:
        if DEBUG:
            raise e
        else:
            print(f'test {Model} failed.')
            return Model.name, False, f'{e}', time.time() - start, -1


# TestCase


class TestTraceModel(TestCase):

    def test_init_from_fx_tracer(self) -> None:
        TestData = fx_passed_library.include_models(FULL_TEST)
        with SetTorchThread(TORCH_THREAD_SIZE):
            if POOL_SIZE != 1:
                with mp.Pool(POOL_SIZE) as p:
                    result = p.map(
                        partial(_test_a_model, tracer_type='fx'), TestData)
            else:
                result = map(
                    partial(_test_a_model, tracer_type='fx'), TestData)
        self.report(result, fx_passed_library, 'fx')

    def test_init_from_backward_tracer(self) -> None:
        TestData = backward_passed_library.include_models(FULL_TEST)
        with SetTorchThread(TORCH_THREAD_SIZE):
            if POOL_SIZE != 1:
                with mp.Pool(POOL_SIZE) as p:
                    result = p.map(
                        partial(_test_a_model, tracer_type='backward'),
                        TestData)
            else:
                result = map(
                    partial(_test_a_model, tracer_type='fx'), TestData)
        self.report(result, backward_passed_library, 'backward')

    def report(self, result, model_manager: PassedModelManager, fx_type='fx'):
        print()
        print(f'Trace model summary using {fx_type} tracer.')

        passd_test = [res for res in result if res[1] is True]
        unpassd_test = [res for res in result if res[1] is False]

        # long summary

        print(f'{len(passd_test)},{len(unpassd_test)},'
              f'{len(model_manager.uninclude_models(full_test=FULL_TEST))}')

        print('Passed:')
        print('\tmodel\ttime\tlen(mutable)')
        for model, passed, msg, used_time, out in passd_test:
            with self.subTest(model=model):
                print(f'\t{model}\t{int(used_time)}s\t{out}')
                self.assertTrue(passed, msg)

        print('UnPassed:')
        for model, passed, msg, used_time, out in unpassd_test:
            with self.subTest(model=model):
                print(f'\t{model}\t{int(used_time)}s\t{out}')
                print(f'\t\t{msg}')
                self.assertTrue(passed, msg)

        print('UnTest:')
        for model in model_manager.uninclude_models(full_test=FULL_TEST):
            print(f'\t{model}')

        # short summary
        print('Short Summary:')
        short_passed = set(
            [ModelGenerator.get_short_name(res[0]) for res in passd_test])

        print('Passed\n', short_passed)

        short_unpassed = set(
            [ModelGenerator.get_short_name(res[0]) for res in unpassd_test])
        print('Unpassed\n', short_unpassed)
