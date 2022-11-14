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

DEVICE = torch.device('cpu')
FULL_TEST = os.getenv('FULL_TEST') == 'true'
MP = os.getenv('MP') == 'true'

DEBUG = os.getenv('DEBUG') == 'true'

if MP:
    POOL_SIZE = mp.cpu_count()
    TORCH_THREAD_SIZE = 1
    torch.set_num_interop_threads(1)
else:
    POOL_SIZE = 1
    TORCH_THREAD_SIZE = -1

print(f'DEBUG: {DEBUG}')
print(f'FULL_TEST: {FULL_TEST}')
print(f'POOL_SIZE: {POOL_SIZE}')
print(f'TORCH_THREAD_SIZE: {TORCH_THREAD_SIZE}')

# tools for tesing


@contextmanager
def time_limit(seconds, msg='', activated=(not DEBUG)):

    class TimeoutException(Exception):
        pass

    def signal_handler(signum, frame):
        if activated:
            raise TimeoutException(f'{msg} run over {seconds} s!')

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


# functional functions (need move to code)


def forward_units(model: ModelGenerator,
                  try_units: List[SequentialMutableChannelUnit],
                  units: List[SequentialMutableChannelUnit], template_output):
    model.eval()
    for unit in units:
        unit.current_choice = 1.0
    for unit in try_units:
        unit.current_choice = min(max(0.1, unit.sample_choice()), 0.9)
    x = torch.rand([1, 3, 224, 224]).to(DEVICE)
    tensors = model(x)
    model.assert_model_is_changed(template_output, tensors)


def find_mutable(model, try_units, units, template_tensors):
    if len(try_units) == 0:
        return []
    try:
        forward_units(model, try_units, units, template_tensors)
        return try_units
    except Exception as e:
        if len(try_units) == 1:
            print(f'{model} find an unmutable units.')
            print(f'{e}')
            print(try_units[0])
            return []
        else:
            num = len(try_units)
            return find_mutable(model, try_units[:num // 2], units,
                                template_tensors) + find_mutable(
                                    model, try_units[num // 2:], units,
                                    template_tensors)


class SumLoss:

    def __call__(self, model):
        img = torch.zeros([2, 3, 224, 224])
        y = model(img)
        return self.get_loss(y)

    def get_loss(self, output):
        if isinstance(output, torch.Tensor):
            return output.sum()
        elif isinstance(output, list) or isinstance(output, tuple):
            loss = 0
            for out in output:
                loss += self.get_loss(out)
            return loss
        elif isinstance(output, dict):
            return self.get_loss(list(output.values()))
        else:
            raise NotImplementedError(f'{type(output)}')


def is_dynamic_op_fx(module, name):
    from mmcv.cnn.bricks import Scale

    is_leaf = (
        isinstance(module, DynamicChannelMixin)
        or isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        or isinstance(module, nn.modules.batchnorm._BatchNorm)
        or isinstance(module, Scale))

    return is_leaf


# test functions for mp


def _test_tracer(model, tracer_type='fx'):

    def _test_fx_tracer(model):
        tracer = CustomFxTracer(leaf_module=PruneTracer.default_leaf_modules)
        return tracer.trace(model)

    def _test_backward_tracer(model):
        model.eval()
        tracer = BackwardTracer(loss_calculator=SumLoss())
        return tracer.trace(model)

    if tracer_type == 'fx':
        graph = _test_fx_tracer(model)
    else:
        graph = _test_backward_tracer(model)
    return graph


def _test_tracer_result_2_module_graph(model, tracer_res, tracer_type='fx'):

    def _fx_graph_2_module_graph(model, fx_graph):
        fx_graph.owning_module = model
        fx_graph.graph = BaseGraph[FxBaseNode]()
        base_graph = RazorFxTracer().parse_torch_graph(fx_graph)

        module_graph = FxTracerToGraphConverter(base_graph, model).graph
        module_graph._model = model
        module_graph.refresh_module_name()
        return module_graph

    def _path_2_module_graph(model, path_list):
        module_graph = PathToGraphConverter(path_list, model).graph
        module_graph.refresh_module_name()
        return module_graph

    if tracer_type == 'fx':
        graph = _fx_graph_2_module_graph(model, tracer_res)
    else:
        graph = _path_2_module_graph(model, tracer_res)
    return graph


def _test_units(units: List[SequentialMutableChannelUnit], model):
    x = torch.rand([1, 3, 224, 224]).to(DEVICE)
    model.eval()
    tensors_org = model(x)
    # prune
    for unit in units:
        unit.prepare_for_pruning(model)
    mutable_units = [unit for unit in units if unit.is_mutable]
    found_mutable_units = mutable_units
    # found_mutable_units = find_mutable(model, mutable_units, units,
    #                                    tensors_org)
    assert len(found_mutable_units) >= 1, \
        'len of mutable units should greater or equal than 0.'
    forward_units(model, found_mutable_units, units, tensors_org)
    return found_mutable_units


def _test_a_model(Model, tracer_type='fx'):
    start = time.time()

    try:
        Model.init_model()
        model = Model
        model.eval()
        print(f'test {Model} using {tracer_type} tracer.')
        """
        model
            -> fx_graph/path_list
            -> module_graph
            -> channel_graph
            -> units
        """
        with time_limit(10, 'trace'):
            tracer_result = _test_tracer(model, tracer_type)
            out = len(tracer_result.nodes if tracer_type ==
                      'fx' else tracer_result)

        with time_limit(10, 'to_module_graph'):
            module_graph: ModuleGraph = _test_tracer_result_2_module_graph(
                model, tracer_result, tracer_type)
            module_graph.check(fix=True)
            module_graph.check()
            out = len(module_graph)

        with time_limit(10, 'to channel graph'):
            channel_graph = ChannelGraph.copy_from(
                module_graph, default_channel_node_converter)
            channel_graph.check(fix=True)
            channel_graph.check()

        with time_limit(80, 'to units'):
            channel_graph.forward(3)
            units_config = channel_graph.generate_units_config()
            units = [
                SequentialMutableChannelUnit.init_from_cfg(model, cfg)
                for cfg in units_config.values()
            ]

        with time_limit(80, 'test units'):
            # get unit
            mutable_units = _test_units(units, model)
            out = len(mutable_units)

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
