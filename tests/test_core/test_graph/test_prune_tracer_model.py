# Copyright (c) OpenMMLab. All rights reserved.
import multiprocessing as mp
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from functools import partial
from unittest import TestCase

import torch
from mmengine import MMLogger

from mmrazor.models.task_modules.tracer.channel_analyzer import ChannelAnalyzer
from ...data.model_library import ModelGenerator
from ...data.tracer_passed_models import (PassedModelManager,
                                          backward_passed_library,
                                          fx_passed_library)
from ...utils import SetTorchThread

sys.setrecursionlimit(int(pow(2, 20)))
# test config

DEVICE = torch.device('cpu')
FULL_TEST = os.getenv('FULL_TEST') == 'true'
try:
    MP = int(os.getenv('MP'))
except Exception:
    MP = 1

DEBUG = os.getenv('DEBUG') == 'true'
if DEBUG:
    import logging
    logger = MMLogger.get_current_instance()
    logger.handlers[0].setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)

if MP > 1:
    POOL_SIZE = MP
    TORCH_THREAD_SIZE = mp.cpu_count() // POOL_SIZE
    torch.set_num_interop_threads(TORCH_THREAD_SIZE)
else:
    POOL_SIZE = 1
    TORCH_THREAD_SIZE = -1

print(f'DEBUG: {DEBUG}')
print(f'FULL_TEST: {FULL_TEST}')
print(f'POOL_SIZE: {POOL_SIZE}')
print(f'TORCH_THREAD_SIZE: {TORCH_THREAD_SIZE}')

# tools for tesing

# test functions for mp


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


def _test_a_model(Model, tracer_type='fx'):
    start = time.time()

    try:
        print(f'test {Model}.')
        model = Model.init_model()
        model.eval()
        if tracer_type == 'fx':
            tracer_type = 'FxTracer'
        elif tracer_type == 'backward':
            tracer_type = 'BackwardTracer'
        else:
            raise NotImplementedError()

        tracer = ChannelAnalyzer(
            tracer_type=tracer_type,
            demo_input={
                'type': 'DefaultDemoInput',
                'scope': Model.scope
            })
        with time_limit(60):
            unit_configs = tracer.analyze(model)

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
        from mmrazor import digit_version
        if digit_version(torch.__version__) < digit_version('1.12.0'):
            self.skipTest('version of torch < 1.12.0')
        TestData = fx_passed_library.include_models(FULL_TEST)

        with SetTorchThread(TORCH_THREAD_SIZE):
            if POOL_SIZE != 1:
                with ProcessPoolExecutor(POOL_SIZE) as p:
                    result = p.map(
                        partial(_test_a_model, tracer_type='fx'), TestData)

            else:
                result = map(
                    partial(_test_a_model, tracer_type='fx'), TestData)
        result = list(result)
        self.report(result, fx_passed_library, 'fx')

    def test_init_from_backward_tracer(self) -> None:
        TestData = backward_passed_library.include_models(FULL_TEST)
        with SetTorchThread(TORCH_THREAD_SIZE):
            if POOL_SIZE != 1:
                with ProcessPoolExecutor(POOL_SIZE) as p:
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
        untest_models = model_manager.uninclude_models(full_test=FULL_TEST)
        for model in untest_models:
            print(f'\t{model}')

        # short summary
        short_passed = set(
            [ModelGenerator.get_short_name(res[0]) for res in passd_test])

        short_unpassed = set(
            [ModelGenerator.get_short_name(res[0]) for res in unpassd_test])

        short_untest = set([model.short_name for model in untest_models])

        for name in short_unpassed:
            if name in short_passed:
                short_passed.remove(name)

        print('Short Summary:')
        print('Passed\n', short_passed)
        print('Unpassed\n', short_unpassed)
        print('Untest\n', short_untest)
