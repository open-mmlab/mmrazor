# Copyright (c) OpenMMLab. All rights reserved.
import logging
import time
from typing import Tuple, Union

import torch
from mmengine.logging import print_log


def get_model_latency(model: torch.nn.Module,
                      input_shape: Tuple = (1, 3, 224, 224),
                      unit: str = 'ms',
                      as_strings: bool = False,
                      max_iter: int = 100,
                      num_warmup: int = 5,
                      log_interval: int = 100,
                      repeat_num: int = 1) -> Union[float, str]:
    """Repeat speed measure for multi-times to get more precise results.

    Args:
        model (torch.nn.Module): The measured model.
        input_shape (tuple): Input shape (including batchsize) used for
            calculation. Default to (1, 3, 224, 224).
        unit (str): Unit of latency in string format. Default to 'ms'.
        as_strings (bool): Output latency counts in a string form.
            Default to False.
        max_iter (Optional[int]): Max iteration num for the measurement.
            Default to 100.
        num_warmup (Optional[int]): Iteration num for warm-up stage.
            Default to 5.
        log_interval (Optional[int]): Interval num for logging the results.
            Default to 100.
        repeat_num (Optional[int]): Num of times to repeat the measurement.
            Default to 1.

    Returns:
        latency (Union[float, str]): The measured inference speed of the model.
            if ``as_strings=True``, it will return latency in string format.
    """
    assert repeat_num >= 1

    fps_list = []

    for _ in range(repeat_num):
        fps_list.append(
            _get_model_latency(model, input_shape, max_iter, num_warmup,
                               log_interval))

    latency = round(1000 / fps_list[0], 1)

    if repeat_num > 1:
        _fps_list = [round(fps, 1) for fps in fps_list]
        times_per_img_list = [round(1000 / fps, 1) for fps in fps_list]
        _mean_fps = sum(_fps_list) / len(_fps_list)
        mean_times_per_img = sum(times_per_img_list) / len(times_per_img_list)
        print_log(
            f'Overall fps: {_fps_list}[{_mean_fps:.1f}] img / s, '
            f'times per image: '
            f'{times_per_img_list}[{mean_times_per_img:.1f}] ms/img',
            logger='current',
            level=logging.DEBUG)
        latency = mean_times_per_img

    if as_strings:
        latency = str(latency) + ' ' + unit  # type: ignore

    return latency


def _get_model_latency(model: torch.nn.Module,
                       input_shape: Tuple = (1, 3, 224, 224),
                       max_iter: int = 100,
                       num_warmup: int = 5,
                       log_interval: int = 100) -> float:
    """Measure inference speed on GPU devices.

    Args:
        model (torch.nn.Module): The measured model.
        input_shape (tuple): Input shape (including batchsize) used for
            calculation. Default to (1, 3, 224, 224).
        max_iter (Optional[int]): Max iteration num for the measurement.
            Default to 100.
        num_warmup (Optional[int]): Iteration num for warm-up stage.
            Default to 5.
        log_interval (Optional[int]): Interval num for logging the results.
            Default to 100.

    Returns:
        fps (float): The measured inference speed of the model.
    """
    # the first several iterations may be very slow so skip them
    pure_inf_time = 0.0
    fps = 0.0
    data = dict()
    if next(model.parameters()).is_cuda:
        device = 'cuda'
    else:
        raise NotImplementedError('To use cpu to test latency not supported.')

    # benchmark with {max_iter} image and take the average
    for i in range(1, max_iter):
        if device == 'cuda':
            data = torch.rand(input_shape).cuda()
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print_log(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    logger='current',
                    level=logging.DEBUG)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print_log(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                logger='current',
                level=logging.DEBUG)
            break

        torch.cuda.empty_cache()

    return fps
