# Copyright (c) OpenMMLab. All rights reserved.
import logging
import time
from typing import Tuple

import torch
from mmengine.logging import print_log


def repeat_measure_inference_speed(model: torch.nn.Module,
                                   input_shape: Tuple = (1, 3, 224, 224),
                                   latency_max_iter: int = 100,
                                   latency_num_warmup: int = 5,
                                   latency_log_interval: int = 100,
                                   latency_repeat_num: int = 1,
                                   unit: str = 'ms',
                                   as_strings: bool = False) -> float:
    """Repeat speed measure for multi-times to get more precise results.

    Args:
        model (torch.nn.Module): The measured model.
        input_shape (tuple): Input shape (including batchsize) used for
            calculation. Default to (1, 3, 224, 224).
        latency_max_iter (Optional[int]): Max iteration num for the
            measurement. Default to 100.
        latency_num_warmup (Optional[int]): Iteration num for warm-up stage.
            Default to 5.
        latency_log_interval (Optional[int]): Interval num for logging the
            results. Default to 100.
        latency_repeat_num (Optional[int]): Num of times to repeat the
            measurement. Default to 1.

    Returns:
        fps (float): The measured inference speed of the model.
    """
    assert latency_repeat_num >= 1

    fps_list = []

    for _ in range(latency_repeat_num):

        fps_list.append(
            measure_inference_speed(model, input_shape, latency_max_iter,
                                    latency_num_warmup, latency_log_interval))

    if latency_repeat_num > 1:
        fps_list_ = [round(fps, 1) for fps in fps_list]
        times_per_img_list = [round(1000 / fps, 1) for fps in fps_list]
        mean_fps_ = sum(fps_list_) / len(fps_list_)
        mean_times_per_img = sum(times_per_img_list) / len(times_per_img_list)
        print_log(
            f'Overall fps: {fps_list_}[{mean_fps_:.1f}] img / s, '
            f'times per image: '
            f'{times_per_img_list}[{mean_times_per_img:.1f}] ms/img',
            logger='current',
            level=logging.DEBUG)
        return mean_times_per_img

    latency = round(1000 / fps_list[0], 1)
    return latency


def measure_inference_speed(model: torch.nn.Module,
                            input_shape: Tuple = (1, 3, 224, 224),
                            latency_max_iter: int = 100,
                            latency_num_warmup: int = 5,
                            latency_log_interval: int = 100) -> float:
    """Measure inference speed on GPU devices.

    Args:
        model (torch.nn.Module): The measured model.
        input_shape (tuple): Input shape (including batchsize) used for
            calculation. Default to (1, 3, 224, 224).
        latency_max_iter (Optional[int]): Max iteration num for the
            measurement. Default to 100.
        latency_num_warmup (Optional[int]): Iteration num for warm-up stage.
            Default to 5.
        latency_log_interval (Optional[int]): Interval num for logging the
            results. Default to 100.

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
    # benchmark with {latency_max_iter} image and take the average
    for i in range(1, latency_max_iter):
        if device == 'cuda':
            data = torch.rand(input_shape).cuda()
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= latency_num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % latency_log_interval == 0:
                fps = (i + 1 - latency_num_warmup) / pure_inf_time
                print_log(
                    f'Done image [{i + 1:<3}/ {latency_max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    logger='current',
                    level=logging.DEBUG)

        if (i + 1) == latency_max_iter:
            fps = (i + 1 - latency_num_warmup) / pure_inf_time
            print_log(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                logger='current',
                level=logging.DEBUG)
            break

        torch.cuda.empty_cache()

    return fps
