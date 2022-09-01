# Copyright (c) OpenMMLab. All rights reserved.
import logging
import time

import torch
from mmengine.logging import print_log


def repeat_measure_inference_speed(model,
                                   resource_args,
                                   max_iter: int = 100,
                                   log_interval: int = 100,
                                   repeat_num: int = 1) -> float:
    """Repeat speed measure for multi-times to get more precise results."""
    assert repeat_num >= 1

    fps_list = []

    for _ in range(repeat_num):

        fps_list.append(
            measure_inference_speed(model, resource_args, max_iter,
                                    log_interval))

    if repeat_num > 1:
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


def measure_inference_speed(model, resource_args, max_iter: int,
                            log_interval: int) -> float:
    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0.0
    fps = 0.0
    data = dict()
    if next(model.parameters()).is_cuda:
        device = 'cuda'
    else:
        raise NotImplementedError('To use cpu to test latency not supported.')
    # benchmark with 100 image and take the average
    for i in range(1, max_iter):
        if device == 'cuda':
            data = torch.rand(resource_args['input_shape']).cuda()
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
