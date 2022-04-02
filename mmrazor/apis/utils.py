# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set ``torch.backends.cudnn.deterministic``
            to True and ``torch.backends.cudnn.benchmark`` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def auto_scale_lr(cfg, distributed, logger):
    """Automatically scaling LR according to GPU number and sample per GPU.

    Args:
        cfg (config): Training config.
        distributed (bool): Using distributed or not.
        logger (logging.Logger): Logger.
    """
    warning_msg = 'in your configuration file. Please update all the ' \
                  'configuration files to mmdet >= 2.24.0. ' \
                  'Disable automatic scaling of learning rate.'

    # default config of auto scale lr
    if 'auto_scale_lr_config' not in cfg:
        logger.warning(f'Can not find "auto_scale_lr_config" {warning_msg}')
        return

    # Get flag from config
    auto_scale_lr_flag = cfg.auto_scale_lr_config.get('auto_scale_lr', False)
    if auto_scale_lr_flag is False:
        logger.info('Automatic scaling of learning rate (LR)'
                    ' has been disabled.')
        return

    # Get default batch size from config
    default_batch_size = cfg.auto_scale_lr_config.get('default_batch_size', 0)
    if default_batch_size == 0:
        logger.warning('Can not find "default_batch_size" ' f'{warning_msg}')
        return

    # Get default initial LR from config
    default_initial_lr = cfg.auto_scale_lr_config.get('default_initial_lr', 0)
    if default_initial_lr == 0:
        logger.warning('Can not find "default_initial_lr" ' f'{warning_msg}')
        return

    # Get gpu number
    if distributed:
        _, world_size = get_dist_info()
        num_gpus = range(world_size)
    else:
        num_gpus = len(cfg.gpu_ids)

    # calculate the batch size
    batch_size = num_gpus * cfg.data.samples_per_gpu

    logger.info(f'You are using {num_gpus} GPU(s) '
                f'and {cfg.data.samples_per_gpu} samples per GPU. '
                f'Total batch size is {batch_size}.')

    if batch_size != default_batch_size:

        if cfg.optimizer.lr != default_initial_lr:
            logger.warning(
                'It seems that you changed "cfg.optimizer.lr" to '
                f'{cfg.optimizer.lr} which is not the default initial lr '
                f'({default_initial_lr}) from the config file. The '
                'automatically scaling LR will use the "cfg.optimizer.lr" to'
                ' calculate the new LR. This may not lead to a best result of'
                ' the training. If you know what are you doing, ignore this '
                'warning message.')

        # scale LR with
        # [linear scaling rule](https://arxiv.org/abs/1706.02677)
        scaled_lr = (batch_size / default_batch_size) * cfg.optimizer.lr
        logger.info('LR has been automatically scaled '
                    f'from {cfg.optimizer.lr} to {scaled_lr}')

        cfg.optimizer.lr = scaled_lr

    else:
        logger.info('The batch size match the '
                    f'default batch size: {default_batch_size}, '
                    f'will not scaling the LR ({cfg.optimizer.lr}).')
