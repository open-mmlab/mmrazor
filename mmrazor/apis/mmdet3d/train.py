# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch import distributed as dist

from mmdet.apis import train_detector
from mmseg.apis import train_segmentor


def set_random_seed(seed, deterministic=False):
    """Import `set_random_seed` function here was deprecated in v0.3 and will
    be removed in v0.5.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set ``torch.backends.cudnn.deterministic``
            to True and ``torch.backends.cudnn.benchmark`` to False.
            Default: False.
    """
    warnings.warn(
        'Deprecated in v0.3 and will be removed in v0.5, '
        'please import `set_random_seed` directly from `mmrazor.apis`',
        category=DeprecationWarning)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_mmdet3d_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
