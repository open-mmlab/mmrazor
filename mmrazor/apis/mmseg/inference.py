# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

import mmcv
from mmcv.runner import load_checkpoint
from torch import nn

from mmrazor.models import build_algorithm


def init_mmseg_model(config: Union[str, mmcv.Config],
                     checkpoint: Optional[str] = None,
                     device: str = 'cuda:0') -> nn.Module:
    """Initialize a mmseg model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))

    model_cfg = config.algorithm.architecture.model
    model_cfg.pretrained = None
    model_cfg.train_cfg = None
    algorithm = build_algorithm(config.algorithm)
    model = algorithm.architecture.model

    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model
