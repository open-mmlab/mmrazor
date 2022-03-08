# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Union

import mmcv
from mmcv.runner import load_checkpoint
from torch import nn

from mmrazor.models import build_algorithm


def init_mmcls_model(config: Union[str, mmcv.Config],
                     checkpoint: Optional[str] = None,
                     device: str = 'cuda:0',
                     cfg_options: Optional[Dict] = None) -> nn.Module:
    """Initialize a mmcls model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): cfg_options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed classifier.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model_cfg = config.algorithm.architecture.model
    model_cfg.pretrained = None
    algorithm = build_algorithm(config.algorithm)
    model = algorithm.architecture.model

    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(algorithm, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            from mmcls.datasets import ImageNet
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use imagenet by default.')
            model.CLASSES = ImageNet.CLASSES
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model
