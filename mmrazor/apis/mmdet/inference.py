# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Dict, Optional, Union

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from torch import nn

from mmrazor.models import build_algorithm


def init_mmdet_model(config: Union[str, mmcv.Config],
                     checkpoint: Optional[str] = None,
                     device: str = 'cuda:0',
                     cfg_options: Optional[Dict] = None) -> nn.Module:
    """Initialize a mmdet model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)

    model_cfg = config.algorithm.architecture.model
    if 'pretrained' in model_cfg:
        model_cfg.pretrained = None
    elif 'init_cfg' in model_cfg.backbone:
        model_cfg.backbone.init_cfg = None

    config.model.train_cfg = None
    algorithm = build_algorithm(config.algorithm)
    model = algorithm.architecture.model

    if checkpoint is not None:
        checkpoint = load_checkpoint(algorithm, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()

    return model
