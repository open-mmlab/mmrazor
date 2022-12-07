# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmengine import MMLogger


def set_log_level(level='debug'):
    if isinstance(level, str):
        level = level.upper()
        assert level in logging._nameToLevel
        level = logging._nameToLevel[level]
    elif isinstance(level, int):
        pass
    else:
        raise NotImplementedError()

    logger = MMLogger.get_current_instance()
    logger.handlers[0].setLevel(level)
    logger.setLevel(level)
