# Copyright (c) OpenMMLab. All rights reserved.
import math

from mmengine import Config, MessageHub


class RuntimeInfo():
    """A tools to get runtime info in MessageHub."""

    @classmethod
    def info(cls):
        hub = MessageHub.get_current_instance()
        return hub.runtime_info

    @classmethod
    def get_info(cls, key):
        info = cls.info()
        if key in info:
            return info[key]
        else:
            raise KeyError(key)

    @classmethod
    def epoch(cls):
        return cls.get_info('epoch')

    @classmethod
    def max_epochs(cls):
        return cls.get_info('max_epochs')

    @classmethod
    def iter(cls):
        return cls.get_info('iter')

    @classmethod
    def max_iters(cls):
        return cls.get_info('max_iters')

    @classmethod
    def iter_by_epoch(cls):
        iter_per_epoch = math.ceil(cls.max_iters() / cls.max_epochs())
        return cls.iter() % iter_per_epoch

    @classmethod
    def iter_pre_epoch(cls):
        iter_per_epoch = math.ceil(cls.max_iters() / cls.max_epochs())
        return iter_per_epoch

    @classmethod
    def config(cls):
        cfg: str = cls.get_info('cfg')
        config = Config.fromstring(cfg, '.py')
        return config

    @classmethod
    def work_dir(cls):
        config = cls.config()
        return config['work_dir']
