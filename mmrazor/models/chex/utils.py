# Copyright (c) OpenMMLab. All rights reserved.
import math

from mmengine.logging import MessageHub


class RuntimeInfo():

    @classmethod
    def get_info(cls, key):
        hub = MessageHub.get_current_instance()
        if key in hub.runtime_info:
            return hub.runtime_info[key]
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
