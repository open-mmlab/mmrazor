# Copyright (c) OpenMMLab. All rights reserved.
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
