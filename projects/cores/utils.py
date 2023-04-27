import math

from mmengine.logging import MessageHub
from torch import distributed as torch_dist

from mmrazor.models import BaseAlgorithm
from mmrazor.models.mutators import ChannelMutator


def is_pruning_algorithm(algorithm):
    """Check whether a model is a pruning algorithm."""
    return isinstance(algorithm, BaseAlgorithm) \
             and isinstance(getattr(algorithm, 'mutator', None), ChannelMutator) # noqa


def get_model_from_runner(runner):
    """Get the model from a runner."""
    if torch_dist.is_initialized():
        return runner.model.module
    else:
        return runner.model


class RuntimeInfo():
    """A tools to get runtime info in MessageHub."""

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

    @classmethod
    def iter_pre_epoch(cls):
        iter_per_epoch = math.ceil(cls.max_iters() / cls.max_epochs())
        return iter_per_epoch
