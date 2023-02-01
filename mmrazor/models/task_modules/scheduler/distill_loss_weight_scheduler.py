# Copyright (c) OpenMMLab. All rights reserved.
import math
from collections import Counter
from typing import List, Optional

from mmrazor.registry import PARAM_SCHEDULERS

INF = int(1e9)


class LossWeightScheduler:

    def __init__(self, begin: int = 0, end: int = INF, by_epoch: bool = True):

        if end <= begin:
            raise ValueError('end should be larger than begin, but got'
                             ' begin={}, end={}'.format(begin, end))
        self.begin = begin
        self.end = end

        # if convert_to_iter_based:
        #     assert not by_epoch
        self.by_epoch = by_epoch
        # self.convert_to_iter_based = convert_to_iter_based

    def _get_multiplier(self, base_value, cur_value, cur_step):
        raise NotImplementedError

    def get_multiplier(self, base_value, cur_value, cur_step):
        """Compute value using chainable form of the scheduler."""
        if not self.begin <= cur_step < self.end:
            return None
        return self._get_multiplier(base_value, cur_value, cur_step)


@PARAM_SCHEDULERS.register_module()
class CosineAnnealingLossWeightScheduler(LossWeightScheduler):

    def __init__(self,
                 eta_min: Optional[float] = None,
                 begin: int = 0,
                 end: int = INF,
                 by_epoch: bool = True,
                 eta_min_ratio: Optional[float] = None):
        if eta_min is None and eta_min_ratio is None:
            eta_min = 0.
        assert (eta_min is None) ^ (eta_min_ratio is None), \
            'Either `eta_min` or `eta_min_ratio should be specified'
        self.eta_min = eta_min
        self.eta_min_ratio = eta_min_ratio
        self.T_max = end - begin
        super().__init__(begin, end, by_epoch)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(*args, begin=begin, end=end, by_epoch=by_epoch, **kwargs)

    def _get_multiplier(self, base_value, cur_value, cur_iter):

        def _get_eta_min():
            if self.eta_min_ratio is None:
                return self.eta_min
            return base_value * self.eta_min_ratio

        eta_min = _get_eta_min()
        return eta_min + 0.5 * (base_value - eta_min) * (
            1 + math.cos(math.pi * (cur_iter - self.begin) / self.T_max))


@PARAM_SCHEDULERS.register_module()
class MultiStepLossWeightScheduler(LossWeightScheduler):

    def __init__(self,
                 milestones: List[int],
                 gamma: float = 0.1,
                 begin: int = 0,
                 end: int = INF,
                 by_epoch: bool = True):
        super().__init__(begin, end, by_epoch)
        # todo: check
        milestones = [value + self.begin for value in milestones]
        self.milestones = Counter(milestones)
        self.gamma = gamma

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              milestones,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        milestones = [i * epoch_length for i in milestones]
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(
            *args,
            milestones=milestones,
            begin=begin,
            end=end,
            by_epoch=by_epoch,
            **kwargs)

    def _get_multiplier(self, base_value, cur_value, cur_iter):
        if cur_iter not in self.milestones:
            return cur_value
        return cur_value * self.gamma**self.milestones[cur_iter]


@PARAM_SCHEDULERS.register_module()
class LinearLossWeightScheduler(LossWeightScheduler):

    def __init__(self,
                 start_factor: float = 1.0 / 3,
                 end_factor: float = 1.0,
                 begin: int = 0,
                 end: int = INF,
                 by_epoch: bool = True):
        if start_factor > 1.0 or start_factor < 0:
            raise ValueError(
                'Starting multiplicative factor should between 0 and 1.')

        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                'Ending multiplicative factor should between 0 and 1.')

        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = end - begin - 1
        super().__init__(begin, end, by_epoch)

    @classmethod
    def build_iter_from_epoch(cls,
                              *args,
                              begin=0,
                              end=INF,
                              by_epoch=True,
                              epoch_length=None,
                              **kwargs):
        """Build an iter-based instance of this scheduler from an epoch-based
        config."""
        assert by_epoch, 'Only epoch-based kwargs whose `by_epoch=True` can ' \
                         'be converted to iter-based.'
        assert epoch_length is not None and epoch_length > 0, \
            f'`epoch_length` must be a positive integer, ' \
            f'but got {epoch_length}.'
        by_epoch = False
        begin = int(begin * epoch_length)
        if end != INF:
            end = int(end * epoch_length)
        return cls(*args, begin=begin, end=end, by_epoch=by_epoch, **kwargs)

    def _get_multiplier(self, base_value, cur_value, cur_iter):
        start_eta = base_value * self.start_factor
        end_eta = base_value * self.end_factor
        return start_eta + (end_eta - start_eta) * (
            cur_iter - self.begin) / self.total_iters


class LossWeightSchedulerManager:

    def __init__(self, schedulers):
        self.schedulers = schedulers
        self._base_value = 1.
        self._cur_value = 1.

    @property
    def base_value(self):
        return self._base_value

    @base_value.setter
    def base_value(self, value):
        self._base_value = value

    @property
    def cur_value(self):
        return self._cur_value

    @cur_value.setter
    def cur_value(self, value):
        self._cur_value = value
