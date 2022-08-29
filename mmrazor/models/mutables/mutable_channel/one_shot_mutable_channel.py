# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

from mmrazor.registry import MODELS
from ..derived_mutable import DerivedMutable
from .mutable_channel import MutableChannel

CANDIDATE_CHOICE_TYPE = List[Union[float, int]]


@MODELS.register_module()
class OneShotMutableChannel(MutableChannel[int, Dict]):
    """A type of ``MUTABLES`` for single path supernet such as AutoSlim. In
    single path supernet, each module only has one choice invoked at the same
    time. A path is obtained by sampling all the available choices. It is the
    base class for one shot mutable channel.

    Args:
        num_channels (int): The raw number of channels.
        candidate_choices (List): If `candidate_mode` is "ratio",
            candidate_choices is a list of candidate width ratios. If
            `candidate_mode` is "number", candidate_choices is a list of
            candidate channel number. We note that the width ratio is the ratio
            between the number of reserved channels and that of all channels in
            a layer.
            For example, if `ratios` is [0.25, 0.5], there are 2 cases
            for us to choose from when we sample from a layer with 12 channels.
            One is sampling the very first 3 channels in this layer, another is
            sampling the very first 6 channels in this layer.
        candidate_mode (str): One of "ratio" or "number".
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 num_channels: int,
                 candidate_choices: List[Union[int, float]],
                 candidate_mode: str = 'ratio',
                 init_cfg: Optional[Dict] = None):
        super(OneShotMutableChannel, self).__init__(
            num_channels=num_channels, init_cfg=init_cfg)

        self._current_choice = num_channels

        assert (candidate_mode is None and candidate_choices is None) or (
            candidate_mode is not None and candidate_choices is not None)
        if candidate_mode is not None:
            self._check_candidate_mode(candidate_mode)
        self._candidate_mode = candidate_mode
        if candidate_choices is not None:
            self._check_candidate_choices(candidate_choices)
        self._candidate_choices = candidate_choices

    def _check_candidate_mode(self, candidate_mode: str) -> None:
        assert candidate_mode in ['ratio', 'number']

    def _check_candidate_choices(
            self, candidate_choices: CANDIDATE_CHOICE_TYPE) -> None:
        """Check if the input `candidate_choices` is valid."""
        if self._candidate_mode == 'number':
            assert all([num > 0 and num <= self.num_channels
                        for num in candidate_choices]), \
                f'The candidate channel numbers should be in ' \
                f'range(0, {self.num_channels}].'
            assert all([isinstance(num, int)
                        for num in candidate_choices]), \
                'Type of `candidate_choices` should be int.'
        else:
            assert all([
                ratio > 0 and ratio <= 1 for ratio in candidate_choices
            ]), 'The candidate ratio should be in range(0, 1].'

    def sample_choice(self) -> int:
        """Sample an arbitrary selection from candidate choices.

        Returns:
            int: The chosen number of channels.
        """
        assert len(self.concat_parent_mutables) == 0
        num_channels = np.random.choice(self.choices)
        assert num_channels > 0, \
            f'Sampled number of channels in `Mutable` {self.name}' \
            f' should be a positive integer.'
        return num_channels

    @property
    def min_choice(self) -> int:
        """Minimum number of channels."""
        assert len(self.concat_parent_mutables) == 0
        min_channels = min(self.choices)
        assert min_channels > 0, \
            f'Minimum number of channels in `Mutable` {self.name}' \
            f' should be a positive integer.'
        return min_channels

    @property
    def max_choice(self) -> int:
        """Maximum number of channels."""
        return max(self.choices)

    @property
    def current_choice(self):
        """The current choice of the mutable."""
        assert len(self.concat_parent_mutables) == 0
        return self._current_choice

    @current_choice.setter
    def current_choice(self, choice: int):
        """Set the current choice of the mutable."""
        assert choice in self.choices
        assert len(self.concat_parent_mutables) == 0

        self._current_choice = choice

    def set_candidate_choices(
            self, candidate_mode: str,
            candidate_choices: CANDIDATE_CHOICE_TYPE) -> None:
        assert self._candidate_choices is None, \
            '`candidate_choices` has already been set'
        self._check_candidate_mode(candidate_mode)
        self._candidate_mode = candidate_mode
        self._check_candidate_choices(candidate_choices)
        self._candidate_choices = candidate_choices

    # TODO
    # should return List[int], but this will make mypy complain
    @property
    def choices(self) -> List:
        """list: all choices. """
        assert self._candidate_choices is not None, \
            '`candidate_choices` must be set before access'
        if self._candidate_mode == 'number':
            return self._candidate_choices

        candidate_choices = [
            round(ratio * self.num_channels)
            for ratio in self._candidate_choices
        ]
        return candidate_choices

    @property
    def num_choices(self) -> int:
        return len(self.choices)

    def convert_choice_to_mask(self, choice: int) -> torch.Tensor:
        """Get the mask according to the input choice."""
        num_channels = choice
        mask = torch.zeros(self.num_channels).bool()
        mask[:num_channels] = True
        return mask

    def dump_chosen(self) -> Dict:
        assert self.current_choice is not None

        return dict(
            current_choice=self.current_choice,
            origin_channels=self.num_channels)

    def fix_chosen(self, dumped_chosen: Dict) -> None:
        if self.is_fixed:
            raise RuntimeError('OneShotMutableChannel can not be fixed twice')

        current_choice = dumped_chosen['current_choice']
        origin_channels = dumped_chosen['origin_channels']

        assert current_choice <= origin_channels
        assert origin_channels == self.num_channels

        self.current_choice = current_choice
        self.is_fixed = True

    def __repr__(self):
        concat_mutable_name = [
            mutable.name for mutable in self.concat_parent_mutables
        ]
        repr_str = self.__class__.__name__
        repr_str += f'(name={self.name}, '
        repr_str += f'num_channels={self.num_channels}, '
        repr_str += f'current_choice={self.current_choice}, '
        repr_str += f'choices={self.choices}, '
        repr_str += f'activated_channels={self.current_mask.sum().item()}, '
        repr_str += f'concat_mutable_name={concat_mutable_name})'
        return repr_str

    def __rmul__(self, other) -> DerivedMutable:
        return self * other

    def __mul__(self, other) -> DerivedMutable:
        if isinstance(other, int):
            return self.derive_expand_mutable(other)

        def expand_choice_fn(mutable1: 'OneShotMutableChannel',
                             mutable2: OneShotMutableValue) -> Callable:

            def fn():
                return mutable1.current_choice * mutable2.current_choice

            return fn

        def expand_mask_fn(mutable1: 'OneShotMutableChannel',
                           mutable2: OneShotMutableValue) -> Callable:

            def fn():
                mask = mutable1.current_mask
                max_expand_ratio = mutable2.max_choice
                current_expand_ratio = mutable2.current_choice
                expand_num_channels = mask.size(0) * max_expand_ratio

                expand_choice = mutable1.current_choice * current_expand_ratio
                expand_mask = torch.zeros(expand_num_channels).bool()
                expand_mask[:expand_choice] = True

                return expand_mask

            return fn

        if isinstance(other, OneShotMutableValue):
            return DerivedMutable(
                choice_fn=expand_choice_fn(self, other),
                mask_fn=expand_mask_fn(self, other))

        raise TypeError(f'Unsupported type {type(other)} for mul!')

    def __floordiv__(self, other) -> DerivedMutable:
        if isinstance(other, int):
            return self.derive_divide_mutable(other)
        if isinstance(other, tuple):
            assert len(other) == 2
            return self.derive_divide_mutable(*other)

        raise TypeError(f'Unsupported type {type(other)} for div!')
