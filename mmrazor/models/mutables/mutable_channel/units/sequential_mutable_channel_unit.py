# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Dict, Union

import torch.nn as nn
from mmcv.cnn.bricks import Conv2dAdaptivePadding
from mmengine import MMLogger
from mmengine.model.utils import _BatchNormXd
from mmengine.utils.dl_utils.parrots_wrapper import \
    SyncBatchNorm as EngineSyncBatchNorm

from mmrazor.models.architectures import dynamic_ops
from mmrazor.registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from ..sequential_mutable_channel import SquentialMutableChannel
from .mutable_channel_unit import MutableChannelUnit


# TODO change the name of SequentialMutableChannelUnit
@MODELS.register_module()
class SequentialMutableChannelUnit(MutableChannelUnit):
    """SequentialMutableChannelUnit accepts a intger(number) or float(ratio) as
    the choice, which indicates how many of the channels are remained from left
    to right, like 11110000.

    Args:
        num_channels (int): number of channels.
        choice_mode (str): mode of choice, which is one of 'number' or 'ratio'.
        divisor (int): Used to make choice divisible.
        min_value (int): the minimal value used when make divisible.
        min_ratio (float): the minimal ratio used when make divisible.
    """

    def __init__(
            self,
            num_channels: int,
            choice_mode='number',
            # args for make divisible
            divisor=1,
            min_value=1,
            min_ratio=0.9) -> None:
        super().__init__(num_channels)
        assert choice_mode in ['ratio', 'number']
        self.choice_mode = choice_mode

        self.mutable_channel: SquentialMutableChannel = \
            SquentialMutableChannel(num_channels, choice_mode=choice_mode)

        # for make_divisible
        self.divisor = divisor
        self.min_value = min_value
        self.min_ratio = min_ratio

    @classmethod
    def init_from_mutable_channel(cls,
                                  mutable_channel: SquentialMutableChannel):
        unit = cls(mutable_channel.num_channels, mutable_channel.choice_mode)
        unit.mutable_channel = mutable_channel
        return unit

    def prepare_for_pruning(self, model: nn.Module):
        """Prepare for pruning, including register mutable channels."""
        # register MutableMask
        self._replace_with_dynamic_ops(
            model, {
                Conv2dAdaptivePadding:
                dynamic_ops.DynamicConv2dAdaptivePadding,
                nn.Conv2d: dynamic_ops.DynamicConv2d,
                nn.BatchNorm2d: dynamic_ops.DynamicBatchNorm2d,
                nn.Linear: dynamic_ops.DynamicLinear,
                nn.SyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                EngineSyncBatchNorm: dynamic_ops.DynamicSyncBatchNorm,
                _BatchNormXd: dynamic_ops.DynamicBatchNormXd,
            })
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    #  ~

    @property
    def is_num_mode(self):
        return self.choice_mode == 'number'

    def fix_chosen(self, choice=None):
        """fix chosen."""
        super().fix_chosen(choice)
        self.mutable_channel.fix_chosen()

    def config_template(self,
                        with_init_args=False,
                        with_channels=False) -> Dict:
        """Template of config."""
        config = super().config_template(with_init_args, with_channels)
        if with_init_args:
            init_args: Dict = config['init_args']
            init_args.update(
                dict(
                    choice_mode=self.choice_mode,
                    divisor=self.divisor,
                    min_value=self.min_value,
                    min_ratio=self.min_ratio))
        return config

    # choice

    @property
    def current_choice(self) -> Union[int, float]:
        """return current choice."""
        return self.mutable_channel.current_choice

    @current_choice.setter
    def current_choice(self, choice: Union[int, float]):
        """set choice."""
        choice_num_ = self._get_valid_int_choice(choice)
        self.mutable_channel.current_choice = choice_num_

    def sample_choice(self) -> Union[int, float]:
        """Sample a choice in (0,1]"""
        num_choice = random.randint(1, self.num_channels)
        num_choice = self._make_divisible(num_choice)
        if self.is_num_mode:
            return num_choice
        else:
            return self._num2ratio(num_choice)

    # private methods
    def _get_valid_int_choice(self, choice: Union[float, int]) -> int:
        choice_num = self._ratio2num(choice)
        choice_num_ = self._make_divisible(choice_num)
        if choice_num != choice_num_:
            self._make_divisible_info(choice, self.current_choice)
        return choice_num_

    def _make_divisible(self, choice_int: int):
        """Make the choice divisible."""
        from mmrazor.models.utils import make_divisible
        return make_divisible(choice_int, self.divisor, self.min_value,
                              self.min_ratio)

    def _num2ratio(self, choice: Union[int, float]) -> float:
        """Convert the a number choice to a ratio choice."""
        if isinstance(choice, float):
            return choice
        else:
            return choice / self.num_channels

    def _ratio2num(self, choice: Union[int, float]) -> int:
        """Convert the a ratio choice to a number choice."""
        if isinstance(choice, int):
            return choice
        else:
            return max(1, int(self.num_channels * choice))

    def _make_divisible_info(self, choice, new_choice):
        logger = MMLogger.get_current_instance()
        logger.info(f'The choice={choice}, which is set to {self.name}, '
                    f'is changed to {new_choice} for a divisible choice.')
