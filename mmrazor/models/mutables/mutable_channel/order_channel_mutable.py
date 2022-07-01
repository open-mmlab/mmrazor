# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

from mmrazor.registry import MODELS
from .one_shot_channel_mutable import OneShotChannelMutable


@MODELS.register_module()
class OrderChannelMutable(OneShotChannelMutable):
    """A type of ``OneShotChannelMutable``. The input candidate choices are
    candidate channel numbers.

    Args:
        name (str): Mutable name.
        mask_type (str): One of 'in_mask' or 'out_mask'.
        num_channels (int): The raw number of channels.
        candidate_choices (list | tuple): A list or tuple of candidate
            channel numbers.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 name: str,
                 mask_type: str,
                 num_channels: int,
                 candidate_choices: Union[List, Tuple],
                 init_cfg: Optional[Dict] = None):
        super(OrderChannelMutable, self).__init__(
            name, mask_type, num_channels, init_cfg=init_cfg)

        assert len(candidate_choices) > 0, \
            f'Number of candidate choices must be greater than 0, ' \
            f'but got: {len(candidate_choices)}'
        self._candidate_choices = list(candidate_choices)

        assert all([num > 0 and num <= self.num_channels
                    for num in self._candidate_choices]), \
            f'The candidate channel numbers should be in ' \
            f'range(0, {self.num_channels}].'
        assert all([isinstance(num, int)
                    for num in self._candidate_choices]),\
            'Type of `candidate_choices` should be int.'

    @property
    def choices(self) -> List[int]:
        """list: all choices. """
        return self._candidate_choices
