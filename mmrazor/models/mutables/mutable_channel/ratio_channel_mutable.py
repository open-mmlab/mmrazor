# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

from mmrazor.registry import MODELS
from .one_shot_channel_mutable import OneShotChannelMutable


@MODELS.register_module()
class RatioChannelMutable(OneShotChannelMutable):
    """A type of ``OneShotChannelMutable``. The input candidate choices are
    candidate width ratios.

    Notes:
        We first calculate the candidate channel numbers according to
    the input candidate ratios (`candidate_choices`) and regard them as
    available choices.

    Args:
        name (str): Mutable name.
        mask_type (str): One of 'in_mask' or 'out_mask'.
        num_channels (int): The raw number of channels.
        candidate_choices (list | tuple): A list or tuple of candidate width
            ratios. The width ratio is the ratio between the number of reserved
            channels and that of all channels in a layer.
            For example, if `ratios` is [0.25, 0.5], there are 2 cases
            for us to choose from when we sample from a layer with 12 channels.
            One is sampling the very first 3 channels in this layer, another is
            sampling the very first 6 channels in this layer.
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
        super(RatioChannelMutable, self).__init__(
            name, mask_type, num_channels, init_cfg=init_cfg)

        assert len(candidate_choices) > 0, \
            f'Number of candidate choices must be greater than 0, ' \
            f'but got: {len(candidate_choices)}'
        self._candidate_choices = candidate_choices

        assert all([
            ratio > 0 and ratio <= 1 for ratio in self._candidate_choices
        ]), 'The candidate ratio should be in range(0, 1].'

    @property
    def choices(self) -> List[int]:
        """list: all choices. """
        return [
            round(ratio * self.num_channels)
            for ratio in self._candidate_choices
        ]
