# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import torch

from mmrazor.registry import MODELS
from .mutable_channel import MutableChannel


@MODELS.register_module()
class SlimmableMutableChannel(MutableChannel[int, Dict[str, int]]):
    """A type of ``MUTABLES`` to train several subnet together, such as the
    retraining stage in AutoSlim.

    Notes:
        We need to set `candidate_choices` after the instantiation of a
        `SlimmableMutableChannel` by ourselves.

    Args:
        num_channels (int): The raw number of channels.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 num_channels: int,
                 candidate_chioces: List[int],
                 init_cfg: Optional[Dict] = None):
        super(SlimmableMutableChannel, self).__init__(
            num_channels=num_channels, init_cfg=init_cfg)

        self._candidate_choices = candidate_chioces
        self._check_candidate_choices()

    def _check_candidate_choices(self) -> None:
        """Check if the input `candidate_choices` is valid."""
        assert all([num > 0 and num <= self.num_channels
                    for num in self._candidate_choices]), \
            f'The candidate channel numbers should be in ' \
            f'range(0, {self.num_channels}].'
        assert all(
            [isinstance(num, int) for num in self._candidate_choices]), \
            'Type of `candidate_choices` should be int.'

    @property
    def choices(self) -> List[int]:
        """Return all subnet indexes."""
        return self._candidate_choices

    def dump_chosen(self) -> Dict:
        assert self.current_choice is not None

        return dict(
            current_choice=self.current_choice,
            origin_channels=self.num_channels)

    def fix_chosen(self, dumped_chosen: Dict) -> None:
        chosen = dumped_chosen['current_choice']
        origin_channels = dumped_chosen['origin_channels']

        assert chosen <= origin_channels

        # TODO
        # remove after remove `current_choice`
        self.current_choice = chosen

        super().fix_chosen(chosen)

    @property
    def num_choices(self) -> int:
        return len(self.choices)

    def convert_choice_to_mask(self, choice: int) -> torch.Tensor:
        """Get the mask according to the input choice."""
        # if not hasattr(self, '_candidate_choices'):
        #     # todo: we trace the supernet before set_candidate_choices.
        #     #  It's hacky
        #     num_channels = self.num_channels
        # else:
        #     num_channels = self.candidate_choices[choice]
        num_channels = choice
        mask = torch.zeros(self.num_channels).bool()
        mask[:num_channels] = True
        return mask
