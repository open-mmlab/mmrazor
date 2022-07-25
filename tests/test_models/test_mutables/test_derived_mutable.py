# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmrazor.models.mutables import (DerivedMutable, DerivedMutableChannel,
                                     OneShotMutableChannel,
                                     OneShotMutableValue)
from mmrazor.models.mutables.base_mutable import BaseMutable


class TestDerivedMutable(TestCase):

    def test_mutable_drived(self) -> None:
        mv = OneShotMutableValue(value_list=[3, 5, 7])

        def expand_ratio_choice_fn(refer_mutable, expand_ratio):

            def real_fn():
                return refer_mutable.current_choice * expand_ratio

            return real_fn

        mv_derived = DerivedMutable(choice_fn=expand_ratio_choice_fn(mv, 4))
        assert isinstance(mv_derived, BaseMutable)
        assert isinstance(mv_derived, DerivedMutable)
        assert not mv_derived.is_fixed
        assert mv_derived.num_choices == 1

        mv.current_choice = mv.max_choice
        assert mv_derived.current_choice == 28
        mv.current_choice = mv.min_choice
        assert mv_derived.current_choice == 12

        with pytest.raises(RuntimeError):
            mv_derived.current_choice = 123

        chosen = mv_derived.dump_chosen()
        assert chosen == {'current_choice': 12}
        mv_derived.fix_chosen(chosen)
        assert mv_derived.is_fixed

        mv.current_choice = 5
        assert mv_derived.current_choice == 20

    def test_mutable_channel_derived(self) -> None:
        mutable1 = OneShotMutableChannel(
            num_channels=3, candidate_choices=[1, 3], candidate_mode='number')
        mutable2 = OneShotMutableChannel(
            num_channels=4, candidate_choices=[1, 4], candidate_mode='number')

        def concat_choice_fn(mutables):

            def real_fn():
                return sum((x.current_choice for x in mutables))

            return real_fn

        def concat_mask_fn(mutables):

            def real_fn():
                return torch.cat([x.current_mask for x in mutables])

            return real_fn

        mutables = [mutable1, mutable2]
        mv_channel_derived = DerivedMutableChannel(
            choice_fn=concat_choice_fn(mutables),
            mask_fn=concat_mask_fn(mutables))

        mutable1.current_choice = 1
        mutable2.current_choice = 4
        assert mv_channel_derived.current_choice == 5
        assert torch.equal(
            mv_channel_derived.current_mask,
            torch.tensor([1, 0, 0, 1, 1, 1, 1], dtype=torch.bool))

        mutable1.current_choice = 1
        mutable2.current_choice = 1
        assert mv_channel_derived.current_choice == 2
        assert torch.equal(
            mv_channel_derived.current_mask,
            torch.tensor([1, 0, 0, 1, 0, 0, 0], dtype=torch.bool))
