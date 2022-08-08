# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmrazor.models.mutables import (DerivedMutable, OneShotMutableChannel,
                                     OneShotMutableValue)
from mmrazor.models.mutables.base_mutable import BaseMutable


class TestDerivedMutable(TestCase):

    def test_mutable_drived(self) -> None:
        mv = OneShotMutableValue(value_list=[3, 5, 7])

        mv_derived = mv * 4
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
        with pytest.raises(RuntimeError):
            _ = mv_derived.current_mask

        chosen = mv_derived.dump_chosen()
        assert chosen == {'current_choice': 12}
        mv_derived.fix_chosen(chosen)
        assert mv_derived.is_fixed

        mv.current_choice = 5
        assert mv_derived.current_choice == 20

    def test_mutable_concat_derived(self) -> None:
        mc1 = OneShotMutableChannel(
            num_channels=3, candidate_choices=[1, 3], candidate_mode='number')
        mc2 = OneShotMutableChannel(
            num_channels=4, candidate_choices=[1, 4], candidate_mode='number')
        ms = [mc1, mc2]

        mc_derived = DerivedMutable.derive_concat_mutable(ms)

        mc1.current_choice = 1
        mc2.current_choice = 4
        assert mc_derived.current_choice == 5
        assert torch.equal(
            mc_derived.current_mask,
            torch.tensor([1, 0, 0, 1, 1, 1, 1], dtype=torch.bool))

        mc1.current_choice = 1
        mc2.current_choice = 1
        assert mc_derived.current_choice == 2
        assert torch.equal(
            mc_derived.current_mask,
            torch.tensor([1, 0, 0, 1, 0, 0, 0], dtype=torch.bool))

    def test_mutable_channel_derived(self) -> None:
        mc = OneShotMutableChannel(
            num_channels=3,
            candidate_choices=[1, 2, 3],
            candidate_mode='number')
        mc_derived = mc * 3

        mc.current_choice = 1
        assert mc_derived.current_choice == 3
        assert torch.equal(
            mc_derived.current_mask,
            torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.bool))

        mc.current_choice = 2
        assert mc_derived.current_choice == 6
        assert torch.equal(
            mc_derived.current_mask,
            torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool))
