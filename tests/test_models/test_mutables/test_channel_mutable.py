# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import pytest
import torch

from mmrazor.models import OneShotMutableChannel


class TestChannelMutables(TestCase):

    def test_mutable_channel_ratio(self):
        with pytest.raises(AssertionError):
            # Test invalid `candidate_mode`
            OneShotMutableChannel(
                num_channels=8,
                candidate_choices=[1 / 4, 2 / 4, 3 / 4, 1.0],
                candidate_mode='xxx')

        with pytest.raises(AssertionError):
            # The candidate ratio should be in range(0, 1].
            OneShotMutableChannel(
                num_channels=8,
                candidate_choices=[0., 1 / 4, 2 / 4, 3 / 4, 1.0],
                candidate_mode='ratio')

        with pytest.raises(AssertionError):
            # Minimum number of channels should be a positive integer.
            out_mutable = OneShotMutableChannel(
                num_channels=8,
                candidate_choices=[0.01, 1 / 4, 2 / 4, 3 / 4, 1.0],
                candidate_mode='ratio')
            out_mutable.bind_mutable_name('op')
            _ = out_mutable.min_choice

        # Test mutable out
        out_mutable = OneShotMutableChannel(
            num_channels=8,
            candidate_choices=[1 / 4, 2 / 4, 3 / 4, 1.0],
            candidate_mode='ratio')

        random_choice = out_mutable.sample_choice()
        assert random_choice in [2, 4, 6, 8]

        max_choice = out_mutable.max_choice
        assert max_choice == 8
        out_mutable.current_choice = max_choice
        assert torch.equal(out_mutable.current_mask,
                           torch.ones_like(out_mutable.current_mask).bool())

        min_choice = out_mutable.min_choice
        assert min_choice == 2
        out_mutable.current_choice = min_choice
        min_mask = torch.zeros_like(out_mutable.current_mask).bool()
        min_mask[:2] = True
        assert torch.equal(out_mutable.current_mask, min_mask)

        # Test mutable in with concat_mutable
        in_mutable = OneShotMutableChannel(
            num_channels=16,
            candidate_choices=[1 / 4, 2 / 4, 3 / 4, 1.0],
            candidate_mode='ratio')
        out_mutable1 = copy.deepcopy(out_mutable)
        out_mutable2 = copy.deepcopy(out_mutable)
        in_mutable.register_same_mutable([out_mutable1, out_mutable2])
        choice1 = out_mutable1.sample_choice()
        out_mutable1.current_choice = choice1
        choice2 = out_mutable2.sample_choice()
        out_mutable2.current_choice = choice2
        assert torch.equal(
            in_mutable.current_mask,
            torch.cat([out_mutable1.current_mask, out_mutable2.current_mask]))

        with pytest.raises(AssertionError):
            # The mask of this in_mutable depends on the out mask of its
            # `concat_mutables`, so the `sample_choice` method should not
            # be called
            in_mutable.sample_choice()

        with pytest.raises(AssertionError):
            # The mask of this in_mutable depends on the out mask of its
            # `concat_mutables`, so the `min_choice` property should not
            # be called
            _ = in_mutable.min_choice

    def test_mutable_channel_number(self):
        with pytest.raises(AssertionError):
            # The candidate ratio should be in range(0, `num_channels`].
            OneShotMutableChannel(
                num_channels=8,
                candidate_choices=[0, 2, 4, 6, 8],
                candidate_mode='number')

        with pytest.raises(AssertionError):
            # Type of `candidate_choices` should be int.
            OneShotMutableChannel(
                num_channels=8,
                candidate_choices=[0., 2, 4, 6, 8],
                candidate_mode='number')

        # Test mutable out
        out_mutable = OneShotMutableChannel(
            num_channels=8,
            candidate_choices=[2, 4, 6, 8],
            candidate_mode='number')

        random_choice = out_mutable.sample_choice()
        assert random_choice in [2, 4, 6, 8]

        max_choice = out_mutable.max_choice
        assert max_choice == 8
        out_mutable.current_choice = max_choice
        assert torch.equal(out_mutable.current_mask,
                           torch.ones_like(out_mutable.current_mask).bool())

        min_choice = out_mutable.min_choice
        assert min_choice == 2
        out_mutable.current_choice = min_choice
        min_mask = torch.zeros_like(out_mutable.current_mask).bool()
        min_mask[:2] = True
        assert torch.equal(out_mutable.current_mask, min_mask)
