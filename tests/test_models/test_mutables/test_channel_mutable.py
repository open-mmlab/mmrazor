# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import pytest
import torch

from mmrazor.models import OrderChannelMutable, RatioChannelMutable


class TestChannelMutables(TestCase):

    def test_ratio_channel_mutable(self):
        with pytest.raises(AssertionError):
            # Test invalid `mask_type`
            RatioChannelMutable(
                name='op',
                mask_type='xxx',
                num_channels=8,
                candidate_choices=[1 / 4, 2 / 4, 3 / 4, 1.0])

        with pytest.raises(AssertionError):
            # Number of candidate choices must be greater than 0
            RatioChannelMutable(
                name='op',
                mask_type='out_mask',
                num_channels=8,
                candidate_choices=list())

        with pytest.raises(AssertionError):
            # The candidate ratio should be in range(0, 1].
            RatioChannelMutable(
                name='op',
                mask_type='out_mask',
                num_channels=8,
                candidate_choices=[0., 1 / 4, 2 / 4, 3 / 4, 1.0])

        with pytest.raises(AssertionError):
            # Minimum number of channels should be a positive integer.
            out_mutable = RatioChannelMutable(
                name='op',
                mask_type='out_mask',
                num_channels=8,
                candidate_choices=[0.01, 1 / 4, 2 / 4, 3 / 4, 1.0])
            _ = out_mutable.min_choice

        with pytest.raises(AssertionError):
            # Minimum number of channels should be a positive integer.
            out_mutable = RatioChannelMutable(
                name='op',
                mask_type='out_mask',
                num_channels=8,
                candidate_choices=[0.01, 1 / 4, 2 / 4, 3 / 4, 1.0])
            out_mutable.get_choice(0)

        # Test out_mutable (mask_type == 'out_mask')
        out_mutable = RatioChannelMutable(
            name='op',
            mask_type='out_mask',
            num_channels=8,
            candidate_choices=[1 / 4, 2 / 4, 3 / 4, 1.0])

        random_choice = out_mutable.sample_choice()
        assert random_choice in [2, 4, 6, 8]

        choice = out_mutable.get_choice(0)
        assert choice == 2

        max_choice = out_mutable.max_choice
        assert max_choice == 8
        out_mutable.current_choice = max_choice
        assert torch.equal(out_mutable.mask,
                           torch.ones_like(out_mutable.mask).bool())

        min_choice = out_mutable.min_choice
        assert min_choice == 2
        out_mutable.current_choice = min_choice
        min_mask = torch.zeros_like(out_mutable.mask).bool()
        min_mask[:2] = True
        assert torch.equal(out_mutable.mask, min_mask)

        with pytest.raises(AssertionError):
            # Only mutables with mask_type == 'in_mask' (named in_mutable) can
            # add `concat_mutables`
            concat_mutables = [copy.deepcopy(out_mutable)] * 2
            out_mutable.register_same_mutable(concat_mutables)

        # Test in_mutable (mask_type == 'in_mask') with concat_mutable
        in_mutable = RatioChannelMutable(
            name='op',
            mask_type='in_mask',
            num_channels=16,
            candidate_choices=[1 / 4, 2 / 4, 3 / 4, 1.0])
        out_mutable1 = copy.deepcopy(out_mutable)
        out_mutable2 = copy.deepcopy(out_mutable)
        in_mutable.register_same_mutable([out_mutable1, out_mutable2])
        choice1 = out_mutable1.sample_choice()
        out_mutable1.current_choice = choice1
        choice2 = out_mutable2.sample_choice()
        out_mutable2.current_choice = choice2
        assert in_mutable.current_choice == choice1 + choice2
        assert torch.equal(in_mutable.mask,
                           torch.cat([out_mutable1.mask, out_mutable2.mask]))

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

        with pytest.raises(AssertionError):
            # The mask of this in_mutable depends on the out mask of its
            # `concat_mutables`, so the `get_choice` method should not
            # be called
            in_mutable.get_choice(0)

    def test_order_channel_mutable(self):
        with pytest.raises(AssertionError):
            # The candidate ratio should be in range(0, `num_channels`].
            OrderChannelMutable(
                name='op',
                mask_type='out_mask',
                num_channels=8,
                candidate_choices=[0, 2, 4, 6, 8])

        with pytest.raises(AssertionError):
            # Type of `candidate_choices` should be int.
            OrderChannelMutable(
                name='op',
                mask_type='out_mask',
                num_channels=8,
                candidate_choices=[0., 2, 4, 6, 8])

        # Test out_mutable (mask_type == 'out_mask')
        out_mutable = OrderChannelMutable(
            name='op',
            mask_type='out_mask',
            num_channels=8,
            candidate_choices=[2, 4, 6, 8])

        random_choice = out_mutable.sample_choice()
        assert random_choice in [2, 4, 6, 8]

        choice = out_mutable.get_choice(0)
        assert choice == 2

        max_choice = out_mutable.max_choice
        assert max_choice == 8
        out_mutable.current_choice = max_choice
        assert torch.equal(out_mutable.mask,
                           torch.ones_like(out_mutable.mask).bool())

        min_choice = out_mutable.min_choice
        assert min_choice == 2
        out_mutable.current_choice = min_choice
        min_mask = torch.zeros_like(out_mutable.mask).bool()
        min_mask[:2] = True
        assert torch.equal(out_mutable.mask, min_mask)
