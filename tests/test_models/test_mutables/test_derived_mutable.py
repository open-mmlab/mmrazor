# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmrazor.models.mutables import (DerivedMutable, OneShotMutableChannel,
                                     OneShotMutableValue)
from mmrazor.models.mutables.base_mutable import BaseMutable


class TestDerivedMutable(TestCase):

    def test_derived_same_mutable(self) -> None:
        mc = OneShotMutableChannel(
            num_channels=3,
            candidate_choices=[1, 2, 3],
            candidate_mode='number')
        mc_derived = mc.derive_same_mutable()
        assert mc_derived.source_mutables == {mc}

        mc.current_choice = 2
        assert mc_derived.current_choice == 2
        assert torch.equal(mc_derived.current_mask,
                           torch.tensor([1, 1, 0], dtype=torch.bool))

    def test_mutable_concat_derived(self) -> None:
        mc1 = OneShotMutableChannel(
            num_channels=3, candidate_choices=[1, 3], candidate_mode='number')
        mc2 = OneShotMutableChannel(
            num_channels=4, candidate_choices=[1, 4], candidate_mode='number')
        ms = [mc1, mc2]

        mc_derived = DerivedMutable.derive_concat_mutable(ms)
        assert mc_derived.source_mutables == set(ms)

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

        mv = OneShotMutableValue(value_list=[1, 2, 3])
        ms = [mc1, mv]
        with pytest.raises(RuntimeError):
            _ = DerivedMutable.derive_concat_mutable(ms)

    def test_mutable_channel_derived(self) -> None:
        mc = OneShotMutableChannel(
            num_channels=3,
            candidate_choices=[1, 2, 3],
            candidate_mode='number')
        mc_derived = mc * 3
        assert mc_derived.source_mutables == {mc}

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

    def test_mutable_divide(self) -> None:
        mc = OneShotMutableChannel(
            num_channels=128,
            candidate_choices=[112, 120, 128],
            candidate_mode='number')
        mc_derived = mc // 8
        assert mc_derived.source_mutables == {mc}

        mc.current_choice = 128
        assert mc_derived.current_choice == 16
        assert torch.equal(mc_derived.current_mask,
                           torch.ones(16, dtype=torch.bool))
        mc.current_choice = 120
        assert mc_derived.current_choice == 16
        assert torch.equal(mc_derived.current_mask,
                           torch.ones(16, dtype=torch.bool))

        mv = OneShotMutableValue(value_list=[112, 120, 128])
        mv_derived = mv // 8
        assert mv_derived.source_mutables == {mv}

        mv.current_choice == 128
        assert mv_derived.current_choice == 16
        mv.current_choice == 120
        assert mv_derived.current_choice == 16

    def test_double_fixed(self) -> None:
        choice_fn = lambda x: x  # noqa: E731
        derived_mutable = DerivedMutable(choice_fn)
        derived_mutable.fix_chosen({})

        with pytest.raises(RuntimeError):
            derived_mutable.fix_chosen({})

    def test_source_mutables(self) -> None:
        mc1 = OneShotMutableChannel(
            num_channels=3, candidate_choices=[1, 3], candidate_mode='number')
        mc2 = OneShotMutableChannel(
            num_channels=4, candidate_choices=[1, 4], candidate_mode='number')
        ms = [mc1, mc2]

        mc_derived1 = DerivedMutable.derive_concat_mutable(ms)

        from mmrazor.models.mutables.derived_mutable import (_concat_choice_fn,
                                                             _concat_mask_fn)
        mc_derived2 = DerivedMutable(
            choice_fn=_concat_choice_fn(ms),
            mask_fn=_concat_mask_fn(ms),
            source_mutables=ms)
        assert mc_derived1.source_mutables == mc_derived2.source_mutables

        dd_mutable = mc_derived1.derive_same_mutable()
        assert dd_mutable.source_mutables == mc_derived1.source_mutables

        with pytest.raises(ValueError):
            _ = DerivedMutable(
                choice_fn=lambda x: x, source_mutables=[mc_derived1])

        def dict_closure_fn(x, y):

            def fn():
                nonlocal x, y

            return fn

        ddd_mutable = DerivedMutable(
            choice_fn=dict_closure_fn({
                mc1: [2, 3],
                mc2: 2
            }, None),
            mask_fn=dict_closure_fn({2: [mc1, mc2]}, {3: dd_mutable}))
        assert ddd_mutable.source_mutables == mc_derived1.source_mutables

        mc3 = OneShotMutableChannel(
            num_channels=4, candidate_choices=[2, 4], candidate_mode='number')
        dddd_mutable = DerivedMutable(
            choice_fn=dict_closure_fn({
                mc1: [2, 3],
                mc2: 2
            }, []),
            mask_fn=dict_closure_fn({2: [mc1, mc2, mc3]}, {3: dd_mutable}))
        assert dddd_mutable.source_mutables == {mc1, mc2, mc3}


@pytest.mark.parametrize('expand_ratio', [1, 2, 3])
def test_derived_expand_mutable(expand_ratio: int) -> None:
    mv = OneShotMutableValue(value_list=[3, 5, 7])

    mv_derived = mv * expand_ratio
    assert mv_derived.source_mutables == {mv}

    assert isinstance(mv_derived, BaseMutable)
    assert isinstance(mv_derived, DerivedMutable)
    assert not mv_derived.is_fixed
    assert mv_derived.num_choices == 1

    mv.current_choice = mv.max_choice
    assert mv_derived.current_choice == mv.current_choice * expand_ratio
    mv.current_choice = mv.min_choice
    assert mv_derived.current_choice == mv.current_choice * expand_ratio

    with pytest.raises(RuntimeError):
        mv_derived.current_choice = 123
    with pytest.raises(RuntimeError):
        _ = mv_derived.current_mask

    chosen = mv_derived.dump_chosen()
    assert chosen == {'current_choice': mv.current_choice * expand_ratio}
    mv_derived.fix_chosen(chosen)
    assert mv_derived.is_fixed

    mv.current_choice = 5
    assert mv_derived.current_choice == 5 * expand_ratio
