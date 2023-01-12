# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import pytest
import torch

from mmrazor.models.mutables import (MutableValue, OneShotMutableValue,
                                     SquentialMutableChannel)


class TestMutableValue(TestCase):

    def test_init_mutable_value(self) -> None:
        value_list = [2, 4, 6]
        mv = MutableValue(value_list=value_list)
        assert mv.current_choice == 2
        assert mv.num_choices == 3

        mv = MutableValue(value_list=value_list, default_value=4)
        assert mv.current_choice == 4

        with pytest.raises(ValueError):
            mv = MutableValue(value_list=value_list, default_value=5)

        mv = MutableValue(value_list=[2])
        assert mv.current_choice == 2
        assert mv.choices == [2]

        with pytest.raises(TypeError):
            mv = MutableValue(value_list=[2, 3.2])

    def test_init_one_shot_mutable_value(self) -> None:
        value_list = [6, 4, 2]
        mv = OneShotMutableValue(value_list=value_list)
        assert mv.current_choice == 6
        assert mv.choices == [2, 4, 6]

        mv = OneShotMutableValue(value_list=value_list, default_value=4)
        assert mv.current_choice == 4

    def test_fix_chosen(self) -> None:
        mv = MutableValue([2, 3, 4])
        chosen = mv.dump_chosen()
        assert chosen.chosen == mv.current_choice
        assert chosen.meta['all_choices'] == mv.choices

        with pytest.raises(AssertionError):
            mv.fix_chosen(5)

        mv.fix_chosen(3)
        assert mv.current_choice == 3

        with pytest.raises(RuntimeError):
            mv.fix_chosen(chosen)

    def test_one_shot_mutable_value_sample(self) -> None:
        mv = OneShotMutableValue(value_list=[2, 3, 4])
        assert mv.max_choice == 4
        assert mv.min_choice == 2

        for _ in range(100):
            assert mv.sample_choice() in mv.choices

    def test_mul(self) -> None:
        mv = MutableValue(value_list=[1, 2, 3], default_value=3)
        mul_derived_mv = mv * 2
        rmul_derived_mv = 2 * mv

        assert mul_derived_mv.current_choice == 6
        assert rmul_derived_mv.current_choice == 6

        mv.current_choice = 2
        assert mul_derived_mv.current_choice == 4
        assert rmul_derived_mv.current_choice == 4

        mv = MutableValue(value_list=[1, 2, 3], default_value=3)
        mc = SquentialMutableChannel(num_channels=4)

        with pytest.raises(TypeError):
            _ = mc * mv
        with pytest.raises(TypeError):
            _ = mv * mc

        mv = OneShotMutableValue(value_list=[1, 2, 3], default_value=3)
        mc.current_choice = 2

        derived1 = mc * mv
        derived2 = mv * mc

        assert derived1.current_choice == 6
        assert derived2.current_choice == 6
        assert torch.equal(derived1.current_mask, derived2.current_mask)

        mv.current_choice = 2
        assert derived1.current_choice == 4
        assert derived2.current_choice == 4
        assert torch.equal(derived1.current_mask, derived2.current_mask)

    def test_floordiv(self) -> None:
        mv = MutableValue(value_list=[120, 128, 136])
        derived_mv = mv // 8

        mv.current_choice = 120
        assert derived_mv.current_choice == 16
        mv.current_choice = 128
        assert derived_mv.current_choice == 16

        derived_mv = mv // (8, 3)
        mv.current_choice = 120
        assert derived_mv.current_choice == 15
        mv.current_choice = 136
        assert derived_mv.current_choice == 18

    def test_repr(self) -> None:
        value_list = [2, 4, 6]
        mv = MutableValue(value_list=value_list)

        assert repr(mv) == \
            f'MutableValue(value_list={value_list}, current_choice=2)'
