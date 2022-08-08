# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Any, Dict, List, Optional

from mmrazor.registry import MODELS
from ..base_mutable import BaseMutable
from ..derived_mutable import DerivedMethodMixin, DerivedMutable


@MODELS.register_module()
class MutableValue(BaseMutable[Any, Dict], DerivedMethodMixin):

    def __init__(self,
                 value_list: List[Any],
                 default_value: Optional[Any] = None,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(alias, init_cfg)

        self._check_is_same_type(value_list)
        self._value_list = value_list

        if default_value is None:
            default_value = value_list[0]
        self.current_choice = default_value

    @staticmethod
    def _check_is_same_type(value_list: List[Any]) -> None:
        if len(value_list) == 1:
            return

        for i in range(1, len(value_list)):
            is_same_type = type(value_list[i - 1]) is \
                type(value_list[i])  # noqa: E721
            if not is_same_type:
                raise TypeError(
                    'All elements in `value_list` must have same '
                    f'type, but both types {type(value_list[i-1])} '
                    f'and type {type(value_list[i])} exist.')

    @property
    def choices(self) -> List[Any]:
        return self._value_list

    def fix_chosen(self, chosen: Dict[str, Any]) -> None:
        if self.is_fixed:
            raise RuntimeError('MutableValue can not be fixed twice')

        all_choices = chosen['all_choices']
        current_choice = chosen['current_choice']

        assert all_choices == self.choices, \
            f'Expect choices to be: {self.choices}, but got: {all_choices}'
        assert current_choice in self.choices

        self.current_choice = current_choice
        self.is_fixed = True

    def dump_chosen(self) -> Dict[str, Any]:
        return dict(
            current_choice=self.current_choice, all_choices=self.choices)

    def num_choices(self) -> int:
        return len(self.choices)

    @property
    def current_choice(self) -> Optional[Any]:
        return self._current_choice

    @current_choice.setter
    def current_choice(self, choice: Any) -> Any:
        if choice not in self.choices:
            raise ValueError(f'Expected choice in: {self.choices}, '
                             f'but got: {choice}')

        self._current_choice = choice

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f'(value_list={self._value_list}, '
        s += f'current_choice={self.current_choice})'

        return s

    def __rmul__(self, other) -> DerivedMutable:
        return self * other

    def __mul__(self, other) -> DerivedMutable:
        if isinstance(other, int):
            return self.derive_expand_mutable(other)

        raise TypeError(f'Unsupported type {type(other)} for mul!')


# TODO
# 1. use comparable for type hint
# 2. use mixin
@MODELS.register_module()
class OneShotMutableValue(MutableValue):

    def __init__(self,
                 value_list: List[Any],
                 default_value: Optional[Any] = None,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        value_list = sorted(value_list)
        # set default value as max value
        if default_value is None:
            default_value = value_list[-1]

        super().__init__(
            value_list=value_list,
            default_value=default_value,
            alias=alias,
            init_cfg=init_cfg)

    def sample_choice(self) -> Any:
        return random.choice(self.choices)

    @property
    def max_choice(self) -> Any:
        return self.choices[-1]

    @property
    def min_choice(self) -> Any:
        return self.choices[0]
