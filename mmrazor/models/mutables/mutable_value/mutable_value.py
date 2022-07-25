# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Any, Dict, List, Optional

from mmrazor.registry import MODELS
from ..base_mutable import BaseMutable


@MODELS.register_module()
class MutableValue(BaseMutable[Any, Dict]):

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
        all_choices = chosen['all_choices']
        current_choice = chosen['current_choice']

        assert all_choices == self.choices
        assert current_choice in self.choices

        self._value_list = [current_choice]
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
