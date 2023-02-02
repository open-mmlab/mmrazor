# Copyright (c) OpenMMLab. All rights reserved.
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from mmrazor.registry import MODELS
from mmrazor.utils.typing import DumpChosen
from ..base_mutable import BaseMutable
from ..derived_mutable import DerivedMethodMixin, DerivedMutable

Value = Union[int, float]


@MODELS.register_module()
class MutableValue(BaseMutable, DerivedMethodMixin):
    """Base class for mutable value.

    A mutable value is actually a mutable that adds some functionality to a
    list containing objects of the same type.

    Args:
        value_list (list): List of value, each value must have the same type.
        default_value (any, optional): Default value, must be one in
            `value_list`. Default to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 value_list: List[Value],
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
        """Check whether value in `value_list` has the same type."""
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
    def mutable_prefix(self) -> str:
        """Mutable prefix."""
        return 'value'

    @property
    def choices(self) -> List[Any]:
        """List of choices."""
        return self._value_list

    def fix_chosen(self, chosen: Value) -> None:
        """Fix mutable value with subnet config.

        Args:
            chosen (dict): the information of chosen.
        """
        if self.is_fixed:
            raise RuntimeError('MutableValue can not be fixed twice')

        assert chosen in self.choices

        self.current_choice = chosen
        self.is_fixed = True

    def dump_chosen(self) -> DumpChosen:
        """Dump information of chosen.

        Returns:
            Dict[str, Any]: Dumped information.
        """
        chosen = self.export_chosen()
        meta = dict(all_choices=self.choices)
        return DumpChosen(chosen=chosen, meta=meta)

    def export_chosen(self):
        return self.current_choice

    @property
    def num_choices(self) -> int:
        """Number of all choices.

        Returns:
            int: Number of choices.
        """
        return len(self.choices)

    @property
    def current_choice(self) -> Value:
        """Current choice of mutable value."""
        return self._current_choice

    @current_choice.setter
    def current_choice(self, choice: Any) -> Any:
        """Setter of current choice."""
        if choice not in self.choices:
            raise ValueError(f'Expected choice in: {self.choices}, '
                             f'but got: {choice}')

        self._current_choice = choice

    def __rmul__(self, other) -> DerivedMutable:
        """Please refer to method :func:`__mul__`."""
        return self * other

    def __mul__(self, other: Union[int, float]) -> DerivedMutable:
        """Overload `*` operator.

        Args:
            other (int): Expand ratio.

        Returns:
            DerivedMutable: Derived expand mutable.
        """
        if isinstance(other, int):
            return self.derive_expand_mutable(other)
        elif isinstance(other, float):
            return self.derive_expand_mutable(other)
        raise TypeError(f'Unsupported type {type(other)} for mul!')

    def __floordiv__(self, other: Union[int, Tuple[int,
                                                   int]]) -> DerivedMutable:
        """Overload `//` operator.

        Args:
            other: (int, tuple): divide ratio for int or
                (divide ratio, divisor) for tuple.

        Returns:
            DerivedMutable: Derived divide mutable.
        """
        if isinstance(other, int):
            return self.derive_divide_mutable(other)
        elif isinstance(other, float):
            return self.derive_divide_mutable(int(other))
        if isinstance(other, tuple):
            assert len(other) == 2
            return self.derive_divide_mutable(*other)

        raise TypeError(f'Unsupported type {type(other)} for div!')

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f'(value_list={self._value_list}, '
        s += f'current_choice={self.current_choice})'

        return s


# TODO
# 1. use comparable for type hint
# 2. use mixin
@MODELS.register_module()
class OneShotMutableValue(MutableValue):
    """Class for one-shot mutable value.

    one-shot mutable value provides `sample_choice` method and `min_choice`,
    `max_choice` properties on the top of mutable value.

    Args:
        value_list (list): List of value, each value must have the same type.
        default_value (any, optional): Default value, must be one in
            `value_list`. Default to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

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
        """Random sampling from choices.

        Returns:
            Any: Selected choice.
        """
        return random.choice(self.choices)

    @property
    def max_choice(self) -> Any:
        """Max choice of all choices.

        Returns:
            Any: Max choice.
        """
        return self.choices[-1]

    @property
    def min_choice(self) -> Any:
        """Min choice of all choices.

        Returns:
            Any: Min choice.
        """
        return self.choices[0]

    def __mul__(self, other) -> DerivedMutable:
        """Overload `*` operator.

        Args:
            other (int, SquentialMutableChannel): Expand ratio or
                SquentialMutableChannel.

        Returns:
            DerivedMutable: Derived expand mutable.
        """
        from ..mutable_channel import SquentialMutableChannel

        if isinstance(other, SquentialMutableChannel):
            return other * self

        return super().__mul__(other)
