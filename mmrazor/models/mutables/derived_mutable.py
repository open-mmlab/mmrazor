# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, Optional

from torch import Tensor

from .base_mutable import CHOICE_TYPE, BaseMutable


class DerivedMethodMixin:

    # HACK
    def derive_same_mutable(self):

        def same_fn(mutable: BaseMutable, attribute: str) -> Callable:

            def fn():
                return getattr(mutable, attribute)

            return fn

        if hasattr(self, 'current_mask'):
            return DerivedMutableChannel(
                choice_fn=same_fn(self, 'current_choice'),
                mask_fn=same_fn(self, 'current_mask'))
        else:
            return DerivedMutable(choice_fn=same_fn(self, 'current_choice'))


class DerivedMutable(BaseMutable[CHOICE_TYPE, Dict], DerivedMethodMixin):

    def __init__(self,
                 choice_fn: Callable,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None,
                 **kwargs) -> None:
        super().__init__(alias, init_cfg)

        self._choice_fn = choice_fn

    # TODO
    # has no effect
    def fix_chosen(self, chosen: Dict) -> None:
        self.is_fixed = True

    def dump_chosen(self) -> Dict:
        return dict(current_choice=self.current_choice)

    @property
    def num_choices(self) -> int:
        return 1

    @property
    def current_choice(self) -> CHOICE_TYPE:
        return self._choice_fn()

    @current_choice.setter
    def current_choice(self, choice: CHOICE_TYPE) -> None:
        raise RuntimeError('The choice of drived mutable can not be set!')

    # TODO
    # should be __str__? but can provide info when debug
    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f'(current_choice={self.current_choice}, '
        s += f'is_fixed={self.is_fixed})'

        return s


class DerivedMutableChannel(DerivedMutable):

    def __init__(self,
                 choice_fn: Callable,
                 mask_fn: Callable,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(choice_fn, alias, init_cfg)

        self._mask_fn = mask_fn

    @property
    def current_mask(self) -> Tensor:
        return self._mask_fn()

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f'(current_choice={self.current_choice}, '
        s += f'current_mask_shape={repr(self.current_mask.shape)}, '
        s += f'is_fixed={self.is_fixed})'

        return s
