# Copyright (c) OpenMMLab. All rights reserved.
import inspect
from collections.abc import Iterable
from typing import Any, Callable, Dict, Optional, Protocol, Set, Union

import torch
from mmcls.models.utils import make_divisible
from mmengine.logging import MMLogger
from torch import Tensor

from .base_mutable import CHOICE_TYPE, BaseMutable

logger = MMLogger.get_current_instance()


class MutableProtocol(Protocol):  # pragma: no cover

    @property
    def current_choice(self) -> Any:
        ...

    def derive_expand_mutable(self, expand_ratio: int) -> Any:
        ...

    def derive_divide_mutable(self, ratio: int, divisor: int) -> Any:
        ...


class ChannelMutableProtocol(MutableProtocol):  # pragma: no cover

    @property
    def current_mask(self) -> Tensor:
        ...


def _expand_choice_fn(mutable: MutableProtocol, expand_ratio: int) -> Callable:

    def fn():
        return mutable.current_choice * expand_ratio

    return fn


def _expand_mask_fn(mutable: MutableProtocol,
                    expand_ratio: int) -> Callable:  # pragma: no cover
    if not hasattr(mutable, 'current_mask'):
        raise ValueError('mutable must have attribute `currnet_mask`')

    def fn():
        mask = mutable.current_mask
        expand_num_channels = mask.size(0) * expand_ratio
        expand_choice = mutable.current_choice * expand_ratio
        expand_mask = torch.zeros(expand_num_channels).bool()
        expand_mask[:expand_choice] = True

        return expand_mask

    return fn


def _divide_and_divise(x: int, ratio: int, divisor: int = 8) -> int:
    new_x = x // ratio

    return make_divisible(new_x, divisor)


def _divide_choice_fn(mutable: MutableProtocol,
                      ratio: int,
                      divisor: int = 8) -> Callable:

    def fn():
        return _divide_and_divise(mutable.current_choice, ratio, divisor)

    return fn


def _divide_mask_fn(mutable: MutableProtocol,
                    ratio: int,
                    divisor: int = 8) -> Callable:  # pragma: no cover
    if not hasattr(mutable, 'current_mask'):
        raise ValueError('mutable must have attribute `currnet_mask`')

    def fn():
        mask = mutable.current_mask
        divide_num_channels = _divide_and_divise(mask.size(0), ratio, divisor)
        divide_choice = _divide_and_divise(mutable.current_choice, ratio,
                                           divisor)
        divide_mask = torch.zeros(divide_num_channels).bool()
        divide_mask[:divide_choice] = True

        return divide_mask

    return fn


def _concat_choice_fn(mutables: Iterable[ChannelMutableProtocol]) -> Callable:

    def fn():
        return sum((m.current_choice for m in mutables))

    return fn


def _concat_mask_fn(mutables: Iterable[ChannelMutableProtocol]) -> Callable:
    for mutable in mutables:
        if not hasattr(mutable, 'current_mask'):
            raise RuntimeError('mutable must have attribute `currnet_mask`')

    def fn():
        return torch.cat([m.current_mask for m in mutables])

    return fn


class DerivedMethodMixin:

    def derive_same_mutable(self: MutableProtocol) -> 'DerivedMutable':
        return self.derive_expand_mutable(expand_ratio=1)

    def derive_expand_mutable(self: MutableProtocol,
                              expand_ratio: int) -> 'DerivedMutable':
        choice_fn = _expand_choice_fn(self, expand_ratio=expand_ratio)

        mask_fn: Optional[Callable] = None
        if hasattr(self, 'current_mask'):
            mask_fn = _expand_mask_fn(self, expand_ratio=expand_ratio)

        return DerivedMutable(choice_fn=choice_fn, mask_fn=mask_fn)

    def derive_divide_mutable(self: MutableProtocol,
                              ratio: int,
                              divisor: int = 8) -> 'DerivedMutable':
        choice_fn = _divide_choice_fn(self, ratio=ratio, divisor=divisor)

        mask_fn: Optional[Callable] = None
        if hasattr(self, 'current_mask'):
            mask_fn = _divide_mask_fn(self, ratio=ratio, divisor=divisor)

        return DerivedMutable(choice_fn=choice_fn, mask_fn=mask_fn)

    @staticmethod
    def derive_concat_mutable(
            mutables: Iterable[ChannelMutableProtocol]) -> 'DerivedMutable':
        choice_fn = _concat_choice_fn(mutables)
        mask_fn = _concat_mask_fn(mutables)

        return DerivedMutable(choice_fn=choice_fn, mask_fn=mask_fn)


class DerivedMutable(BaseMutable[CHOICE_TYPE, Dict], DerivedMethodMixin):

    def __init__(self,
                 choice_fn: Callable,
                 mask_fn: Optional[Callable] = None,
                 source_mutables: Optional[Iterable[BaseMutable]] = None,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(alias, init_cfg)

        self.choice_fn = choice_fn
        self.mask_fn = mask_fn

        if source_mutables is None:
            source_mutables = self._find_source_mutables()
            if len(source_mutables) == 0:
                # TODO
                # warning or raise error?
                logger.warning('Can not find source mutables automatically')
        else:
            source_mutables = set(source_mutables)
        for mutable in source_mutables:
            if not self.is_source_mutable(mutable):
                raise ValueError('Expect all mutable to be source mutable, '
                                 f'but {mutable} is not')
        self.source_mutables = source_mutables

    # TODO
    # has no effect
    def fix_chosen(self, chosen: Dict) -> None:
        if self.is_fixed:
            raise RuntimeError('DerivedMutable can not be fixed twice')

        self.is_fixed = True

    def dump_chosen(self) -> Dict:
        return dict(current_choice=self.current_choice)

    @property
    def num_choices(self) -> int:
        return 1

    @property
    def current_choice(self) -> CHOICE_TYPE:
        return self.choice_fn()

    @current_choice.setter
    def current_choice(self, choice: CHOICE_TYPE) -> None:
        raise RuntimeError('Choice of drived mutable can not be set!')

    @property
    def current_mask(self) -> Tensor:
        if self.mask_fn is None:
            raise RuntimeError(
                '`mask_fn` must be set before access `current_mask`')
        return self.mask_fn()

    @staticmethod
    def _extract_source_mutables_from_fn(fn: Callable) -> Set[BaseMutable]:
        source_mutables: Set[BaseMutable] = set()

        def add_mutables_dfs(
                mutable: Union[Iterable, BaseMutable, Dict]) -> None:
            nonlocal source_mutables
            if isinstance(mutable, BaseMutable):
                if isinstance(mutable, DerivedMutable):
                    source_mutables |= mutable.source_mutables
                else:
                    source_mutables.add(mutable)
            # dict is also iterable, should parse first
            elif isinstance(mutable, dict):
                add_mutables_dfs(mutable.values())
                add_mutables_dfs(mutable.keys())
            elif isinstance(mutable, Iterable):
                for m in mutable:
                    add_mutables_dfs(m)

        noncolcal_pars = inspect.getclosurevars(fn).nonlocals
        add_mutables_dfs(noncolcal_pars.values())

        return source_mutables

    def _find_source_mutables(self) -> Set[BaseMutable]:
        source_mutables = self._extract_source_mutables_from_fn(self.choice_fn)
        if self.mask_fn is not None:
            source_mutables |= self._extract_source_mutables_from_fn(
                self.mask_fn)

        return source_mutables

    @staticmethod
    def is_source_mutable(mutable: BaseMutable) -> bool:
        return isinstance(mutable, BaseMutable) and \
            not isinstance(mutable, DerivedMutable)

    # TODO
    # should be __str__? but can not provide info when debug
    def __repr__(self) -> str:  # pragma: no cover
        s = f'{self.__class__.__name__}('
        if self.choice_fn is not None:
            s += f'current_choice={self.current_choice}, '
        if self.mask_fn is not None:
            s += f'activated_channels={self.current_mask.sum().item()}, '
        s += f'source_mutables={self.source_mutables}, '
        s += f'is_fixed={self.is_fixed})'

        return s
