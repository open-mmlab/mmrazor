# Copyright (c) OpenMMLab. All rights reserved.
import sys

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

import inspect
import logging
from itertools import product
from typing import Any, Callable, Dict, Iterable, Optional, Set, Union

import torch
from mmengine.logging import print_log
from torch import Tensor

from mmrazor.utils.typing import DumpChosen
from ..utils import make_divisible
from .base_mutable import BaseMutable


class MutableProtocol(Protocol):  # pragma: no cover
    """Protocol for Mutable."""

    @property
    def current_choice(self) -> Any:
        """Current choice."""

    def derive_expand_mutable(self, expand_ratio: int) -> Any:
        """Derive expand mutable."""

    def derive_divide_mutable(self, ratio: int, divisor: int) -> Any:
        """Derive divide mutable."""


class MutableChannelProtocol(MutableProtocol):  # pragma: no cover
    """Protocol for MutableChannel."""

    @property
    def current_mask(self) -> Tensor:
        """Current mask."""


def _expand_choice_fn(mutable: MutableProtocol,
                      expand_ratio: Union[int, float]) -> Callable:
    """Helper function to build `choice_fn` for expand derived mutable."""

    def fn():
        return int(mutable.current_choice * expand_ratio)

    return fn


def _expand_mask_fn(
        mutable: MutableProtocol,
        expand_ratio: Union[int, float]) -> Callable:  # pragma: no cover
    """Helper function to build `mask_fn` for expand derived mutable."""
    if not hasattr(mutable, 'current_mask'):
        raise ValueError('mutable must have attribute `currnet_mask`')

    def fn():
        mask = mutable.current_mask
        if isinstance(expand_ratio, int):
            expand_num_channels = mask.size(0) * expand_ratio
            expand_choice = mutable.current_choice * expand_ratio
        elif isinstance(expand_ratio, float):
            expand_num_channels = int(mask.size(0) * expand_ratio)
            expand_choice = int(mutable.current_choice * expand_ratio)
        else:
            raise NotImplementedError(
                f'Not support type of expand_ratio: {type(expand_ratio)}')
        expand_mask = torch.zeros(expand_num_channels).bool()
        expand_mask[:expand_choice] = True

        return expand_mask

    return fn


def _divide_and_divise(x: int, ratio: int, divisor: int = 8) -> int:
    """Helper function for divide and divise."""
    new_x = x // ratio

    return make_divisible(new_x, divisor)


def _divide_choice_fn(mutable: MutableProtocol,
                      ratio: int,
                      divisor: int = 8) -> Callable:
    """Helper function to build `choice_fn` for divide derived mutable."""

    def fn():
        return _divide_and_divise(mutable.current_choice, ratio, divisor)

    return fn


def _divide_mask_fn(mutable: MutableProtocol,
                    ratio: int,
                    divisor: int = 8) -> Callable:  # pragma: no cover
    """Helper function to build `mask_fn` for divide derived mutable."""
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


def _concat_choice_fn(mutables: Iterable[MutableChannelProtocol]) -> Callable:
    """Helper function to build `choice_fn` for concat derived mutable."""

    def fn():
        return sum((m.current_choice for m in mutables))

    return fn


def _concat_mask_fn(mutables: Iterable[MutableChannelProtocol]) -> Callable:
    """Helper function to build `mask_fn` for concat derived mutable."""

    def fn():
        return torch.cat([m.current_mask for m in mutables])

    return fn


class DerivedMethodMixin:
    """A mixin that provides some useful method to derive mutable."""

    def derive_same_mutable(self: MutableProtocol) -> 'DerivedMutable':
        """Derive same mutable as the source."""
        return self.derive_expand_mutable(expand_ratio=1)

    def derive_expand_mutable(
            self: MutableProtocol,
            expand_ratio: Union[int, BaseMutable, float]) -> 'DerivedMutable':
        """Derive expand mutable, usually used with `expand_ratio`."""
        # avoid circular import
        if isinstance(expand_ratio, int):
            choice_fn = _expand_choice_fn(self, expand_ratio=expand_ratio)
        elif isinstance(expand_ratio, float):
            choice_fn = _expand_choice_fn(self, expand_ratio=expand_ratio)
        elif isinstance(expand_ratio, BaseMutable):
            current_ratio = expand_ratio.current_choice
            choice_fn = _expand_choice_fn(self, expand_ratio=current_ratio)
        else:
            raise NotImplementedError(
                f'Not support type of ratio: {type(expand_ratio)}')

        mask_fn: Optional[Callable] = None
        if hasattr(self, 'current_mask'):
            if isinstance(expand_ratio, int):
                mask_fn = _expand_mask_fn(self, expand_ratio=expand_ratio)
            elif isinstance(expand_ratio, float):
                mask_fn = _expand_mask_fn(self, expand_ratio=expand_ratio)
            elif isinstance(expand_ratio, BaseMutable):
                mask_fn = _expand_mask_fn(self, expand_ratio=current_ratio)
            else:
                raise NotImplementedError(
                    f'Not support type of ratio: {type(expand_ratio)}')

        return DerivedMutable(choice_fn=choice_fn, mask_fn=mask_fn)

    def derive_divide_mutable(self: MutableProtocol,
                              ratio: Union[int, float, BaseMutable],
                              divisor: int = 8) -> 'DerivedMutable':
        """Derive divide mutable, usually used with `make_divisable`."""
        from .mutable_channel import BaseMutableChannel

        # avoid circular import
        if isinstance(ratio, int):
            choice_fn = _divide_choice_fn(self, ratio=ratio, divisor=divisor)
            current_ratio = ratio
        elif isinstance(ratio, float):
            current_ratio = int(ratio)
            choice_fn = _divide_choice_fn(self, ratio=current_ratio, divisor=1)
        elif isinstance(ratio, BaseMutable):
            current_ratio = int(ratio.current_choice)
            choice_fn = _divide_choice_fn(self, ratio=current_ratio, divisor=1)
        else:
            raise NotImplementedError(
                f'Not support type of ratio: {type(ratio)}')

        mask_fn: Optional[Callable] = None
        if isinstance(self, BaseMutableChannel) and hasattr(
                self, 'current_mask'):
            mask_fn = _divide_mask_fn(
                self, ratio=current_ratio, divisor=divisor)
        elif getattr(self, 'mask_fn', None):  # OneShotMutableChannel
            mask_fn = _divide_mask_fn(
                self, ratio=current_ratio, divisor=divisor)

        return DerivedMutable(choice_fn=choice_fn, mask_fn=mask_fn)

    @staticmethod
    def derive_concat_mutable(
            mutables: Iterable[MutableChannelProtocol]) -> 'DerivedMutable':
        """Derive concat mutable, usually used with `torch.cat`."""
        for mutable in mutables:
            if not hasattr(mutable, 'current_mask'):
                raise RuntimeError('Source mutable of concat derived mutable '
                                   'must have attribute `currnet_mask`')

        choice_fn = _concat_choice_fn(mutables)
        mask_fn = _concat_mask_fn(mutables)

        return DerivedMutable(choice_fn=choice_fn, mask_fn=mask_fn)


class DerivedMutable(BaseMutable, DerivedMethodMixin):
    """Class for derived mutable.

    A derived mutable is a mutable derived from other mutables that has
    `current_choice` and `current_mask` attributes (if any).

    Note:
        A derived mutable does not have its own search space, so it is
        not legal to modify its `current_choice` or `current_mask` directly.
        And the only way to modify them is by modifying `current_choice` or
        `current_mask` in corresponding source mutables.

    Args:
        choice_fn (callable): A closure that controls how to generate
            `current_choice`.
        mask_fn (callable, optional): A closure that controls how to generate
            `current_mask`. Defaults to None.
        source_mutables (iterable, optional): Specify source mutables for this
            derived mutable. If the argument is None, source mutables will be
            traced automatically by parsing mutables in closure variables.
            Defaults to None.
        alias (str, optional): alias of the `MUTABLE`. Defaults to None.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`. Defaults to None.

    Examples:
        >>> from mmrazor.models.mutables import SquentialMutableChannel
        >>> mutable_channel = SquentialMutableChannel(num_channels=3)
        >>> # derive expand mutable
        >>> derived_mutable_channel = mutable_channel * 2
        >>> # source mutables will be traced automatically
        >>> derived_mutable_channel.source_mutables
        {SquentialMutableChannel(name=unbind, num_channels=3, current_choice=3)}  # noqa: E501
        >>> # modify `current_choice` of `mutable_channel`
        >>> mutable_channel.current_choice = 2
        >>> # `current_choice` and `current_mask` of derived mutable will be modified automatically  # noqa: E501
        >>> derived_mutable_channel
        DerivedMutable(current_choice=4, activated_channels=4, source_mutables={SquentialMutableChannel(name=unbind, num_channels=3, current_choice=2)}, is_fixed=False)  # noqa: E501
    """

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
            source_mutables = self._trace_source_mutables()
            if len(source_mutables) == 0:
                raise RuntimeError(
                    'Can not find source mutables automatically, '
                    'please provide manually.')
        else:
            source_mutables = set(source_mutables)
        for mutable in source_mutables:
            if not self.is_source_mutable(mutable):
                raise ValueError('Expect all mutable to be source mutable, '
                                 f'but {mutable} is not')
        self.source_mutables = source_mutables

    # TODO
    # has no effect
    def fix_chosen(self, chosen) -> None:
        """Fix mutable with subnet config.

        Warning:
            Fix derived mutable will have no actually effect.
        """
        print_log(
            'Trying to fix chosen for derived mutable, '
            'which will have no effect.',
            level=logging.WARNING)

    def dump_chosen(self) -> DumpChosen:
        """Dump information of chosen.

        Returns:
            Dict: Dumped information.
        """
        print_log(
            'Trying to dump chosen for derived mutable, '
            'but its value depend on the source mutables.',
            level=logging.WARNING)
        return DumpChosen(chosen=self.export_chosen(), meta=None)

    def export_chosen(self):
        return self.current_choice

    @property
    def is_fixed(self) -> bool:
        """Whether the derived mutable is fixed.

        Note:
            Depends on whether all source mutables are already fixed.
        """
        return all(m.is_fixed for m in self.source_mutables)

    @is_fixed.setter
    def is_fixed(self, is_fixed: bool) -> bool:
        """Setter of is fixed."""
        raise RuntimeError(
            '`is_fixed` of derived mutable should not be modified directly')

    @property
    def choices(self):
        origin_choices = [m.current_choice for m in self.source_mutables]

        all_choices = [m.choices for m in self.source_mutables]

        product_choices = product(*all_choices)

        derived_choices = list()
        for item_choices in product_choices:
            for m, choice in zip(self.source_mutables, item_choices):
                m.current_choice = choice

            derived_choices.append(self.choice_fn())

        for m, choice in zip(self.source_mutables, origin_choices):
            m.current_choice = choice

        return derived_choices

    @property
    def num_choices(self) -> int:
        """Number of all choices.

        Note:
            Since derive mutable does not have its own search space, the number
            of choices will always be `1`.

        Returns:
            int: Number of choices.
        """
        return 1

    @property
    def current_choice(self):
        """Current choice of derived mutable."""
        return self.choice_fn()

    @current_choice.setter
    def current_choice(self, choice) -> None:
        """Setter of current choice.

        Raises:
            RuntimeError: Error when `current_choice` of derived mutable
                is modified directly.
        """
        raise RuntimeError('Choice of drived mutable can not be set.')

    @property
    def current_mask(self) -> Tensor:
        """Current mask of derived mutable."""
        if self.mask_fn is None:
            raise RuntimeError(
                '`mask_fn` must be set before access `current_mask`.')
        return self.mask_fn()

    @current_mask.setter
    def current_mask(self, mask: Tensor) -> None:
        """Setter of current mask.

        Raises:
            RuntimeError: Error when `current_mask` of derived mutable
                is modified directly.
        """
        raise RuntimeError('Mask of drived mutable can not be set.')

    @staticmethod
    def _trace_source_mutables_from_closure(
            closure: Callable) -> Set[BaseMutable]:
        """Trace source mutables from closure."""
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

        noncolcal_pars = inspect.getclosurevars(closure).nonlocals
        add_mutables_dfs(noncolcal_pars.values())

        return source_mutables

    def _trace_source_mutables(self) -> Set[BaseMutable]:
        """Trace source mutables."""
        source_mutables = self._trace_source_mutables_from_closure(
            self.choice_fn)
        if self.mask_fn is not None:
            source_mutables |= self._trace_source_mutables_from_closure(
                self.mask_fn)

        return source_mutables

    @staticmethod
    def is_source_mutable(mutable: object) -> bool:
        """Judge whether an object is source mutable(not derived mutable).

        Args:
            mutable (object): An object.

        Returns:
            bool: Indicate whether the object is source mutable or not.
        """
        return isinstance(mutable, BaseMutable) and \
            not isinstance(mutable, DerivedMutable)

    # TODO
    # should be __str__? but can not provide info when debug
    def __repr__(self) -> str:  # pragma: no cover
        s = f'{self.__class__.__name__}('
        s += f'current_choice={self.current_choice}, '
        if self.mask_fn is not None:
            s += f'activated_channels={self.current_mask.sum().item()}, '
        s += f'source_mutables={self.source_mutables}, '
        s += f'is_fixed={self.is_fixed})'

        return s
