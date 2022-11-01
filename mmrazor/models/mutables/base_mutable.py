# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, Optional

from mmengine.model import BaseModule

from mmrazor.utils.typing import DumpChosen


class BaseMutable(BaseModule, ABC):
    """Base Class for mutables. Mutable means a searchable module widely used
    in Neural Architecture Search(NAS).

    It mainly consists of some optional operations, and achieving
    searchable function by handling choice with ``MUTATOR``.

    All subclass should implement the following APIs:

    - ``fix_chosen()``
    - ``dump_chosen()``
    - ``current_choice.setter()``
    - ``current_choice.getter()``

    Args:
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.alias = alias
        self._is_fixed = False

    @property  # type: ignore
    @abstractmethod
    def current_choice(self):
        """Current choice will affect :meth:`forward` and will be used in
        :func:`mmrazor.core.subnet.utils.export_fix_subnet` or mutator.
        """

    @current_choice.setter  # type: ignore
    @abstractmethod
    def current_choice(self, choice) -> None:
        """Current choice setter will be executed in mutator."""

    @property
    def is_fixed(self) -> bool:
        """bool: whether the mutable is fixed.

        Note:
            If a mutable is fixed, it is no longer a searchable module, just
                a normal fixed module.
            If a mutable is not fixed, it still is a searchable module.
        """
        return self._is_fixed

    @is_fixed.setter
    def is_fixed(self, is_fixed: bool) -> None:
        """Set the status of `is_fixed`."""
        assert isinstance(is_fixed, bool), \
            f'The type of `is_fixed` need to be bool type, ' \
            f'but got: {type(is_fixed)}'
        if self._is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not set `is_fixed` function repeatedly.')
        self._is_fixed = is_fixed

    @abstractmethod
    def fix_chosen(self, chosen) -> None:
        """Fix mutable with chosen. This function would fix the chosen of
        mutable. The :attr:`is_fixed` will be set to True and only the selected
        operations can be retained. All subclasses must implement this method.

        Note:
            This operation is irreversible.
        """
        raise NotImplementedError()

    @abstractmethod
    def dump_chosen(self) -> DumpChosen:
        """Save the current state of the mutable as a dictionary.

        ``DumpChosen`` has ``chosen`` and ``meta`` fields. ``chosen`` is
        necessary, ``fix_chosen`` will use the ``chosen`` . ``meta`` is used to
        store some non-essential information.
        """
        raise NotImplementedError()
