# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from ..base_mutable import BaseMutable


class MutableModule(BaseMutable):
    """Base Class for mutables. Mutable means a searchable module widely used
    in Neural Architecture Search(NAS).

    It mainly consists of some optional operations, and achieving
    searchable function by handling choice with ``MUTATOR``.

    All subclass should implement the following APIs and the other
    abstract method in ``BaseMutable``:

    - ``forward()``
    - ``forward_all()``
    - ``forward_fix()``
    - ``choices()``

    Args:
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(alias, init_cfg)

        self.module_kwargs = module_kwargs
        self._current_choice = None

    @property
    def current_choice(self):
        """Current choice will affect :meth:`forward` and will be used in
        :func:`mmrazor.core.subnet.utils.export_fix_subnet` or mutator.
        """
        return self._current_choice

    @current_choice.setter
    def current_choice(self, choice) -> None:
        """Current choice setter will be executed in mutator."""
        self._current_choice = choice

    @property
    @abstractmethod
    def choices(self) -> List[str]:
        """list: all choices.  All subclasses must implement this method."""

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward computation."""

    @abstractmethod
    def forward_fixed(self, x):
        """Forward with the fixed mutable.

        All subclasses must implement this method.
        """

    @abstractmethod
    def forward_all(self, x):
        """Forward all choices.

        All subclasses must implement this method.
        """

    @property
    def num_choices(self) -> int:
        """Number of choices."""
        return len(self.choices)
