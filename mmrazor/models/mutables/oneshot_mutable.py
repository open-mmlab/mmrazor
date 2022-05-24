# Copyright (c) OpenMMLab. All rights reserved.
import random
from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from mmcv.runner import ModuleDict
from torch import Tensor

from mmrazor.registry import MODELS
from .base_mutable import CHOICE_TYPE, BaseMutable


class OneShotMutable(BaseMutable[CHOICE_TYPE]):
    """Base class for one shot mutables.

    Args:
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.

    Note:
        :meth:`forward_all` is called when calculating FLOPs.
    """

    def __init__(self,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(module_kwargs=module_kwargs, init_cfg=init_cfg)

    def forward(self, x: Any, choice: Optional[CHOICE_TYPE] = None) -> Any:
        """Calls either :func:`forward_fixed` or :func:`forward_choice`
        depending on whether :func:`is_fixed` is ``True``.

        Note:
            :meth:`forward_fixed` is called when in `fixed` mode.
            :meth:`forward_choice` is called when in `unfixed` mode.

        Args:
            x (Any): input data for forward computation.
            choice (CHOICE_TYPE, optional): the chosen key in ``MUTABLE``.

        Returns:
            Any: the result of forward
        """
        if self.is_fixed:
            return self.forward_fixed(x)
        else:
            return self.forward_choice(x, choice=choice)

    @property
    def random_choice(self) -> CHOICE_TYPE:
        """Sample random choice during searching.

        Returns:
            CHOICE_TYPE: the chosen key in ``MUTABLE``.
        """

    @abstractmethod
    def forward_fixed(self, x: Any) -> Any:
        """Forward when the mutable is fixed.

        All subclasses must implement this method.
        """

    @abstractmethod
    def forward_all(self, x: Any) -> Any:
        """Forward all choices."""

    @abstractmethod
    def forward_choice(self,
                       x: Any,
                       choice: Optional[CHOICE_TYPE] = None) -> Any:
        """Forward when the mutable is not fixed.

        All subclasses must implement this method.
        """

    def set_forward_args(self, choice: CHOICE_TYPE) -> None:
        """Interface for modifying the choice using partial."""
        forward_with_default_args: Callable[[Any, Optional[CHOICE_TYPE]], Any] = \
            partial(self.forward, choice=choice)  # noqa:E501
        setattr(self, 'forward', forward_with_default_args)


@MODELS.register_module()
class OneShotOP(OneShotMutable[str]):
    """A type of ``MUTABLES`` for single path supernet, such as Single Path One
    Shot. In single path supernet, each choice block only has one choice
    invoked at the same time. A path is obtained by sampling all the choice
    blocks.

    Args:
        candidate_ops (dict[str, dict]): the configs for the candidate
            operations.
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.

    Examples:
        >>> import mmrazor.models
        >>> from mmrazor.registry import MODELS
        >>> import torch
        >>> norm_cfg = dict(type='BN', requires_grad=True)
        >>> op_cfg = dict(
        ...     type='OneShotOP',
        ...     candidate_ops=dict(
        ...         shuffle_3x3=dict(
        ...             type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=3),
        ...         shuffle_5x5=dict(
        ...             type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=5),
        ...         shuffle_7x7=dict(
        ...             type='ShuffleBlock', norm_cfg=norm_cfg, kernel_size=7),
        ...         shuffle_xception=dict(
        ...             type='ShuffleXception',
        ...             norm_cfg=norm_cfg,
        ...         ),
        ...     ),
        ...     module_kwargs=dict(in_channels=32, out_channels=32, stride=1))
        >>> op = MODELS.build(op_cfg)

        >>> input = torch.randn(4, 32, 64, 64)
        >>> op = MODELS.build(op_cfg)

        >>> op.set_forward_args('shuffle_3x3')
        >>> unfix_output = op.forward(input)

        >>> op.choices
        ['shuffle_3x3', 'shuffle_5x5', 'shuffle_7x7', 'shuffle_xception']
        >>> op.num_choices
        4

        >>> op.fix_choice('shuffle_3x3')
        >>> fix_output = op.forward(input)
        >>> torch.all(fix_output == unfix_output)
        True

        >>> op.is_fixed
        True
        >>> op.choices
        ['shuffle_3x3']
        >>> op.num_choices
    """

    def __init__(
        self,
        candidate_ops: Dict[str, Dict],
        module_kwargs: Optional[Dict[str, Dict]] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(module_kwargs=module_kwargs, init_cfg=init_cfg)
        assert len(candidate_ops) >= 1, \
            f'Number of candidate op must greater than 1, ' \
            f'but got: {len(candidate_ops)}'

        self._is_fixed = False
        self._chosen: Optional[str] = None
        self._candidate_ops = self._build_ops(candidate_ops,
                                              self.module_kwargs)

    @staticmethod
    def _build_ops(candidate_ops: Dict[str, Dict],
                   module_kwargs: Optional[Dict[str, Dict]]) -> ModuleDict:
        """Build candidate operations based on choice configures.

        Args:
            candidate_ops (dict[str, dict]): the configs for the candidate
                operations.
            module_kwargs (dict[str, dict], optional): Module initialization
                named arguments.

        Returns:
            ModuleDict (dict[str, Any], optional):  the key of ``ops`` is
                the name of each choice in configs and the value of ``ops``
                is the corresponding candidate operation.
        """
        ops = ModuleDict()
        for name, op_cfg in candidate_ops.items():
            assert name not in ops
            if module_kwargs is not None:
                op_cfg.update(module_kwargs)
            ops[name] = MODELS.build(op_cfg)
        return ops

    def forward_fixed(self, x: Any) -> Tensor:
        """Forward when the mutable is in `fixed` mode.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.

        Returns:
            Tensor: the result of forward the fixed operation.
        """
        return self._candidate_ops[self._chosen](x)

    def forward_choice(self, x: Any, choice: Optional[str] = None) -> Tensor:
        """Forward when the mutable is in `unfixed` mode.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
            choice (str, optional): the chosen key in ``MUTABLE``.

        Returns:
            Tensor: the result of forward the ``choice`` operation.
        """
        if choice is None:
            return self.forward_all(x)
        else:
            return self._candidate_ops[choice](x)

    def forward_all(self, x: Any) -> Tensor:
        """Forward all choices. Used to calculate FLOPs.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.

        Returns:
            Tensor: the result of forward all of the ``choice`` operation.
        """
        outputs = list()
        for op in self._candidate_ops.values():
            outputs.append(op(x))
        return sum(outputs)

    def fix_choice(self, choice: str) -> None:
        """Fix mutable with subnet config. This operation would convert
        `unfixed` mode to `fixed` mode. The :attr:`is_fixed` will be set to
        True and only the selected operations can be retained.

        Args:
            choice (_type_): the chosen key in ``MUTABLE``. Defaults to None.
        """
        if self.is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not call `fix_choice` function again.')

        for c in self.choices:
            if c != choice:
                self._candidate_ops.pop(c)

        self._chosen = choice
        self.is_fixed = True

    @property
    def random_choice(self) -> str:
        """uniform sampling."""
        choice = np.random.choice(self.choices, 1)[0]
        return choice

    @property
    def choices(self) -> List[str]:
        """list: all choices. """
        return list(self._candidate_ops.keys())


@MODELS.register_module()
class OneShotProbOP(OneShotOP):
    """Sampling candidate operation according to probability.

    Args:
        candidate_ops (dict[str, dict]): the configs for the candidate
            operations.
        choice_probs (list): the probability of sampling each
            candidate operation.
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 candidate_ops: Dict[str, Dict],
                 choice_probs: list = None,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(
            candidate_ops=candidate_ops,
            module_kwargs=module_kwargs,
            init_cfg=init_cfg)
        assert choice_probs is not None
        assert sum(choice_probs) - 1 < np.finfo(np.float).eps, \
            f'Please make sure the sum of the {choice_probs} is 1.'
        self.choice_probs = choice_probs

    @property
    def random_choice(self) -> str:
        """Sampling with probabilities."""
        assert len(self.choice_probs) == len(self._candidate_ops.keys())
        choice = random.choices(
            self.choices, weights=self.choice_probs, k=1)[0]
        return choice
