# Copyright (c) OpenMMLab. All rights reserved.
import random
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch.nn as nn
from torch import Tensor

from mmrazor.registry import MODELS
from ..base_mutable import CHOICE_TYPE, CHOSEN_TYPE
from .mutable_module import MutableModule


class OneShotMutableModule(MutableModule[CHOICE_TYPE, CHOSEN_TYPE]):
    """Base class for one shot mutable module. A base type of ``MUTABLES`` for
    single path supernet such as Single Path One Shot.

    All subclass should implement the following APIs:

    - ``sample_choice()``
    - ``forward_fixed()``
    - ``forward_all()``
    - ``forward_choice()``

    Args:
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.

    Note:
        :meth:`forward_all` is called when calculating FLOPs.
    """

    def forward(self, x: Any) -> Any:
        """Calls either :func:`forward_fixed` or :func:`forward_choice`
        depending on whether :func:`is_fixed` is ``True`` and whether
        :func:`current_choice` is None.

        Note:
            :meth:`forward_fixed` is called in `fixed` mode.
            :meth:`forward_all` is called in `unfixed` mode with
                :func:`current_choice` is None.
            :meth:`forward_choice` is called in `unfixed` mode with
                :func:`current_choice` is not None.

        Args:
            x (Any): input data for forward computation.
            choice (CHOICE_TYPE, optional): the chosen key in ``MUTABLE``.

        Returns:
            Any: the result of forward
        """
        if self.is_fixed:
            return self.forward_fixed(x)
        if self.current_choice is None:
            return self.forward_all(x)
        else:
            return self.forward_choice(x, choice=self.current_choice)

    @abstractmethod
    def sample_choice(self) -> CHOICE_TYPE:
        """Sample random choice.

        Returns:
            CHOICE_TYPE: the chosen key in ``MUTABLE``.
        """

    @abstractmethod
    def forward_fixed(self, x: Any) -> Any:
        """Forward with the fixed mutable.

        All subclasses must implement this method.
        """

    @abstractmethod
    def forward_all(self, x: Any) -> Any:
        """Forward all choices.

        All subclasses must implement this method.
        """

    @abstractmethod
    def forward_choice(self, x: Any, choice: CHOICE_TYPE) -> Any:
        """Forward with the unfixed mutable and current_choice is not None.

        All subclasses must implement this method.
        """


@MODELS.register_module()
class OneShotMutableOP(OneShotMutableModule[str, str]):
    """A type of ``MUTABLES`` for single path supernet, such as Single Path One
    Shot. In single path supernet, each choice block only has one choice
    invoked at the same time. A path is obtained by sampling all the choice
    blocks.

    Args:
        candidate_ops (dict[str, dict]): the configs for the candidate
            operations.
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.

    Examples:
        >>> import torch
        >>> from mmrazor.models.mutables import OneShotMutableOP

        >>> candidate_ops = nn.ModuleDict({
        ...     'conv3x3': nn.Conv2d(32, 32, 3, 1, 1),
        ...     'conv5x5': nn.Conv2d(32, 32, 5, 1, 2),
        ...     'conv7x7': nn.Conv2d(32, 32, 7, 1, 3)})

        >>> input = torch.randn(1, 32, 64, 64)
        >>> op = OneShotMutableOP(candidate_ops)

        >>> op.choices
        ['conv3x3', 'conv5x5', 'conv7x7']
        >>> op.num_choices
        3
        >>> op.is_fixed
        False

        >>> op.current_choice = 'conv3x3'
        >>> unfix_output = op.forward(input)
        >>> torch.all(unfixed_output == candidate_ops['conv3x3'](input))
        True

        >>> op.fix_chosen('conv3x3')
        >>> fix_output = op.forward(input)
        >>> torch.all(fix_output == unfix_output)
        True

        >>> op.choices
        ['conv3x3']
        >>> op.num_choices
        1
        >>> op.is_fixed
        True
    """

    def __init__(self, candidate_ops: Union[Dict[str, Dict], nn.ModuleDict],
                 **kwargs) -> None:
        super().__init__(**kwargs)
        assert len(candidate_ops) >= 1, \
            f'Number of candidate op must greater than 1, ' \
            f'but got: {len(candidate_ops)}'

        self._chosen: Optional[str] = None
        if isinstance(candidate_ops, dict):
            self._candidate_ops = self._build_ops(candidate_ops,
                                                  self.module_kwargs)
        elif isinstance(candidate_ops, nn.ModuleDict):
            self._candidate_ops = candidate_ops
        else:
            raise TypeError('candidata_ops should be a `dict` or '
                            f'`nn.ModuleDict` instance, but got '
                            f'{type(candidate_ops)}')

        assert len(self._candidate_ops) >= 1, \
            f'Number of candidate op must greater than or equal to 1, ' \
            f'but got {len(self._candidate_ops)}'

    @staticmethod
    def _build_ops(
            candidate_ops: Union[Dict[str, Dict], nn.ModuleDict],
            module_kwargs: Optional[Dict[str, Dict]] = None) -> nn.ModuleDict:
        """Build candidate operations based on choice configures.

        Args:
            candidate_ops (dict[str, dict] | :obj:`nn.ModuleDict`): the configs
                for the candidate operations or nn.ModuleDict.
            module_kwargs (dict[str, dict], optional): Module initialization
                named arguments.

        Returns:
            ModuleDict (dict[str, Any], optional):  the key of ``ops`` is
                the name of each choice in configs and the value of ``ops``
                is the corresponding candidate operation.
        """
        if isinstance(candidate_ops, nn.ModuleDict):
            return candidate_ops

        ops = nn.ModuleDict()
        for name, op_cfg in candidate_ops.items():
            assert name not in ops
            if module_kwargs is not None:
                op_cfg.update(module_kwargs)
            ops[name] = MODELS.build(op_cfg)
        return ops

    def forward_fixed(self, x: Any) -> Tensor:
        """Forward with the `fixed` mutable.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.

        Returns:
            Tensor: the result of forward the fixed operation.
        """
        return self._candidate_ops[self._chosen](x)

    def forward_choice(self, x: Any, choice: str) -> Tensor:
        """Forward with the `unfixed` mutable and current choice is not None.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
            choice (str): the chosen key in ``OneShotMutableOP``.

        Returns:
            Tensor: the result of forward the ``choice`` operation.
        """
        assert isinstance(choice, str) and choice in self.choices
        return self._candidate_ops[choice](x)

    def forward_all(self, x: Any) -> Tensor:
        """Forward all choices. Used to calculate FLOPs.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.

        Returns:
            Tensor: the result of forward all of the ``choice`` operation.
        """
        outputs = [op(x) for op in self._candidate_ops.values()]
        return sum(outputs)

    def fix_chosen(self, chosen: str) -> None:
        """Fix mutable with subnet config. This operation would convert
        `unfixed` mode to `fixed` mode. The :attr:`is_fixed` will be set to
        True and only the selected operations can be retained.

        Args:
            chosen (str): the chosen key in ``MUTABLE``. Defaults to None.
        """
        if self.is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        for c in self.choices:
            if c != chosen:
                self._candidate_ops.pop(c)

        self._chosen = chosen
        self.is_fixed = True

    def sample_choice(self) -> str:
        """uniform sampling."""
        return np.random.choice(self.choices, 1)[0]

    @property
    def choices(self) -> List[str]:
        """list: all choices. """
        return list(self._candidate_ops.keys())

    @property
    def num_choices(self):
        return len(self.choices)


@MODELS.register_module()
class OneShotProbMutableOP(OneShotMutableOP):
    """Sampling candidate operation according to probability.

    Args:
        candidate_ops (dict[str, dict]): the configs for the candidate
            operations.
        choice_probs (list): the probability of sampling each
            candidate operation.
        module_kwargs (dict[str, dict], optional): Module initialization named
            arguments. Defaults to None.
        alias (str, optional): alias of the `MUTABLE`.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self,
                 candidate_ops: Dict[str, Dict],
                 choice_probs: list = None,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(
            candidate_ops=candidate_ops,
            module_kwargs=module_kwargs,
            alias=alias,
            init_cfg=init_cfg)
        assert choice_probs is not None
        assert sum(choice_probs) - 1 < np.finfo(np.float64).eps, \
            f'Please make sure the sum of the {choice_probs} is 1.'
        self.choice_probs = choice_probs

    def sample_choice(self) -> str:
        """Sampling with probabilities."""
        assert len(self.choice_probs) == len(self._candidate_ops.keys())
        return random.choices(self.choices, weights=self.choice_probs, k=1)[0]
