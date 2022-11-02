# Copyright (c) OpenMMLab. All rights reserved.
import random
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch.nn as nn
from torch import Tensor

from mmrazor.registry import MODELS
from mmrazor.utils.typing import DumpChosen
from .mutable_module import MutableModule


class OneShotMutableModule(MutableModule):
    """Base class for one shot mutable module. A base type of ``MUTABLES`` for
    single path supernet such as Single Path One Shot.

    All subclass should implement the following APIs and the other
    abstract method in ``MutableModule``:

    - ``sample_choice()``
    - ``forward_choice()``

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
    def sample_choice(self) -> str:
        """Sample random choice.

        Returns:
            str: the chosen key in ``MUTABLE``.
        """

    @abstractmethod
    def forward_choice(self, x, choice: str):
        """Forward with the unfixed mutable and current_choice is not None.

        All subclasses must implement this method.
        """


@MODELS.register_module()
class OneShotMutableOP(OneShotMutableModule):
    """A type of ``MUTABLES`` for single path supernet, such as Single Path One
    Shot. In single path supernet, each choice block only has one choice
    invoked at the same time. A path is obtained by sampling all the choice
    blocks.

    Args:
        candidates (dict[str, dict]): the configs for the candidate
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

        >>> candidates = nn.ModuleDict({
        ...     'conv3x3': nn.Conv2d(32, 32, 3, 1, 1),
        ...     'conv5x5': nn.Conv2d(32, 32, 5, 1, 2),

        >>> input = torch.randn(1, 32, 64, 64)
        >>> op = OneShotMutableOP(candidates)

        >>> op.choices
        ['conv3x3', 'conv5x5', 'conv7x7']
        >>> op.num_choices
        3
        >>> op.is_fixed
        False

        >>> op.current_choice = 'conv3x3'
        >>> unfix_output = op.forward(input)
        >>> torch.all(unfixed_output == candidates['conv3x3'](input))
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

    def __init__(
        self,
        candidates: Union[Dict[str, Dict], nn.ModuleDict],
        module_kwargs: Optional[Dict[str, Dict]] = None,
        alias: Optional[str] = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            module_kwargs=module_kwargs, alias=alias, init_cfg=init_cfg)
        assert len(candidates) >= 1, \
            f'Number of candidate op must greater than 1, ' \
            f'but got: {len(candidates)}'

        self._chosen: Optional[str] = None
        if isinstance(candidates, dict):
            self._candidates = self._build_ops(candidates, self.module_kwargs)
        elif isinstance(candidates, nn.ModuleDict):
            self._candidates = candidates
        else:
            raise TypeError('candidata_ops should be a `dict` or '
                            f'`nn.ModuleDict` instance, but got '
                            f'{type(candidates)}')

        assert len(self._candidates) >= 1, \
            f'Number of candidate op must greater than or equal to 1, ' \
            f'but got {len(self._candidates)}'

    @staticmethod
    def _build_ops(
            candidates: Union[Dict[str, Dict], nn.ModuleDict],
            module_kwargs: Optional[Dict[str, Dict]] = None) -> nn.ModuleDict:
        """Build candidate operations based on choice configures.

        Args:
            candidates (dict[str, dict] | :obj:`nn.ModuleDict`): the configs
                for the candidate operations or nn.ModuleDict.
            module_kwargs (dict[str, dict], optional): Module initialization
                named arguments.

        Returns:
            ModuleDict (dict[str, Any], optional):  the key of ``ops`` is
                the name of each choice in configs and the value of ``ops``
                is the corresponding candidate operation.
        """
        if isinstance(candidates, nn.ModuleDict):
            return candidates

        ops = nn.ModuleDict()
        for name, op_cfg in candidates.items():
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
        return self._candidates[self._chosen](x)

    def forward_choice(self, x, choice: str) -> Tensor:
        """Forward with the `unfixed` mutable and current choice is not None.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
            choice (str): the chosen key in ``OneShotMutableOP``.

        Returns:
            Tensor: the result of forward the ``choice`` operation.
        """
        assert isinstance(choice, str) and choice in self.choices
        return self._candidates[choice](x)

    def forward_all(self, x) -> Tensor:
        """Forward all choices. Used to calculate FLOPs.

        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.

        Returns:
            Tensor: the result of forward all of the ``choice`` operation.
        """
        outputs = list()
        for op in self._candidates.values():
            outputs.append(op(x))
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
                self._candidates.pop(c)

        self._chosen = chosen
        self.is_fixed = True

    def dump_chosen(self) -> DumpChosen:
        chosen = self.export_chosen()
        meta = dict(all_choices=self.choices)
        return DumpChosen(chosen=chosen, meta=meta)

    def export_chosen(self) -> str:
        assert self.current_choice is not None
        return self.current_choice

    def sample_choice(self) -> str:
        """uniform sampling."""
        return np.random.choice(self.choices, 1)[0]

    @property
    def choices(self) -> List[str]:
        """list: all choices. """
        return list(self._candidates.keys())


@MODELS.register_module()
class OneShotProbMutableOP(OneShotMutableOP):
    """Sampling candidate operation according to probability.

    Args:
        candidates (dict[str, dict]): the configs for the candidate
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
                 candidates: Dict[str, Dict],
                 choice_probs: list = None,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(
            candidates=candidates,
            module_kwargs=module_kwargs,
            alias=alias,
            init_cfg=init_cfg)
        assert choice_probs is not None
        assert sum(choice_probs) - 1 < np.finfo(np.float64).eps, \
            f'Please make sure the sum of the {choice_probs} is 1.'
        self.choice_probs = choice_probs

    def sample_choice(self) -> str:
        """Sampling with probabilities."""
        assert len(self.choice_probs) == len(self._candidates.keys())
        choice = random.choices(
            self.choices, weights=self.choice_probs, k=1)[0]
        return choice
