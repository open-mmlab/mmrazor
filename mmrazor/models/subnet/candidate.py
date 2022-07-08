# Copyright (c) OpenMMLab. All rights reserved.
from collections import UserList
from typing import Any, Dict, List, Optional, Tuple, Union


class Candidates(UserList):
    """The data structure of sampled candidate. The format is [(any, float),
    (any, float), ...].

    Examples:
        >>> candidates = Candidates()
        >>> subnet_1 = {'choice_1': 'layer_1', 'choice_2': 'layer_2'}
        >>> candidates.append(subnet_1)
        >>> candidates
        [({'choice_1': 'layer_1', 'choice_2': 'layer_2'}, 0.0)]
        >>> candidates.set_score(0, 0.9)
        >>> candidates
        [({'choice_1': 'layer_1', 'choice_2': 'layer_2'}, 0.9)]
        >>> subnet_2 = {'choice_3': 'layer_3', 'choice_4': 'layer_4'}
        >>> candidates.append((subnet_2, 0.5))
        >>> candidates
        [({'choice_1': 'layer_1', 'choice_2': 'layer_2'}, 0.9),
        ({'choice_3': 'layer_3', 'choice_4': 'layer_4'}, 0.5)]
        >>> candidates.subnets
        [{'choice_1': 'layer_1', 'choice_2': 'layer_2'},
        {'choice_3': 'layer_3', 'choice_4': 'layer_4'}]
        >>> candidates.scores
        [0.9, 0.5]
    """
    _format_return = Union[Tuple[Any, float], List[Tuple[Any, float]]]

    def __init__(self, initdata: Optional[Any] = None):
        self.data = []
        if initdata is not None:
            initdata = self._format(initdata)
            if isinstance(initdata, list):
                self.data = initdata
            else:
                self.data.append(initdata)

    @property
    def scores(self) -> List[float]:
        """The scores of candidates."""
        return [item[1] for item in self.data]

    @property
    def subnets(self) -> List[Dict]:
        """The subnets of candidates."""
        return [item[0] for item in self.data]

    def _format(self, data: Any) -> _format_return:
        """Transform [any, ...] to [tuple(any, float), ...] Transform any to
        tuple(any, float)."""

        def _format_item(item: Any):
            """Transform any to tuple(any, float)."""
            if isinstance(item, tuple):
                return (item[0], float(item[1]))
            else:
                return (item, 0.)

        if isinstance(data, UserList):
            return [_format_item(i) for i in data.data]

        elif isinstance(data, list):
            return [_format_item(i) for i in data]

        else:
            return _format_item(data)

    def append(self, item: Any) -> None:
        """Append operation."""
        item = self._format(item)
        self.data.append(item)

    def insert(self, i: int, item: Any) -> None:
        """Insert operation."""
        item = self._format(item)
        self.data.insert(i, item)

    def extend(self, other: Any) -> None:
        """Extend operation."""
        other = self._format(other)
        if isinstance(other, list):
            self.data.extend(other)
        else:
            self.data.extend([other])

    def set_score(self, i: int, score: float) -> None:
        """Set score to the specified subnet by index."""
        self.data[i] = (self.data[i][0], float(score))
