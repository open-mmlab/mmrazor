# Copyright (c) OpenMMLab. All rights reserved.
from collections import UserList
from typing import Any, Dict, List, Optional, Union


class Candidates(UserList):
    """The data structure of sampled candidate. The format is Union[Dict[Any,
    Dict], List[Dict[Any, Dict]]].

    Examples:
        >>> candidates = Candidates()
        >>> subnet_1 = {'1': 'choice1', '2': 'choice2'}
        >>> candidates.append(subnet_1)
        >>> candidates
        [{"{'1': 'choice1', '2': 'choice2'}":
        {'score': 0.0, 'flops': 0.0, 'params': 0.0, 'latency': 0.0}}]
        >>> candidates.set_resources(0, 49.9, 'flops')
        >>> candidates.set_score(0, 100.)
        >>> candidates
        [{"{'1': 'choice1', '2': 'choice2'}":
        {'score': 100.0, 'flops': 49.9, 'params': 0.0, 'latency': 0.0}}]
        >>> subnet_2 = {'choice_3': 'layer_3', 'choice_4': 'layer_4'}
        >>> candidates.append(subnet_2)
        >>> candidates
        [{"{'1': 'choice1', '2': 'choice2'}":
        {'score': 100.0, 'flops': 49.9, 'params': 0.0, 'latency': 0.0}},
        {"{'choice_3': 'layer_3', 'choice_4':'layer_4'}":
        {'score': 0.0, 'flops': 0.0, 'params': 0.0, 'latency': 0.0}}]
        >>> candidates.subnets
        [{'1': 'choice1', '2': 'choice2'},
        {'choice_3': 'layer_3', 'choice_4': 'layer_4'}]
        >>> candidates.resources('flops')
        [49.9, 0.0]
        >>> candidates.scores
        [100.0, 0.0]
    """
    _format_return = Union[Dict[Any, Dict], List[Dict[Any, Dict]]]
    _indicators = ('score', 'flops', 'params', 'latency')

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
        return [
            value.get('score', 0.) for item in self.data
            for _, value in item.items()
        ]

    def resources(self, key_indicator: str = 'flops') -> List[float]:
        """The resources of candidates."""
        assert key_indicator in ['flops', 'params', 'latency']
        return [
            value.get(key_indicator, 0.) for item in self.data
            for _, value in item.items()
        ]

    @property
    def subnets(self) -> List[Dict]:
        """The subnets of candidates."""
        return [eval(key) for item in self.data for key, _ in item.items()]

    def _format(self, data: Any) -> _format_return:
        """Transform [Dict, ...] to [Dict[Any, Dict], ...]."""

        def _format_item(cond: Any):
            """Transform Dict to str(Dict)."""
            if len(cond.keys()) > 1 and isinstance(
                    list(cond.values())[0], str):
                return {str(cond): {}.fromkeys(self._indicators, -1)}
            else:
                for value in list(cond.values()):
                    for key in list(self._indicators):
                        value.setdefault(key, 0.)
                return cond

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
        for _, value in self.data[i].items():
            value['score'] = score

    def set_resources(self,
                      i: int,
                      resources: float,
                      key_indicator: str = 'flops') -> None:
        """Set resources to the specified subnet by index."""
        assert key_indicator in ['flops', 'params', 'latency']
        for _, value in self.data[i].items():
            value[key_indicator] = resources

    def sort_by(self, key_indicator='score', reverse=True):
        """Sort by a specific indicator.

        Default score.
        """
        self.data.sort(
            key=lambda x: list(x.values())[0][key_indicator], reverse=reverse)
