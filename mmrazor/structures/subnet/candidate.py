# Copyright (c) OpenMMLab. All rights reserved.
from collections import UserList
from typing import Any, Dict, List, Optional, Union


class Candidates(UserList):
    """The data structure of sampled candidate. The format is Union[Dict[str,
    Dict], List[Dict[str, Dict]]].
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
    _format_return = Union[Dict[str, Dict], List[Dict[str, Dict]]]
    _format_input = Union[Dict, List[Dict], Dict[str, Dict], List[Dict[str,
                                                                       Dict]]]
    _indicators = ('score', 'flops', 'params', 'latency')

    def __init__(self, initdata: Optional[_format_input] = None):
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
            round(value.get('score', 0.), 2) for item in self.data
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
        import copy
        assert len(self.data) > 0, ('Got empty candidates.')
        if 'value_subnet' in self.data[0]:
            subnets = []
            for data in self.data:
                subnet = dict()
                _data = copy.deepcopy(data)
                for k1 in ['value_subnet', 'channel_subnet']:
                    for k2 in self._indicators:
                        _data[k1].pop(k2)
                    subnet[k1] = _data[k1]
                subnets.append(subnet)
            return subnets
        else:
            return [eval(key) for item in self.data for key, _ in item.items()]

    def _format(self, data: _format_input) -> _format_return:
        """Transform [Dict, ...] to Union[Dict[str, Dict], List[Dict[str,
        Dict]]].

        Args:
            data: Four types of input are supported:
                1. Dict: only include network information.
                2. List[Dict]: multiple candidates only include network
                    information.
                3. Dict[str, Dict]: network information and the corresponding
                    resources.
                4. List[Dict[str, Dict]]: multiple candidate information.
        Returns:
            Union[Dict[str, Dict], UserList[Dict[str, Dict]]]:
                A dict or a list of dict that contains a pair of network
                information and the corresponding Score | FLOPs | Params |
                Latency results in each candidate.
        Notes:
            Score | FLOPs | Params | Latency:
                1. a candidate resources with a default value of -1 indicates
                    that it has not been estimated.
                2. a candidate resources with a default value of 0 indicates
                    that some indicators have been evaluated.
        """

        def _format_item(
                cond: Union[Dict, Dict[str, Dict]]) -> Dict[str, Dict]:
            """Transform Dict to Dict[str, Dict]."""
            if isinstance(list(cond.values())[0], dict):
                for value in list(cond.values()):
                    for key in list(self._indicators):
                        value.setdefault(key, 0.)
                return cond
            else:
                return {str(cond): {}.fromkeys(self._indicators, -1)}

        if isinstance(data, UserList):
            return [_format_item(i) for i in data.data]

        elif isinstance(data, list):
            return [_format_item(i) for i in data]

        else:
            return _format_item(data)

    def append(self, item: _format_input) -> None:
        """Append operation."""
        item = self._format(item)
        if isinstance(item, list):
            self.data = self.data + item
        else:
            self.data.append(item)

    def insert(self, i: int, item: _format_input) -> None:
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
        self.set_resource(i, score, 'score')

    def set_resource(self,
                     i: int,
                     resources: float,
                     key_indicator: str = 'flops') -> None:
        """Set resources to the specified subnet by index."""
        assert key_indicator in ['score', 'flops', 'params', 'latency']
        for _, value in self.data[i].items():
            value[key_indicator] = resources

    def update_resources(self, resources: list, start: int = 0) -> None:
        """Update resources to the specified candidate."""
        end = start + len(resources)
        assert len(
            self.data) >= end, 'Check the number of candidate resources.'
        for i, item in enumerate(self.data[start:end]):
            for _, value in item.items():
                value.update(resources[i])

    def sort_by(self,
                key_indicator: str = 'score',
                reverse: bool = True) -> None:
        """Sort by a specific indicator in descending order.

        Args:
            key_indicator (str): sort all candidates by key_indicator.
                Defaults to 'score'.
            reverse (bool): sort all candidates in descending order.
        """
        self.data.sort(
            key=lambda x: list(x.values())[0][key_indicator], reverse=reverse)
