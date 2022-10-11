# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Tuple, Union

# Channels


class BaseChannel:
    """BaseChannel records information about channels for pruning.

    Args:
        name (str): The name of the channel. When the channel is related with
            a module, the name should be the name of the module in the model.
        module (Any): Module of the channel.
        index (Tuple[int,int]): Index(start,end) of the Channel in the Module
        node (ChannelNode, optional): A ChannelNode corresponding to the
            Channel. Defaults to None.
        is_output_channel (bool, optional): Is the channel output channel.
            Defaults to True.
        expand_ratio (int, optional): Expand ratio of the mask. Defaults to 1.
    """

    # init

    def __init__(self,
                 name,
                 module,
                 index,
                 node=None,
                 is_output_channel=True,
                 expand_ratio=1) -> None:
        self.name = name
        self.module = module
        self.index = index
        self.start = index[0]
        self.end = index[1]

        self.node = node

        self.is_output_channel = is_output_channel
        self.expand_ratio = expand_ratio

    @property
    def num_channels(self) -> int:
        """The number of channels in the Channel."""
        return self.index[1] - self.index[0]

    # others

    def __repr__(self) -> str:
        return f'{self.name}\t{self.index}\t \
        {"out" if self.is_output_channel else "in"}\t\
        expand:{self.expand_ratio}'

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, BaseChannel):
            return self.name == obj.name \
                and self.module == obj.module \
                and self.index == obj.index \
                and self.is_output_channel == obj.is_output_channel \
                and self.expand_ratio == obj.expand_ratio \
                and self.node == obj.node
        else:
            return False


class BaseChannelUnit:
    """BaseChannelUnit is a collection of BaseChannel.

    All  BaseChannels are saved in two lists: self.input_related and
    self.output_related.
    """

    def __init__(self) -> None:

        self.channel_elems: Dict[int, List[ChannelElement]] = {}
        self.input_related: List[BaseChannel] = []
        self.output_related: List[BaseChannel] = []

    # ~

    def add_channel_elem(self, channel_elem: 'ChannelElement', index):
        """Add a ChannelElement to the BaseChannelUnit."""
        self._add_channel_info(channel_elem, index)
        if channel_elem.unit is not None:
            channel_elem.remove_from_unit()
        channel_elem._register_unit(self, index)

    # unit operations

    @classmethod
    def union_units(cls, units: List['BaseChannelUnit']):
        """Union units."""
        assert len(units) > 1
        union_unit = units[0]

        for unit in units[1:]:
            union_unit = BaseChannelUnit.union_two_units(union_unit, unit)
        return union_unit

    @classmethod
    def union_two_units(cls, unit1: 'BaseChannelUnit',
                        unit2: 'BaseChannelUnit'):
        """Union two units."""
        if unit1 is unit2:
            return unit1
        else:
            assert len(unit1) == len(unit2)
            for i in unit1:
                for channel_elem in copy.copy(unit2[i]):
                    unit1.add_channel_elem(channel_elem, i)
            return unit1

    @classmethod
    def split_unit(cls, unit: 'BaseChannelUnit', nums: List[int]):
        """Split a unit to multiple units."""
        new_units = []
        if len(nums) == 1:
            return [unit]
        assert sum(nums) == len(unit)
        for num in nums:
            new_unit = unit._split_a_new_unit(list(range(0, num)))
            new_units.append(new_unit)
        return new_units

    # private methods

    def _clean_channel_info(self, channel_elem: 'ChannelElement', index: int):
        """Clean the info of a ChannelElement."""
        self[index].remove(channel_elem)

    def _add_channel_info(self, channel_elem: 'ChannelElement', index):
        """Add the info of a ChannelElemnt."""
        assert channel_elem.unit is not self
        if index not in self.channel_elems:
            self.channel_elems[index] = []
        self.channel_elems[index].append(channel_elem)

    def _split_a_new_unit(self, indexes: List[int]):
        """Split a part of the unit to a new unit."""
        new_unit = BaseChannelUnit()
        j = 0
        for i in indexes:
            for channel_elem in copy.copy(self[i]):
                new_unit.add_channel_elem(channel_elem, j)
            self.channel_elems.pop(i)
            j += 1
        self._reindex()
        return new_unit

    def _reindex(self):
        """Re-index the owning ChannelElements."""
        j = 0
        for i in copy.copy(self.channel_elems):
            if len(self.channel_elems[i]) == 0:
                self.channel_elems.pop(i)
            else:
                if j < i:
                    for channel_elem in copy.copy(self.channel_elems[i]):
                        if channel_elem.unit is not None:
                            channel_elem.remove_from_unit()
                        self.add_channel_elem(channel_elem, j)
                    self.channel_elems.pop(i)
                    j += 1
                elif j == i:
                    pass
                else:
                    raise Exception()

    # others

    def __repr__(self) -> str:

        def add_prefix(string: str, prefix='  '):
            str_list = string.split('\n')
            str_list = [
                prefix + line if line != '' else line for line in str_list
            ]
            return '\n'.join(str_list)

        def list_repr(lit: List):
            s = '[\n'
            for item in lit:
                s += add_prefix(item.__repr__(), '  ') + '\n'
            s += ']\n'
            return s

        s = ('xxxxx_'
             f'\t{len(self.output_related)},{len(self.input_related)}\n')
        s += '  output_related:\n'
        s += add_prefix(list_repr(self.output_related), ' ' * 4)
        s += '  input_related\n'
        s += add_prefix(list_repr(self.input_related), ' ' * 4)
        return s

    def __iter__(self):
        for i in self.channel_elems:
            yield i

    def __len__(self):
        return len(self.channel_elems)

    def __getitem__(self, key):
        return self.channel_elems[key]


class ChannelElement:
    """Each ChannelElement is the basic element of  a ChannelTensor. It records
    its owing ChannelTensor and BaseChannelUnit.

    Args:
        index (int): The index of the ChannelElement in the ChannelTensor.
    """

    def __init__(self, index_in_tensor: int) -> None:

        self.index_in_channel_tensor = index_in_tensor

        self.unit: Union[BaseChannelUnit, None] = None
        self.index_in_unit = -1

    def remove_from_unit(self):
        """Remove the ChannelElement from its owning BaseChannelUnit."""
        self.unit._clean_channel_info(self, self.index_in_unit)
        self._clean_unit_info()

    # private methods

    def _register_unit(self, unit, index):
        """Register the ChannelElement to a BaseChannelUnit."""
        self.unit = unit
        self.index_in_unit = index

    def _clean_unit_info(self):
        """Clean the unit info in the ChannelElement."""
        self.unit = None
        self.index_in_unit = -1


class ChannelTensor:
    """A ChannelTensor is a list of ChannelElemnts. It can forward through a
    ChannelGraph.

    Args:
        num_channel_elems (int): Number of ChannelElements.
    """

    def __init__(self, num_channel_elems: int) -> None:

        unit = BaseChannelUnit()
        self.channel_elems: List[ChannelElement] = [
            ChannelElement(i) for i in range(num_channel_elems)
        ]
        for channel_elem in self.channel_elems:
            unit.add_channel_elem(channel_elem,
                                  channel_elem.index_in_channel_tensor)

    # unit operations

    def align_units_with_nums(self, nums: List[int]):
        """Align owning units to certain lengths."""
        i = 0
        for start, end in self.unit_dict:
            start_ = start
            new_nums = []
            while start_ < end:
                new_nums.append(nums[i])
                start_ += nums[i]
                i += 1
            BaseChannelUnit.split_unit(self.unit_dict[(start, end)], new_nums)

    @property
    def unit_dict(self) -> Dict[Tuple[int, int], BaseChannelUnit]:
        """Get a dict of owning units."""
        units: Dict[Tuple[int, int], BaseChannelUnit] = {}
        # current_unit = ...
        current_unit_idx = -1
        start = 0
        for i in range(len(self)):
            if i == 0:
                current_unit = self[i].unit
                current_unit_idx = self[i].index_in_unit
                start = 0
            else:
                if current_unit is not self[i].unit or \
                        current_unit_idx > self[i].index_in_unit:
                    units[(start, i)] = current_unit
                    current_unit = self[i].unit
                    current_unit_idx = self[i].index_in_unit
                    start = i
            current_unit_idx = self[i].index_in_unit
        units[(start, len(self))] = current_unit
        return units

    @property
    def unit_list(self) -> List[BaseChannelUnit]:
        """Get a list of owning units."""
        return list(self.unit_dict.values())

    # tensor operations

    @classmethod
    def align_tensors(cls, *tensors: 'ChannelTensor'):
        """Align the lengths of the units of the tensors."""
        assert len(tensors) >= 2
        for tensor in tensors:
            assert len(tensor) == len(
                tensors[0]), f'{len(tensor)}!={len(tensors[0])}'
        aligned_index = cls._index2points(
            *[list(tenser.unit_dict.keys()) for tenser in tensors])
        nums = cls._points2num(aligned_index)
        if len(nums) > 1:
            for tensor in tensors:
                tensor.align_units_with_nums(nums)

    def union(self, tensor1: 'ChannelTensor'):
        """Union the units with the tensor1."""
        # align
        ChannelTensor.align_tensors(self, tensor1)
        # union
        for ch1, ch2 in zip(self.channel_elems, tensor1.channel_elems):
            assert ch1.unit is not None and ch2.unit is not None
            for ch in copy.copy(ch2.unit.channel_elems[ch2.index_in_unit]):
                ch1.unit.add_channel_elem(ch, ch1.index_in_unit)

    def expand(self, ratio) -> 'ChannelTensor':
        """Get a new ChannelTensor which is expanded from this
        ChannelTensor."""
        expanded_tensor = ChannelTensor(len(self) * ratio)
        for i, ch in enumerate(self.channel_elems):
            assert ch.unit is not None
            unit = ch.unit
            for j in range(0, ratio):
                ex_ch = expanded_tensor[i * ratio + j]
                unit.add_channel_elem(ex_ch, ch.index_in_unit)
        return expanded_tensor

    # others

    def __getitem__(self, i: int):
        """Get ith ChannelElement in the ChannelTensor."""
        return self.channel_elems[i]

    def __len__(self):
        """Get length of the ChannelTensor."""
        return len(self.channel_elems)

    @classmethod
    def _index2points(cls, *indexes: List[Tuple[int, int]]):
        """Convert indexes to points."""
        new_index = []
        for index in indexes:
            new_index.extend(index)
        points = set()
        for start, end in new_index:
            points.add(start)
            points.add(end)
        points_list = list(points)
        points_list.sort()
        return points_list

    @classmethod
    def _points2num(cls, indexes: List[int]):
        """Convert a list of sorted points to the length of each block."""
        if len(indexes) == 0:
            return []
        nums = []
        start = 0
        for end in indexes[1:]:
            nums.append(end - start)
            start = end
        return nums
