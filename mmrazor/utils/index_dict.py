# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict
from typing import Tuple, TypeVar

VT = TypeVar('VT')  # Value type


class IndexDict(OrderedDict):
    """IndexDict inherits from OrderedDict[Tuple[int, int], VT]. Each IndexDict
    object is a OrderDict object which using index(Tuple[int,int]) as key and
    Any as value.

    The key type is Tuple[a: int,b: int]. It indicates a range in
    the [a,b).

    IndexDict has three features:
    1. ensure a key always is a index(Tuple[int,int]).
    1. ensure the the indexes are sorted by ascending order.
    2. ensure there is no overlap among indexes.
    """

    def __setitem__(self, __k: Tuple[int, int], __v):
        """set item."""
        start, end = __k
        assert start < end
        self._assert_no_over_lap(start, end)
        super().__setitem__(__k, __v)
        self._sort()

    def _sort(self):
        """sort the dict accorrding to index."""
        items = sorted(self.items())
        self.clear()
        for k, v in items:
            super().__setitem__(k, v)

    def _assert_no_over_lap(self, start, end):
        """Assert the index [start,end) has no over lav with existed
        indexes."""
        assert (start, end) not in self, 'index overlap'

    def __contains__(self, __o) -> bool:
        """Bool: if the index has any overlap with existed indexes"""
        if super().__contains__(__o):
            return True
        else:
            self._assert_is_index(__o)
            start, end = __o
            existed = False
            for s, e in self.keys():
                existed = (s <= start < e or s < end < e or
                           (s < start and end < e)) or existed

            return existed

    def _assert_is_index(self, index):
        """Assert the index is an instance of Tuple[int,int]"""
        assert isinstance(index, Tuple) \
            and len(index) == 2 \
            and isinstance(index[0], int) \
            and isinstance(index[1], int)
