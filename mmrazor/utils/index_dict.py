# Copyright (c) OpenMMLab. All rights reserved.
from typing import OrderedDict, Tuple, TypeVar

VT = TypeVar('VT')


class IndexDict(OrderedDict[Tuple[int, int], VT]):

    def __setitem__(self, __k, __v):
        start, end = __k
        assert start < end
        self._assert_no_over_lap(start, end)
        super().__setitem__(__k, __v)
        self._sort_index()

    def _sort_index(self):
        items = sorted(self.items())
        self.clear()
        for k, v in items:
            super().__setitem__(k, v)

    def _assert_no_over_lap(self, start, end):

        assert (start, end) not in self, 'index overlap'

    def __contains__(self, __o) -> bool:
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
        assert isinstance(index, Tuple) \
            and len(index) == 2 \
            and isinstance(index[0], int) \
            and isinstance(index[1], int)
