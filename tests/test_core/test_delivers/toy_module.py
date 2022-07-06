# Copyright (c) OpenMMLab. All rights reserved.
import random

TOY_VAR = 'aaa'


def toy_func():
    return random.randint(0, 1000)


class ToyClass:

    def __init__(self):
        self._count = 0

    def random_int(self):
        return random.randint(0, 1000)

    @property
    def count(self):
        return self._count

    def __call__(self):
        self._count += 1
        return self._count
