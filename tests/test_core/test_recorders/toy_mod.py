# Copyright (c) OpenMMLab. All rights reserved.
import random

TOY_VAR = 'aaa'


def toy_func(a):
    return a


def toy_func2(a, b):
    return a, b


def toy_list_func(a):
    return [a, a, a]


def execute_toy_func(a):
    toy_func(a)


def execute_toy_func2(a, b):
    toy_func2(a, b)


def execute_toy_list_func(a):
    toy_list_func(a)


class ToyClass:

    TOY_CLS = 'TOY_CLASS'

    def __init__(self):
        self._count = 0

    def toy(self):
        self._count += 1
        return self._count

    def func(self, x, y=0):
        return x + y

    def __call__(self):
        self._count += 1
        return self._count


class Toy():

    def toy_func(self):
        return random.randint(0, 1000)

    def toy_list_func(self):
        return [random.randint(0, 1000) for _ in range(3)]
