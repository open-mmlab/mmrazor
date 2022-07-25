# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.engine.runner.utils import crossover


def test_crossover():
    fake_random_subnet1 = {}
    fake_random_subnet2 = {}
    for i in range(50):
        fake_random_subnet1[i] = f'{i}_choice1'
        fake_random_subnet2[i] = f'{i}_choice2'

    result = crossover(fake_random_subnet1, fake_random_subnet2)

    assert type(result) == type(fake_random_subnet1)
    assert len(result) == len(fake_random_subnet1)
