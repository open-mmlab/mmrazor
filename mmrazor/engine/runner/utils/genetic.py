# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np

from mmrazor.utils import SingleMutatorRandomSubnet


def crossover(random_subnet1: SingleMutatorRandomSubnet,
              random_subnet2: SingleMutatorRandomSubnet,
              prob: float = 0.5) -> SingleMutatorRandomSubnet:
    """Crossover in genetic algorithm.

    Args:
        random_subnet1 (SINGLE_MUTATOR_RANDOM_SUBNET): One of the subnets to
            crossover.
        random_subnet2 (SINGLE_MUTATOR_RANDOM_SUBNET): One of the subnets to
            crossover.
        prob (float): The probablity of getting choice from `random_subnet2`.
            Defaults to 0.5.

    Returns:
        SINGLE_MUTATOR_RANDOM_SUBNET: The result of crossover.
    """
    assert prob >= 0. and prob <= 1.,  \
        'The probability of crossover has to be between 0 and 1'
    crossover_subnet = copy.deepcopy(random_subnet1)
    for group_id, choice in random_subnet2.items():
        if np.random.random_sample() < prob:
            crossover_subnet[group_id] = choice
    return crossover_subnet
