# Copyright (c) OpenMMLab. All rights reserved.
import copy

import numpy as np

from mmrazor.models.subnet import SINGLE_MUTATOR_RANDOM_SUBNET


def crossover(random_subnet1: SINGLE_MUTATOR_RANDOM_SUBNET,
              random_subnet2: SINGLE_MUTATOR_RANDOM_SUBNET,
              prob: float = 0.5) -> SINGLE_MUTATOR_RANDOM_SUBNET:
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
    crossover_subnet = copy.deepcopy(random_subnet1)
    for group_id, choice in random_subnet2.items():
        if np.random.random_sample() < prob:
            crossover_subnet[group_id] = choice
    return crossover_subnet
