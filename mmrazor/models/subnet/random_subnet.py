# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List

# RANDOM_SUBNET means the subnet sampled by one or more mutators. Usually used
# for supernet training or searching.

# SINGLE_MUTATOR_RANDOM_SUBNET sampled by a mutator, its format is a dict, the
# keys of the dict are the group_id in the mutator‘s search groups, and the
# values ​​of the dict are the choices corresponding to all mutables in each
# search group.

# One search group may contains N mutables. More details of search groups see
# docs for :class:`mmrazor.models.mutators.OneShotMutator`.
SINGLE_MUTATOR_RANDOM_SUBNET = Dict[int, Any]

# For some more complex algorithms, multiple mutators may be used, and the
# corresponding format will be a list
MULTI_MUTATORS_RANDOM_SUBNET = List[SINGLE_MUTATOR_RANDOM_SUBNET]
