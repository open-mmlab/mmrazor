# Copyright (c) OpenMMLab. All rights reserved.
from collections import namedtuple
from typing import Any, Dict, List, Optional

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

FIX_MODULES = Dict[str, Any]
FIX_CHANNELS = Dict[str, Any]


# TODO add unittests after mr #29 merged
class FixSubnet(namedtuple('FixSubnet', ['modules', 'channels'])):
    """Subnet data format that can be imported or exported.

    FixSubnet is different from `RANDOM_SUBNET`, it can be directly loaded by
    :func:`mmrazor.core.subnet.utils.load_fix_subnet`. Usually used for
    subnet retraining or transfer learning.

    FixSubnet contains `modules` and `channels`:

    - The keys of `modules` are mutable modules' names, and the values are
      the corresponding choices.

    # TODO add channels examples after mr #29 merged.
    Examples:
      >>> # Assume that a toy NAS model consists of three mutables and some
      >>> # normal pytorch modules.
      >>> # The module names of mutables ​​are op1, op2, and op3.
      >>> # Each mutable contains 4 choices: choice1, choice2,
      >>> # choice3 and choice4.
      >>> # Current choice for each mutable is choice1, choice2, and choice3.
      >>> supernet = ToyNASModel()

      >>> from mmrazor.core.subnet import export_fix_subnet
      >>> fix_subnet = export_fix_subnet(supernet)
      >>> fix_subnet.modules
      {'op1': 'choice1', 'op2': 'choice2', 'op3': 'choice3'}
      >>> fix_subnet.channels
      None
    """

    # TODO design the channels format .
    def __new__(cls,
                modules: Optional[FIX_MODULES] = None,
                channels: Optional[FIX_CHANNELS] = None):
        # catch common mistakes
        assert modules or channels, \
            'modules and channels cannot both be None'
        return super(FixSubnet, cls).__new__(cls, modules, channels)
