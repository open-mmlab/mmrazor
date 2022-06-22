# Copyright (c) OpenMMLab. All rights reserved.
from collections import namedtuple
from typing import Any, Dict, Optional, Union

import mmcv
from torch import nn

from mmrazor.models.mutables.base_mutable import BaseMutable

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


# TODO add unittests after mr #29 merged
def _load_fix_modules(model: nn.Module, fix_modules: FIX_MODULES):
    """Load mutable modules' chosen choice."""
    for name, module in model.named_modules():
        # The format of `chosen`` is different for each type of mutable.
        # In the corresponding mutable, it will check whether the `chosen`
        # format is correct.
        chosen = fix_modules.get(name, None)
        if chosen:
            assert isinstance(module, BaseMutable), \
                f'{name} should be a `BaseMutable` instance, but got ' \
                f'{type(module)}'
            module.fix_chosen(chosen)


# TODO support load fix channels after mr #29 merged
# TODO add unittests after mr #29 merged
def _load_fix_channels(model: nn.Module, fix_modules: FIX_MODULES):
    """Load pruned models."""
    pass


# TODO add unittests after mr #29 merged
def load_fix_subnet(model: nn.Module, fix_subnet: Union[str, FixSubnet,
                                                        Dict[str, Dict]]):
    """Load fix subnet."""
    if isinstance(fix_subnet, str):
        subnet = mmcv.fileio.load(fix_subnet)
        subnet = FixSubnet(**subnet)
    elif isinstance(fix_subnet, dict):
        subnet = FixSubnet(**fix_subnet)
    elif isinstance(fix_subnet, FixSubnet):
        subnet = fix_subnet
    else:
        raise TypeError('fix_subnet should be a `str` or `dict` or '
                        f'`FIX_SUBNET` instance, but got '
                        f'{type(fix_subnet)}')

    fix_modules: FIX_MODULES = getattr(subnet, 'modules', None)
    fix_channels: FIX_CHANNELS = getattr(subnet, 'channels', None)

    assert fix_modules or fix_channels, \
        'Please check fix_subnet, modules and channels cannot both be None'

    if fix_modules:
        _load_fix_modules(model, fix_modules)

    if fix_channels:
        _load_fix_channels(model, fix_channels)


# TODO refactor after mr #29 merged
def export_fix_subnet(model: nn.Module) -> FixSubnet:
    """Export subnet that can be loaded by :func:`load_fix_subnet`."""
    fix_subnet = dict()
    for name, module in model.named_modules():
        if isinstance(module, BaseMutable):
            fix_subnet[name] = module.current_choice
    return FixSubnet(**fix_subnet)
