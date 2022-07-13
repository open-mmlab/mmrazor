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
        """build FixSubnet with modules and channels."""
        # catch common mistakes
        assert modules or channels, \
            'modules and channels cannot both be None'
        return super(FixSubnet, cls).__new__(cls, modules, channels)


class FixSubnetMixin:
    """A Mixin for modules that need to load and export subnets.

    In MMRazor, FixSubnetMixin can be applied to `ALGORITHM` and `BACKBONE`
    to provide them with the ability to load and export subnets.

    Examples:
     >>> # Take the `ALGORITHM` SPOS as example:
     >>> class SPOS(BaseAlgorithm, FixSubnetMixin):
     >>>     def __init__(self, fix_subnet):
     >>>         if fix_subnet:
     >>>             self.load_fix_subnet(fix_subnet)

     >>> # Export fixed subnet in `ALGORITHM` SPOS.
     >>> spos = SPOS(fix_subnet=VALID_FIX_SUBNET)
     >>> spos.export_fix_subnet()
     {'modules': {'layer1': 'shuffle_7x7',
                  'layer2': 'shuffle_3x3',
                  'layer3': 'shuffle_5x5'}}
    """

    # TODO add unittests after mr #29 merged
    def _load_fix_modules(self: nn.Module,
                          fix_modules: FIX_MODULES,
                          prefix: str = ''):
        """Load mutable modules' chosen choice.

        Args:
            fix_modules (FIX_MODULES): the chosen choices in subnet.
            prefix (str): when subnet_dict is not compatible with
                module_name of Supernet.
        """
        for name, module in self.named_modules():
            # The format of `chosen`` is different for each type of mutable.
            # In the corresponding mutable, it will check whether the `chosen`
            # format is correct.
            if isinstance(module, BaseMutable):
                if getattr(module, 'alias', None):
                    alias = module.alias
                    assert alias in fix_modules, \
                        f'The alias {alias} is not in fix_modules ' \
                        f'{fix_modules}, please check your `fix_subnet`.'
                    chosen = fix_modules.get(alias, None)
                else:
                    mutable_name = name.lstrip(prefix)
                    assert mutable_name in fix_modules, \
                        f'The module name {mutable_name} is not in ' \
                        f'fix_modules  {fix_modules} ' \
                        'please check your `fix_subnet`.'
                    chosen = fix_modules.get(mutable_name, None)
                module.fix_chosen(chosen)

    # TODO support load fix channels after mr #29 merged
    # TODO add unittests after mr #29 merged
    def _load_fix_channels(self: nn.Module,
                           fix_modules: FIX_MODULES,
                           prefix: str = ''):
        """Load pruned models."""
        pass

    # TODO add unittests after mr #29 merged
    def load_fix_subnet(self,
                        fix_subnet: Union[str, FixSubnet, Dict[str, Dict]],
                        prefix: str = '') -> None:
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
            self._load_fix_modules(fix_modules, prefix)

        if fix_channels:
            self._load_fix_channels(fix_channels, prefix)

    # TODO refactor after mr #29 merged
    def export_fix_subnet(self: nn.Module) -> FixSubnet:
        """Export subnet that can be loaded by :func:`load_fix_subnet`."""
        fix_subnet: FIX_MODULES = {
            'channels': {},
            'modules': {},
        }
        for name, module in self.named_modules():
            if isinstance(module, BaseMutable):
                fix_subnet['modules'][name] = module.current_choice
        return FixSubnet(**fix_subnet)
