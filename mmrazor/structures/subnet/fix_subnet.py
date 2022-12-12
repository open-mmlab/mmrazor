# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging

import torch
from mmengine import fileio
from mmengine.logging import print_log
from torch import nn

from mmrazor.utils import FixMutable, ValidFixMutable
from mmrazor.utils.typing import DumpChosen


def _dynamic_to_static(model: nn.Module) -> None:
    # Avoid circular import
    from mmrazor.models.architectures.dynamic_ops import DynamicMixin

    def traverse_children(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, DynamicMixin):
                setattr(module, name, child.to_static_op())
            else:
                traverse_children(child)

    if isinstance(model, DynamicMixin):
        raise RuntimeError('Root model can not be dynamic op.')

    traverse_children(model)


def load_fix_subnet(model: nn.Module,
                    fix_mutable: ValidFixMutable,
                    prefix: str = '',
                    extra_prefix: str = '') -> None:
    """Load fix subnet."""
    if prefix and extra_prefix:
        raise RuntimeError('`prefix` and `extra_prefix` can not be set at the '
                           f'same time, but got {prefix} vs {extra_prefix}')
    if isinstance(fix_mutable, str):
        fix_mutable = fileio.load(fix_mutable)
    if not isinstance(fix_mutable, dict):
        raise TypeError('fix_mutable should be a `str` or `dict`'
                        f'but got {type(fix_mutable)}')

    from mmrazor.models.architectures.dynamic_ops import DynamicMixin
    if isinstance(model, DynamicMixin):
        raise RuntimeError('Root model can not be dynamic op.')

    # Avoid circular import
    from mmrazor.models.mutables import DerivedMutable, MutableChannelContainer
    from mmrazor.models.mutables.base_mutable import BaseMutable

    for name, module in model.named_modules():
        # The format of `chosen`` is different for each type of mutable.
        # In the corresponding mutable, it will check whether the `chosen`
        # format is correct.
        if isinstance(module, (MutableChannelContainer, DerivedMutable)):
            continue
        if isinstance(module, BaseMutable):
            if not module.is_fixed:
                if getattr(module, 'alias', None):
                    alias = module.alias
                    assert alias in fix_mutable, \
                        f'The alias {alias} is not in fix_modules, ' \
                        'please check your `fix_mutable`.'
                    # {chosen=xx, meta=xx)
                    chosen = fix_mutable.get(alias, None)
                else:
                    if prefix:
                        mutable_name = name.lstrip(prefix)
                    elif extra_prefix:
                        mutable_name = extra_prefix + name
                    else:
                        mutable_name = name
                    if mutable_name not in fix_mutable and not isinstance(
                            module, (DerivedMutable, MutableChannelContainer)):
                        raise RuntimeError(
                            f'The module name {mutable_name} is not in '
                            'fix_mutable, please check your `fix_mutable`.')
                    # {chosen=xx, meta=xx)
                    chosen = fix_mutable.get(mutable_name, None)

                if not isinstance(chosen, DumpChosen):
                    chosen = DumpChosen(**chosen)
                module.fix_chosen(chosen.chosen)

    # convert dynamic op to static op
    _dynamic_to_static(model)


def export_fix_subnet(model: nn.Module,
                      dump_mutable_container: bool = False,
                      dump_derived_mutable: bool = False,
                      export_weight_path: str = '') -> FixMutable:
    """Export subnet that can be loaded by :func:`load_fix_subnet`.

    Args:
        model (nn.Module): The target model to export.
        dump_derived_mutable (bool): Dump information for all derived mutables.
            Default to False.
        export_weight_path (str): Export subnet weight path.
            If empty stop export weight. Default to ''.
    """
    if dump_derived_mutable:
        print_log(
            'Trying to dump information of all derived mutables, '
            'this might harm readability of the exported configurations.',
            level=logging.WARNING)

    # Avoid circular import
    from mmrazor.models.mutables import DerivedMutable, MutableChannelContainer
    from mmrazor.models.mutables.base_mutable import BaseMutable

    if export_weight_path:
        # export subnet ckpt
        static_model = copy.deepcopy(model)
        _dynamic_to_static(static_model)
        torch.save(static_model.state_dict(), export_weight_path)
    fix_subnet = dict()
    for name, module in model.named_modules():
        if isinstance(module, BaseMutable):
            if isinstance(
                    module,
                    MutableChannelContainer) and not dump_mutable_container:
                continue
            if isinstance(module, DerivedMutable) and not dump_derived_mutable:
                continue

            if module.alias:
                fix_subnet[module.alias] = module.dump_chosen()
            else:
                fix_subnet[name] = module.dump_chosen()

    return fix_subnet
