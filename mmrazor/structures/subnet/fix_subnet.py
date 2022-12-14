# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict

from mmengine import fileio
from torch import nn

from mmrazor.utils import FixMutable, ValidFixMutable
from mmrazor.utils.typing import DumpChosen


def _dynamic_to_static(model: nn.Module, first_visit: bool = True) -> None:
    # Avoid circular import
    from mmrazor.models.architectures.dynamic_ops import DynamicMixin

    def traverse_children(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, DynamicMixin):
                # TODO: Maybe deal with DynamicSequential in a better way.
                if 'layer' in name:
                    _dynamic_to_static(child, first_visit=False)
                setattr(module, name, child.to_static_op())
            else:
                traverse_children(child)

    if isinstance(model, DynamicMixin) and first_visit:
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

    def load_fix_module(module):
        """Load fix module."""
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
                    module, MutableChannelContainer):
                raise RuntimeError(
                    f'The module name {mutable_name} is not in '
                    'fix_mutable, please check your `fix_mutable`.')
            # {chosen=xx, meta=xx)
            chosen = fix_mutable.get(mutable_name, None)

        if not isinstance(chosen, DumpChosen):
            chosen = DumpChosen(**chosen)
        if not module.is_fixed:
            module.fix_chosen(chosen.chosen)

    for name, module in model.named_modules():
        # The format of `chosen`` is different for each type of mutable.
        # In the corresponding mutable, it will check whether the `chosen`
        # format is correct.
        if isinstance(module, (MutableChannelContainer)):
            continue

        if isinstance(module, BaseMutable):
            if isinstance(module, DerivedMutable):
                for source_mutable in module.source_mutables:
                    load_fix_module(source_mutable)
            else:
                load_fix_module(module)

    # convert dynamic op to static op
    _dynamic_to_static(model)


def export_fix_subnet(model: nn.Module) -> FixMutable:
    """Export subnet that can be loaded by :func:`load_fix_subnet`."""
    # Avoid circular import
    from mmrazor.models.mutables import DerivedMutable, MutableChannelContainer
    from mmrazor.models.mutables.base_mutable import BaseMutable

    def module_dump_chosen(module, fix_subnet):
        if module.alias:
            fix_subnet[module.alias] = module.dump_chosen()
        else:
            fix_subnet[name] = module.dump_chosen()

    fix_subnet: Dict[str, DumpChosen] = dict()
    for name, module in model.named_modules():
        if isinstance(module, BaseMutable):
            if isinstance(module, MutableChannelContainer):
                continue

            elif isinstance(module, DerivedMutable):
                for source_mutable in module.source_mutables:
                    module_dump_chosen(source_mutable, fix_subnet)

            else:
                module_dump_chosen(module, fix_subnet)

    return fix_subnet
