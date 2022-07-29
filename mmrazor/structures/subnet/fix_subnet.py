# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
from torch import nn

from mmrazor.utils import FixMutable, ValidFixMutable


def _dynamic_to_static(model: nn.Module) -> None:
    from mmrazor.models.architectures.dynamic_op.base import DynamicOP

    def traverse_children(module: nn.Module) -> None:
        for name, child in module.named_children():
            traverse_children(child)
            if isinstance(child, DynamicOP):
                setattr(module, name, child.to_static_op())

    if isinstance(model, DynamicOP):
        raise RuntimeError('Supernet can not be a dynamic model!')

    traverse_children(model)


def load_fix_subnet(model: nn.Module,
                    fix_mutable: ValidFixMutable,
                    prefix: str = '') -> None:
    """Load fix subnet."""
    if isinstance(fix_mutable, str):
        fix_mutable = mmcv.fileio.load(fix_mutable)
    if not isinstance(fix_mutable, dict):
        raise TypeError('fix_mutable should be a `str` or `dict`'
                        f'but got {type(fix_mutable)}')
    # Avoid circular import
    from mmrazor.models.mutables.base_mutable import BaseMutable

    for name, module in model.named_modules():
        # The format of `chosen`` is different for each type of mutable.
        # In the corresponding mutable, it will check whether the `chosen`
        # format is correct.
        if isinstance(module, BaseMutable):
            if getattr(module, 'alias', None):
                alias = module.alias
                assert alias in fix_mutable, \
                    f'The alias {alias} is not in fix_modules, ' \
                    'please check your `fix_mutable`.'
                chosen = fix_mutable.get(alias, None)
            else:
                mutable_name = name.lstrip(prefix)
                assert mutable_name in fix_mutable, \
                    f'The module name {mutable_name} is not in ' \
                    'fix_mutable, please check your `fix_mutable`.'
                chosen = fix_mutable.get(mutable_name, None)
            module.fix_chosen(chosen)

    # convert dynamic op to static op
    _dynamic_to_static(model)


def export_fix_subnet(model: nn.Module) -> FixMutable:
    """Export subnet that can be loaded by :func:`load_fix_subnet`."""

    # Avoid circular import
    from mmrazor.models.mutables.base_mutable import BaseMutable

    fix_subnet = dict()
    for name, module in model.named_modules():
        if isinstance(module, BaseMutable):
            assert not module.is_fixed
            if module.alias:
                fix_subnet[module.alias] = module.dump_chosen()
            else:
                fix_subnet[name] = module.dump_chosen()

    return fix_subnet
