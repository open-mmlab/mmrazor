# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Any, Dict, Union

import mmcv
from torch import nn

from mmrazor.models.architectures.dynamic_op.base import DynamicOP
from mmrazor.models.mutables.base_mutable import BaseMutable

FIX_MUTABLE = Dict[str, Any]
VALID_FIX_MUTABLE_TYPE = Union[str, Path, FIX_MUTABLE]


def _dynamic_to_static(model: nn.Module) -> None:

    def traverse_children(module: nn.Module) -> None:
        # TODO
        # dynamicop must have no dynamic child
        for name, child in module.named_children():
            if isinstance(child, DynamicOP):
                setattr(module, name, child.to_static_op())
            else:
                traverse_children(child)

    if isinstance(model, DynamicOP):
        raise RuntimeError(f'Supernet can not be a dynamic model!')

    traverse_children(model)


def load_fix_subnet(model: nn.Module,
                    fix_mutable: VALID_FIX_MUTABLE_TYPE,
                    prefix: str = '') -> None:
    """Load fix subnet."""
    if isinstance(fix_mutable, str):
        fix_mutable = mmcv.fileio.load(fix_mutable)
    if not isinstance(fix_mutable, dict):
        raise TypeError('fix_mutable should be a `str` or `dict`'
                        f'but got {type(fix_mutable)}')

    # fix mutable
    for name, module in model.named_modules():
        if isinstance(module, BaseMutable):
            mutable_name = name.lstrip(prefix)
            assert mutable_name in fix_mutable, \
                f'{mutable_name} is not in fix_mutable {fix_mutable}, ' \
                'please check your `fix_mutable`.'

            chosen = fix_mutable.get(mutable_name)
            module.fix_chosen(chosen)

    # convert dynamic op to static op
    _dynamic_to_static(model)


def export_fix_mutable(model: nn.Module) -> FIX_MUTABLE:
    """Export subnet that can be loaded by :func:`load_fix_subnet`."""
    fix_subnet = dict()
    for name, module in model.named_modules():
        if isinstance(module, BaseMutable):
            assert not module.is_fixed
            fix_subnet[name] = module.dump_chosen()

    return fix_subnet
