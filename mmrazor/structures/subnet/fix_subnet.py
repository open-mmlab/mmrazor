# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
from typing import Dict

from mmengine import fileio
from mmengine.logging import print_log
from torch import nn

from mmrazor.registry import MODELS
from mmrazor.utils import ValidFixMutable
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
                    load_subnet_mode: str = 'mutable',
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

    if load_subnet_mode == 'mutable':
        _load_fix_subnet_by_mutable(model, fix_mutable, prefix, extra_prefix)
    elif load_subnet_mode == 'mutator':
        _load_fix_subnet_by_mutator(model, fix_mutable)
    else:
        raise ValueError(f'Invalid load_subnet_mode {load_subnet_mode}, '
                         'only mutable or mutator is supported.')

    # convert dynamic op to static op
    _dynamic_to_static(model)


def _load_fix_subnet_by_mutable(model: nn.Module,
                                fix_mutable: Dict,
                                prefix: str = '',
                                extra_prefix: str = '') -> None:
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


def _load_fix_subnet_by_mutator(model: nn.Module, mutator_cfg: Dict) -> None:
    if 'channel_unit_cfg' not in mutator_cfg:
        raise ValueError('mutator_cfg must contain key channel_unit_cfg, '
                         f'but got mutator_cfg:'
                         f'{mutator_cfg}')
    mutator_cfg['parse_cfg'] = {'type': 'Config'}
    mutator = MODELS.build(mutator_cfg)
    mutator.prepare_from_supernet(model)
    mutator.set_choices(mutator.current_choices)


def export_fix_subnet(model: nn.Module,
                      export_subnet_mode: str = 'mutable',
                      dump_derived_mutable: bool = False,
                      export_weight: bool = False):
    """Export subnet that can be loaded by :func:`load_fix_subnet`. Include
    subnet structure and subnet weight.

    Args:
        model (nn.Module): The target model to export.
        export_subnet_mode (bool): Subnet export method choice.
            Export by `mutable.dump_chosen()` when set to 'mutable' (NAS)
            Export by `mutator.config_template()` when set to 'mutator' (Prune)
        dump_derived_mutable (bool): Dump information for all derived mutables.
            Valid when `export_subnet_mode`='mutable'. Default to False.
        export_weight (bool): Export subnet weight. Default to False.

    Return:
        fix_subnet (ValidFixMutable): Exported subnet choice config.
        static_model (nn.Module): Exported static model.
            Valid when `export_weight`=True.
    """

    static_model = copy.deepcopy(model)

    fix_subnet = dict()
    if export_subnet_mode == 'mutable':
        fix_subnet = _export_subnet_by_mutable(static_model,
                                               dump_derived_mutable)
    elif export_subnet_mode == 'mutator':
        fix_subnet = _export_subnet_by_mutator(static_model)
    else:
        raise ValueError(f'Invalid export_subnet_mode {export_subnet_mode}, '
                         'only mutable or mutator is supported.')

    if export_weight:
        # export subnet ckpt
        print_log('Exporting fixed subnet weight')
        _dynamic_to_static(static_model)
        return fix_subnet, static_model
    else:
        return fix_subnet


def _export_subnet_by_mutable(model: nn.Module,
                              dump_derived_mutable: bool) -> Dict:
    if dump_derived_mutable:
        print_log(
            'Trying to dump information of all derived mutables, '
            'this might harm readability of the exported configurations.',
            level=logging.WARNING)

    # Avoid circular import
    from mmrazor.models.mutables import DerivedMutable, MutableChannelContainer
    from mmrazor.models.mutables.base_mutable import BaseMutable

    fix_subnet = dict()
    for name, module in model.named_modules():
        if isinstance(module, BaseMutable):
            if isinstance(module,
                          (MutableChannelContainer,
                           DerivedMutable)) and not dump_derived_mutable:
                continue

            if module.alias:
                fix_subnet[module.alias] = module.dump_chosen()
            else:
                fix_subnet[name] = module.dump_chosen()
    return fix_subnet


def _export_subnet_by_mutator(model: nn.Module) -> Dict:
    if not hasattr(model, 'mutator'):
        raise ValueError('model should contain `mutator` instance, but got '
                         f'{type(model)} model')
    fix_subnet = model.mutator.config_template(
        with_channels=False, with_unit_init_args=True)

    return fix_subnet
