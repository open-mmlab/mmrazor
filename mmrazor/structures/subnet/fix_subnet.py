# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple

from mmengine import fileio
from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from torch import nn

from mmrazor.registry import MODELS
from mmrazor.utils import FixMutable, ValidFixMutable
from mmrazor.utils.typing import DumpChosen


def _dynamic_to_static(model: nn.Module) -> None:
    # Avoid circular import
    from mmrazor.models.architectures.dynamic_ops import DynamicMixin

    def traverse_children(module: nn.Module) -> None:
        for name, mutable in module.items():
            if isinstance(mutable, DynamicMixin):
                module[name] = mutable.to_static_op()
            if hasattr(mutable, '_modules'):
                traverse_children(mutable._modules)

    if isinstance(model, DynamicMixin):
        raise RuntimeError('Root model can not be dynamic op.')

    if hasattr(model, '_modules'):
        traverse_children(model._modules)


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


def _load_fix_subnet_by_mutator(model: nn.Module, mutator_cfg: Dict) -> None:
    if 'channel_unit_cfg' not in mutator_cfg:
        raise ValueError('mutator_cfg must contain key channel_unit_cfg, '
                         f'but got mutator_cfg:'
                         f'{mutator_cfg}')
    mutator_cfg['parse_cfg'] = {'type': 'Config'}
    mutator = MODELS.build(mutator_cfg)
    mutator.prepare_from_supernet(model)
    mutator.set_choices(mutator.current_choices)


def export_fix_subnet(
        model: nn.Module,
        export_subnet_mode: str = 'mutable',
        slice_weight: bool = False) -> Tuple[FixMutable, Optional[Dict]]:
    """Export subnet that can be loaded by :func:`load_fix_subnet`. Include
    subnet structure and subnet weight.

    Args:
        model (nn.Module): The target model to export.
        export_subnet_mode (bool): Subnet export method choice.
            Export by `mutable.dump_chosen()` when set to 'mutable' (NAS)
            Export by `mutator.config_template()` when set to 'mutator' (Prune)
        slice_weight (bool): Export subnet weight. Default to False.

    Return:
        fix_subnet (ValidFixMutable): Exported subnet choice config.
        static_model (Optional[Dict]): Exported static model state_dict.
            Valid when `slice_weight`=True.
    """

    static_model = copy.deepcopy(model)

    fix_subnet = dict()
    if export_subnet_mode == 'mutable':
        fix_subnet = _export_subnet_by_mutable(static_model)
    elif export_subnet_mode == 'mutator':
        fix_subnet = _export_subnet_by_mutator(static_model)
    else:
        raise ValueError(f'Invalid export_subnet_mode {export_subnet_mode}, '
                         'only mutable or mutator is supported.')

    if slice_weight:
        # export subnet ckpt
        print_log('Exporting fixed subnet weight')
        _dynamic_to_static(static_model)
        if next(static_model.parameters()).is_cuda:
            static_model.cuda()
        return fix_subnet, static_model
    else:
        return fix_subnet, None


def _export_subnet_by_mutable(model: nn.Module) -> Dict:

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


def _export_subnet_by_mutator(model: nn.Module) -> Dict:
    if is_model_wrapper(model):
        model = model.module
    if not hasattr(model, 'mutator'):
        raise ValueError('model should contain `mutator` attribute, but got '
                         f'{type(model)} model')
    fix_subnet = model.mutator.config_template(
        with_channels=False, with_unit_init_args=True)

    return fix_subnet
