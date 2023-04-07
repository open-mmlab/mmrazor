# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Dict, Generic, List, Optional, Tuple, Type, Union

from mmengine import fileio
from torch.nn import Module, ModuleList

from mmrazor.models.mutables import (ChannelUnitType, MutableChannelUnit,
                                     SequentialMutableChannelUnit)
from mmrazor.models.mutables.mutable_channel.units.channel_unit import \
    ChannelUnit
from mmrazor.models.task_modules.tracer.channel_analyzer import ChannelAnalyzer
from mmrazor.registry import MODELS, TASK_UTILS
from ..base_mutator import BaseMutator


@MODELS.register_module()
class ChannelMutator(BaseMutator, Generic[ChannelUnitType]):
    """ChannelMutator manages the pruning structure of a model.

    Args:
        channel_unit_cfg (Union[ dict, Type[MutableChannelUnit]], optional):
            The config of ChannelUnits. When the channel_unit_cfg
            is a dict, it should follow the template below:
                channel_unit_cfg = dict(
                    # type of used MutableChannelUnit
                    type ='XxxMutableChannelUnit',
                    # default args for MutableChannelUnit
                    default_args={},
                    units = {
                        # config of a unit
                        "xxx_unit_name": {},
                        ...
                    }
                ),
            The config template of 'units' can be got using
            MutableChannelUnit.config_template()
            Defaults to SequentialMutableChannelUnit.

        parse_cfg (Dict, optional):
            The config to parse the model.
            Defaults to
                dict(
                     type='ChannelAnalyzer',
                     demo_input=(1, 3, 224, 224),
                     tracer_type='BackwardTracer')

        init_cfg (dict, optional): initialization configuration dict for
            BaseModule.

    Note:
        There are three ways used in ChannelMutator to parse a model and
        get MutableChannelUnits.
        1. Using tracer. It needs parse_cfg to be the config of the
        ChannelAnalyzer.
        2. Using config. When parse_cfg['type']='Config'. It needs that
        channel_unit_cfg['unit']['xxx_unit_name] has a key 'channels',
        otherwise tracer is required.
        3. Using the model with pre-defined dynamic-ops and mutablechannels:
        When parse_cfg['type']='Predefined'.
    """

    # init

    def __init__(self,
                 channel_unit_cfg: Union[
                     dict,
                     Type[MutableChannelUnit]] = SequentialMutableChannelUnit,
                 parse_cfg: Dict = dict(
                     _scope_='mmrazor',
                     type='ChannelAnalyzer',
                     demo_input=(1, 3, 224, 224),
                     tracer_type='BackwardTracer'),
                 init_cfg: Optional[Dict] = None) -> None:

        super().__init__(init_cfg)

        # tracer
        if isinstance(parse_cfg, dict):
            assert parse_cfg['type'] in [
                'ChannelAnalyzer', 'Config', 'Predefined'
            ]
        self.parse_cfg = parse_cfg

        # units
        self._name2unit: Dict[str, ChannelUnitType] = {}
        self.units: ModuleList[ChannelUnitType] = ModuleList()

        # unit config
        self.channel_unit_cfg = channel_unit_cfg
        self.unit_class, self.unit_default_args, self.units_cfg = \
            self._parse_channel_unit_cfg(
                channel_unit_cfg)

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Prepare from a model for pruning.

        It includes two steps:
        1. parse the model and get MutableChannelUnits.
        2. call unit.prepare_for_pruning for each unit.
        """
        from mmrazor.models.utils import get_module_device
        device = get_module_device(supernet)

        self._name2module = dict(supernet.named_modules())

        if isinstance(self.parse_cfg,
                      ChannelAnalyzer) or 'Analyzer' in self.parse_cfg['type']:
            if isinstance(self.parse_cfg,
                          dict) and 'from_cfg' in self.parse_cfg:
                units = self._prepare_from_cfg(supernet, self.units_cfg)
            else:
                units = self._prepare_from_tracer(supernet, self.parse_cfg)
        elif self.parse_cfg['type'] == 'Config' \
                or 'from_cfg' in self.parse_cfg:
            units = self._prepare_from_cfg(supernet, self.units_cfg)
        elif self.parse_cfg['type'] == 'Predefined':
            units = self._prepare_from_predefined_model(supernet)
        else:
            raise NotImplementedError()
        for i in range(len(units)):
            units[i] = units[i].to(device)
            units[i].prepare_for_pruning(supernet)
            self._name2unit[units[i].name] = units[i]

        self.units = ModuleList(units)

    @property
    def mutable_units(self) -> List[ChannelUnitType]:
        """Prunable units."""
        return [unit for unit in self.units if unit.is_mutable]

    def config_template(self,
                        only_mutable_units=False,
                        with_unit_init_args=False,
                        with_channels=False):
        """Config template of the mutator.

        Args:
            only_mutable_units (bool, optional): Whether only return config of
                prunable units. It can omit unmutable MutableChannelUnits
                to decrease the length of the config. Defaults to False.
            with_unit_init_args (bool, optional): Whether return init_args of
                units. Let it be true, when you want to change the init
                args of units. Defaults to False.
            with_channels (bool, optional): Whether return channel info.
                The channel info can initialization the units without
                tracer. When you want to prune your model without a
                tracer next time, let it be true. Defaults to False.
        Example:
            dict(
                channel_unit_cfg = dict(
                    # type of used MutableChannelUnit
                    type ='XxxMutableChannelUnit',
                    # default args for MutableChananelUnit
                    default_args={},
                    # config of units
                    units = {
                        # config of a unit
                        "xxx_unit_name": {
                            'init_args':{}, # if with_unit_init_args
                            'channels':{} # if with_channels
                        },
                        ...
                    }
                ),
                # config of tracer
                parse_cfg={}
            )


        About the detail of the config of each unit, please refer to
        MutableChannelUnit.config_template()
        """
        # template of units
        units = self.mutable_units if only_mutable_units else self.units
        units_template = {}
        for unit in units:
            units_template[unit.name] = unit.config_template(
                with_init_args=with_unit_init_args,
                with_channels=with_channels)

        # template of mutator
        template = dict(
            type=str(self.__class__.__name__),
            channel_unit_cfg=dict(
                type=str(self.unit_class.__name__),
                default_args=self.unit_default_args,
                units=units_template),
            parse_cfg=self.parse_cfg)

        return template

    def fix_channel_mutables(self):
        """Fix ChannelMutables."""
        for unit in self.units:
            unit.fix_chosen()

    # choice manage

    def sample_choices(self, kind: str = 'random') -> Dict[str, Any]:
        """Sampling by search groups.

        The sampling result of the first mutable of each group is the sampling
        result of this group.

        Returns:
            Dict[int, Any]: Random choices dict.
        """
        assert kind == 'random', f'unsupported the {kind} sample method.'
        template = self.choice_template
        for key in template:
            template[key] = self._name2unit[key].sample_choice()
        return template

    def set_choices(self, choices: Dict[str, Any]) -> None:
        """Set mutables' current choice according to choices sample by
        :func:`sample_choices`.

        Args:
            choices (Dict[int, Any]): Choices dict. The key is group_id in
                search groups, and the value is the sampling results
                corresponding to this group.
        """
        for name, choice in choices.items():
            unit = self._name2unit[name]
            unit.current_choice = choice

    @property
    def current_choices(self) -> Dict:
        """Get current choices."""
        config = self.choice_template
        for unit in self.mutable_units:
            config[unit.name] = unit.current_choice
        return config

    @property
    def choice_template(self) -> Dict:
        """Get the chocie template of the Mutator.

        Example:
            {
                'xxx_unit_name': xx_choice_value,
                ...
            }
        """
        template = {}
        for unit in self.mutable_units:
            template[unit.name] = unit.current_choice
        return template

    @property
    def search_groups(self) -> Dict[int, List]:
        """Search group of the supernet.

        Note:
            Search group is different from search space. The key of search
            group is called ``group_id``, and the value is corresponding
            searchable modules. The searchable modules will have the same
            search space if they are in the same group.

        Returns:
            dict: Search group.
        """
        return self._search_groups

    # private methods

    def _convert_channel_unit_to_mutable(self, units: List[ChannelUnit]):
        """Convert ChannelUnits to MutableChannelUnits."""
        mutable_units = []
        for unit in units:
            args = copy.copy(self.unit_default_args)
            if unit.name in self.units_cfg and \
                    'init_args' in self.units_cfg[unit.name]:
                args = self.units_cfg[unit.name]['init_args']
            mutable_unit = self.unit_class.init_from_channel_unit(unit, args)
            mutable_units.append(mutable_unit)
        return mutable_units

    def _parse_channel_unit_cfg(
            self,
            channel_unit_cfg) -> Tuple[Type[ChannelUnitType], Dict, Dict]:
        """Parse channel_unit_cfg."""
        if isinstance(channel_unit_cfg, dict):
            unit_class = MODELS.module_dict[channel_unit_cfg['type']]

            default_unit_args = channel_unit_cfg[
                'default_args'] if 'default_args' in channel_unit_cfg else {}

            unit_init_cfg = channel_unit_cfg[
                'units'] if 'units' in channel_unit_cfg else {}
            if isinstance(unit_init_cfg, str):
                # load config file
                unit_init_cfg = fileio.load(unit_init_cfg)
        elif issubclass(channel_unit_cfg, MutableChannelUnit):
            unit_class = channel_unit_cfg
            default_unit_args = {}
            unit_init_cfg = {}
        else:
            raise NotImplementedError()
        return unit_class, default_unit_args, unit_init_cfg

    def _prepare_from_tracer(self, model: Module, parse_cfg: Dict):
        """Initialize units using a tracer."""

        if isinstance(parse_cfg, Dict):
            tracer: ChannelAnalyzer = TASK_UTILS.build(parse_cfg)
        else:
            tracer = parse_cfg
        unit_configs = tracer.analyze(model)

        # get ChannelUnits
        units = [
            ChannelUnit.init_from_cfg(model, cfg)
            for cfg in unit_configs.values()
        ]
        # convert to MutableChannelUnits
        units = self._convert_channel_unit_to_mutable(units)
        return units

    def _prepare_from_cfg(self, model, config: Dict):
        """Initialize units using config dict."""
        assert isinstance(self.channel_unit_cfg, dict)
        assert 'units' in self.channel_unit_cfg
        config = self.channel_unit_cfg['units']
        if isinstance(config, str):
            config = fileio.load(config)
        assert isinstance(config, dict)

        if 'Analyzer' in self.parse_cfg['type']:
            self.parse_cfg.pop('from_cfg')
            tracer = TASK_UTILS.build(self.parse_cfg)
            unit_configs = tracer.analyze(model)

        units = []
        for unit_key in config:
            init_args = copy.deepcopy(self.unit_default_args)
            if 'init_args' in config[unit_key]:
                init_args.update(config[unit_key]['init_args'])
            config[unit_key]['init_args'] = init_args
            if 'channels' in config[unit_key]:
                unit = self.unit_class.init_from_cfg(model, config[unit_key])
                unit.name = unit_key
            else:
                try:
                    unit = self._prepare_unit_from_init_cfg(
                        model, config[unit_key], unit_configs[unit_key])
                except ValueError:
                    raise ValueError(
                        'Initializing channel_mutator from the config needs'
                        'to include `channels` or `Analyzer` in the config.')
            units.append(unit)
        return units

    def _prepare_unit_from_init_cfg(self, model: Module, channel_cfg: dict,
                                    init_cfg: dict):
        """Initialize units using the init_cfg, which created by tracer."""
        unit = ChannelUnit.init_from_cfg(model, init_cfg)
        unit = self._convert_channel_unit_to_mutable([unit])[0]
        if 'choice' in channel_cfg:
            unit.current_choice = channel_cfg['choice']
        return unit

    def _prepare_from_predefined_model(self, model: Module):
        """Initialize units using the model with pre-defined dynamicops and
        mutable-channels."""

        units = self.unit_class.init_from_predefined_model(model)

        for unit in units:
            unit.unit_predefined = self.unit_default_args.pop(
                'unit_predefined', False)
        return units
