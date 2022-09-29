# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Generic, List, Optional, Set, Tuple, Type, Union

from mmengine import fileio
from torch.nn import Module

from mmrazor.models.architectures.dynamic_ops import DynamicChannelMixin
from mmrazor.models.mutables import (BaseMutableChannel, ChannelUnitType,
                                     DerivedMutable, MutableChannelContainer,
                                     MutableChannelUnit,
                                     SequentialMutableChannelUnit)
from mmrazor.models.mutables.mutable_channel.units.channel_unit import (
    Channel, ChannelUnit)
from mmrazor.registry import MODELS
from mmrazor.structures.graph import ModuleGraph
from ..base_mutator import BaseMutator


def is_dynamic_op_for_fx_tracer(module, name):
    return isinstance(module, DynamicChannelMixin)


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
                    # default args for MutableChananelGroup
                    default_args={},
                    units = {
                        # config of a group
                        "xxx_group_name": {},
                        ...
                    }
                ),
            The config template of 'units' can be got using
            MutableChannelUnit.config_template()
            Defaults to SequentialMutableChannelUnit.

        parse_cfg (Dict, optional):
            The config to parse the model.
            Defaults to
                dict( type='BackwardTracer',
                loss_calculator=dict(type='ImageClassifierPseudoLoss')).

        init_cfg (dict, optional): initialization configuration dict for
            BaseModule.

    Note:
        There are three ways used in ChannelMutator to parse a model and
        get MutableChannelUnits.
        1. Using tracer. It needs parse_cfg to be the config of a tracer.
        2. Using config. When parse_cfg['type']='Config'. It needs that
        channel_unit_cfg['group']['xxx_group_name] has a key 'channels'.
        3. Using the model with pre-defined dynamic-ops and mutablechannels:
        When parse_cfg['type']='Predefined'.
    """

    # init

    def __init__(
            self,
            channel_unit_cfg: Union[
                dict,
                Type[MutableChannelUnit]] = SequentialMutableChannelUnit,
            parse_cfg: Dict = dict(
                type='BackwardTracer',
                loss_calculator=dict(type='ImageClassifierPseudoLoss')),
            init_cfg: Optional[Dict] = None) -> None:

        super().__init__(init_cfg)

        # tracer
        if isinstance(parse_cfg, dict):
            assert parse_cfg['type'] in [
                'RazorFxTracer', 'BackwardTracer', 'Config', 'Predefined'
            ]
        self.parse_cfg = parse_cfg

        # units
        self._name2group: Dict[str, ChannelUnitType] = {}
        self.units: List[ChannelUnitType] = []

        # group config
        self.channel_unit_cfg = channel_unit_cfg
        self.group_class, self.group_default_args, self.units_cfg = \
            self._parse_channel_unit_cfg(
                channel_unit_cfg)

    def prepare_from_supernet(self, supernet: Module) -> None:
        """Prepare from a model for pruning.

        It includes two steps:
        1. parse the model and get MutableChannelUnits.
        2. call group.prepare_for_pruning for each group.
        """

        self._name2module = dict(supernet.named_modules())

        if 'Tracer' in self.parse_cfg['type']:
            units = self._prepare_from_tracer(supernet, self.parse_cfg)
        elif self.parse_cfg['type'] == 'Config':
            units = self._prepare_from_cfg(supernet, self.units_cfg)
        elif self.parse_cfg['type'] == 'Predefined':
            units = self._prepare_from_predefined_model(supernet)
        else:
            raise NotImplementedError()

        for group in units:
            group.prepare_for_pruning(supernet)
            self._name2group[group.name] = group
        self.units = units

    # ~

    @property
    def mutable_units(self) -> List[ChannelUnitType]:
        """Prunable units."""
        return [group for group in self.units if group.is_mutable]

    def config_template(self,
                        only_mutable_units=False,
                        with_group_init_args=False,
                        with_channels=False):
        """Config template of the mutator.

        Args:
            only_mutable_units (bool, optional): If only return config of
                prunable units. Defaults to False.
            with_group_init_args (bool, optional): If return init_args of
                units. Defaults to False.
            with_channels (bool, optional): if return channel info.
                Defaults to False.

        Example:
            dict(
                channel_unit_cfg = dict(
                    # type of used MutableChannelUnit
                    type ='XxxMutableChannelUnit',
                    # default args for MutableChananelGroup
                    default_args={},
                    # config of units
                    units = {
                        # config of a group
                        "xxx_group_name": {
                            'init_args':{}, # if with_group_init_args
                            'channels':{} # if with_channels
                        },
                        ...
                    }
                ),
                # config of tracer
                parse_cfg={}
            )


        About the detail of the config of each group, please refer to
        MutableChannelUnit.config_template()
        """
        # template of units
        units = self.mutable_units if only_mutable_units else self.units
        units_template = {}
        for group in units:
            units_template[group.name] = group.config_template(
                with_init_args=with_group_init_args,
                with_channels=with_channels)

        # template of mutator
        template = dict(
            type=str(self.__class__.__name__),
            channel_unit_cfg=dict(
                type=str(self.group_class.__name__),
                default_args=self.group_default_args,
                units=units_template),
            parse_cfg=self.parse_cfg)

        return template

    def fix_channel_mutables(self):
        """Fix ChannelMutables."""
        for group in self.units:
            group.fix_chosen()

    # choice manage

    @property
    def current_choices(self) -> Dict:
        """Get current choices."""
        config = self.choice_template
        for group in self.mutable_units:
            config[group.name] = group.current_choice
        return config

    def set_choices(self, config: Dict[str, Union[int, float]]):
        """Set choices."""
        for name, choice in config.items():
            group = self._name2group[name]
            group.current_choice = choice

    def sample_choices(self) -> Dict[str, Union[int, float]]:
        """Sample choices(pruning structure)."""
        template = self.choice_template
        for key in template:
            template[key] = self._name2group[key].sample_choice()
        return template

    @property
    def choice_template(self) -> Dict:
        """Get the chocie template of the Mutator.

        Example:
            {
                'xxx_group_name': xx_choice_value,
                ...
            }
        """
        template = {}
        for group in self.mutable_units:
            template[group.name] = group.current_choice
        return template

    # implementation of abstract functions

    def search_groups(self) -> Dict:
        return self._name2group

    def mutable_class_type(self) -> Type[ChannelUnitType]:
        return self.group_class

    # private methods

    def _convert_channel_unit_to_mutable(self, units: List[ChannelUnit]):
        """Convert ChannelUnits to MutableChannelUnits."""
        mutable_units = []
        for group in units:
            args = copy.copy(self.group_default_args)
            if group.name in self.units_cfg and \
                    'init_args' in self.units_cfg[group.name]:
                args = self.units_cfg[group.name]['init_args']
            mutable_group = self.group_class.init_from_channel_unit(
                group, args)
            mutable_units.append(mutable_group)
        return mutable_units

    def _parse_channel_unit_cfg(
            self,
            channel_unit_cfg) -> Tuple[Type[ChannelUnitType], Dict, Dict]:
        """Parse channel_unit_cfg."""
        if isinstance(channel_unit_cfg, dict):
            group_class = MODELS.module_dict[channel_unit_cfg['type']]

            default_group_args = channel_unit_cfg[
                'default_args'] if 'default_args' in channel_unit_cfg else {}

            group_init_cfg = channel_unit_cfg[
                'units'] if 'units' in channel_unit_cfg else {}
            if isinstance(group_init_cfg, str):
                # load config file
                group_init_cfg = fileio.load(group_init_cfg)
        elif issubclass(channel_unit_cfg, MutableChannelUnit):
            group_class = channel_unit_cfg
            default_group_args = {}
            group_init_cfg = {}
        else:
            raise NotImplementedError()
        return group_class, default_group_args, group_init_cfg

    def _prepare_from_tracer(self, model: Module, parse_cfg: Dict):
        """Initialize units using a tracer."""
        if 'num_input_channel' in parse_cfg:
            num_input_channel = parse_cfg.pop('num_input_channel')
        else:
            num_input_channel = 3
        if self.parse_cfg['type'] == 'BackwardTracer':
            graph = ModuleGraph.init_from_backward_tracer(model, parse_cfg)
        elif self.parse_cfg['type'] == 'RazorFxTracer':
            graph = ModuleGraph.init_from_fx_tracer(model, fx_tracer=parse_cfg)
        else:
            raise NotImplementedError()
        self._graph = graph
        # get ChannelUnits
        units = ChannelUnit.init_from_graph(
            graph, num_input_channel=num_input_channel)
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
        units = []
        for group_key in config:
            init_args = copy.deepcopy(self.group_default_args)
            if 'init_args' in config[group_key]:
                init_args.update(config[group_key]['init_args'])
            config[group_key]['init_args'] = init_args
            group = self.group_class.init_from_cfg(model, config[group_key])
            units.append(group)
        return units

    def _prepare_from_predefined_model(self, model: Module):
        """Initialize units using the model with pre-defined dynamicops and
        mutable-channels."""

        def process_container(contanier: MutableChannelContainer,
                              module,
                              module_name,
                              mutable2units,
                              is_output=True):
            for index, mutable in contanier.mutable_channels.items():
                if isinstance(mutable, DerivedMutable):
                    source_mutables: Set = \
                        mutable._trace_source_mutables()
                    source_channel_mutables = [
                        mutable for mutable in source_mutables
                        if isinstance(mutable, BaseMutableChannel)
                    ]
                    assert len(source_channel_mutables) == 1, (
                        'only support one mutable channel '
                        'used in DerivedMutable')
                    mutable = list(source_channel_mutables)[0]

                if mutable not in mutable2units:
                    if hasattr(self.group_class, 'init_from_mutable_channel'):
                        mutable2units[
                            mutable] = \
                                self.group_class.init_from_mutable_channel(
                                mutable)
                    else:
                        mutable2units[mutable] = self.group_class(
                            mutable.num_channels, **self.group_default_args)

                group: MutableChannelUnit = mutable2units[mutable]
                if is_output:
                    group.add_ouptut_related(
                        Channel(
                            module_name,
                            module,
                            index,
                            is_output_channel=is_output))
                else:
                    group.add_input_related(
                        Channel(
                            module_name,
                            module,
                            index,
                            is_output_channel=is_output))

        mutable2units: Dict = {}
        for name, module in model.named_modules():
            if isinstance(module, DynamicChannelMixin):
                in_container: MutableChannelContainer = \
                    module.get_mutable_attr(
                        'in_channels')
                out_container: MutableChannelContainer = \
                    module.get_mutable_attr(
                        'out_channels')
                process_container(in_container, module, name, mutable2units,
                                  False)
                process_container(out_container, module, name, mutable2units,
                                  True)
        for mutable, group in mutable2units.items():
            if isinstance(mutable, DerivedMutable):
                continue
            else:
                group.mutable_channel = mutable
        units = list(mutable2units.values())
        for group in units:
            self._name2group[group.name] = group
        return units
