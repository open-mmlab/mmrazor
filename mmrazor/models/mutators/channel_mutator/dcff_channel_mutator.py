# Copyright (c) OpenMMLab. All rights reserved.
from typing import Type, Union

from mmrazor.models.architectures.dynamic_ops import FuseConv2d
from mmrazor.models.mutables import DCFFChannelUnit
from mmrazor.registry import MODELS
from .channel_mutator import ChannelMutator, ChannelUnitType


@MODELS.register_module()
class DCFFChannelMutator(ChannelMutator[DCFFChannelUnit]):
    """DCFF channel mutable based channel mutator. It uses DCFFChannelUnit.

    Args:
        channel_unit_cfg (Union[dict, Type[ChannelUnitType]], optional):
            Config of MutableChannelUnits. Defaults to
            dict( type='DCFFChannelUnit', units={}).
        parse_cfg (Dict): The config of the tracer to parse the model.
            Defaults to dict( type='BackwardTracer',
                loss_calculator=dict(type='ImageClassifierPseudoLoss')).
            Change loss_calculator according to task and backbone.
    """

    def __init__(self,
                 channel_unit_cfg: Union[dict, Type[ChannelUnitType]] = dict(
                     type='DCFFChannelUnit', units={}),
                 parse_cfg=dict(
                     type='ChannelAnalyzer',
                     demo_input=(1, 3, 224, 224),
                     tracer_type='BackwardTracer'),
                 **kwargs) -> None:
        super().__init__(channel_unit_cfg, parse_cfg, **kwargs)

    def calc_information(self, tau: float):
        """Calculate channel's kl and apply softmax pooling on channel to solve
        CUDA out of memory problem. KL calculation & pool are conducted in ops.

        Args:
            tau (float): temporature calculated by iter or epoch
        """
        # Calculate the filter importance of the current epoch.
        for layerid, unit in enumerate(self.units):
            for channel in unit.output_related:
                if isinstance(channel.module, FuseConv2d):
                    layeri_softmaxp = channel.module.get_pooled_channel(tau)
                    # update fuseconv op's selected layeri_softmax
                    channel.module.set_forward_args(choice=layeri_softmaxp)
