# Copyright (c) OpenMMLab. All rights reserved.
import json

import torch.nn as nn

from mmrazor.registry import MODELS
from mmrazor.structures.subnet.fix_subnet import (export_fix_subnet,
                                                  load_fix_subnet)
from mmrazor.utils import print_log


@MODELS.register_module()
def PruneDeploySubModel(architecture,
                        mutable_cfg={},
                        divisor=1,
                        data_preprocessor=None,
                        init_cfg=None):
    """A submodel to deploy a pruned model.

    Args:
        architecture (_type_): the model to be pruned.
        mutable_cfg (dict, optional): the channel remaining ratio for each
            unit. Defaults to {}.
        divisor (int, optional): The divisor to make the channel number
            divisible. Defaults to 1.
        data_preprocessor (_type_, optional): Defaults to None.
        init_cfg (_type_, optional):  Defaults to None.

    Returns:
        BaseModel: a BaseModel of mmengine.
    """
    # import avoid circular import
    from mmrazor.models.mutables import SequentialMutableChannelUnit
    from mmrazor.models.mutators import ChannelMutator
    from ..expandable_utils.unit import ExpandableUnit

    #  build architecture
    if isinstance(architecture, dict):
        architecture = MODELS.build(architecture)
    assert isinstance(architecture, nn.Module)

    # to dynamic model

    mutator = ChannelMutator[ExpandableUnit](
        channel_unit_cfg=SequentialMutableChannelUnit)
    mutator.prepare_from_supernet(architecture)
    mutator.set_choices(mutable_cfg)
    print_log(json.dumps(mutator.current_choices, indent=4))

    mutables = export_fix_subnet(architecture)[0]
    load_fix_subnet(architecture, mutables)

    # cooperate with mmdeploy to make the channel divisible after load
    # the checkpoint.
    if divisor != 1:
        setattr(architecture, '_razor_divisor', divisor)

    return architecture
