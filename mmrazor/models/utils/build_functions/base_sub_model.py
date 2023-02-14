# Copyright (c) OpenMMLab. All rights reserved.
# manage sub models for downstream repos
from typing import Dict, Optional, Union

from mmrazor.registry import MODELS
from mmrazor.utils import print_log


@MODELS.register_module()
def BaseSubModel(cfg: dict,
                 fix_subnet: Union[dict, str],
                 mode: str = 'mutable',
                 prefix: str = '',
                 extra_prefix: str = '',
                 init_weight_from_supernet: bool = False,
                 init_cfg: Optional[Dict] = None):
    """_summary_

    Args:
        cfg (dict): Model config.
        fix_subnet (Union[dict, str]): The subnet config.
        mode (str, optional): Subnet mode. Defaults to 'mutable'.
        prefix (str, optional): Prefix to load subnet. Defaults to ''.
        extra_prefix (str, optional): Extra prefix to load subnet.
            Defaults to ''.
        init_weight_from_supernet (bool, optional): Whether to load weight
            from the supernet. Defaults to False.
        init_cfg (Optional[Dict], optional): initial config. Defaults to None.

    Returns:
        _type_: _description_
    """
    model = MODELS.build(cfg)
    # Save path type cfg process, set init_cfg directly.
    if init_cfg:
        # update init_cfg when init_cfg is valid.
        model.init_cfg = init_cfg
    if init_weight_from_supernet:
        # Supernet is modified after load_fix_subnet(), init weight here.
        model.init_weights()
    from mmrazor.structures import load_fix_subnet

    load_fix_subnet(
        model,
        fix_subnet,
        load_subnet_mode=mode,
        prefix=prefix,
        extra_prefix=extra_prefix)

    if init_weight_from_supernet:
        # Supernet is modified after load_fix_subnet().
        model.init_cfg = None

    return model


@MODELS.register_module()
def sub_model(cfg,
              fix_subnet,
              mode: str = 'mutable',
              prefix: str = '',
              extra_prefix: str = '',
              init_weight_from_supernet: bool = False,
              init_cfg: Optional[Dict] = None):
    print_log('sub_model will be deprecated, please use BaseSubModel instead.')
    return BaseSubModel(
        cfg,
        fix_subnet,
        mode=mode,
        prefix=prefix,
        extra_prefix=extra_prefix,
        init_weight_from_supernet=init_weight_from_supernet,
        init_cfg=init_cfg)
