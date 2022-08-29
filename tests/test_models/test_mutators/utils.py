# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List

from mmengine import fileio

from mmrazor.models.algorithms import SlimmableNetwork


def load_and_merge_channel_cfgs(channel_cfg_paths: List[str]) -> Dict:
    channel_cfgs = list()
    for channel_cfg_path in channel_cfg_paths:
        channel_cfg = fileio.load(channel_cfg_path)
        channel_cfgs.append(channel_cfg)

    return SlimmableNetwork.merge_channel_cfgs(channel_cfgs)
