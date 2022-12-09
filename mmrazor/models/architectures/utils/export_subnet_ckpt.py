# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import time

import torch
from mmengine.logging import MMLogger

from mmrazor.structures import load_fix_subnet

logger = MMLogger.get_current_instance()


def export_subnet_checkpoint(model, fix_subnet, prefix: str = ''):
    """Export slimmed model according to fix_subnet."""
    copied_model = copy.deepcopy(model)

    load_fix_subnet(copied_model, fix_subnet)
    if next(copied_model.parameters()).is_cuda:
        copied_model.cuda()

    timestamp_subnet = time.strftime('%Y%m%d_%H%M', time.localtime())
    model_name = f'subnet_{timestamp_subnet}.pth'

    state_dict = copied_model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = v

    save_path = osp.join(prefix, model_name)
    torch.save({'state_dict': new_state_dict, 'meta': {}}, save_path)
    logger.info(f'Subnet checkpoint {model_name} saved in {prefix}')
