import argparse
import copy
import os
import os.path as osp
import time
import torch.nn as nn
import torch.nn.functional as F

import mmcv
import torch
from mmcls import __version__
from mmcls.datasets import build_dataset
from mmcls.utils import collect_env, get_root_logger
from mmcv import Config, DictAction, ConfigDict
from mmcv.runner import get_dist_info, init_dist

# Differences from mmclassification
from mmrazor.apis.mmcls.train import set_random_seed, train_model
from mmrazor.models import build_algorithm
from mmrazor.models.builder import build_pruner


# subnet_dict = mmcv.fileio.load('bcnet_search/subnet_top0_score_51.41000175476074.yaml')
cfg = mmcv.Config.fromfile('configs/pruning/bcnet/search.py')
# cfg.algorithm.channel_cfg = 'bcnet_official.yaml'
# cfg.algorithm.retraining=True
a = [30, 30, 14, 72, 72, 22, 100, 100, 22, 129, 129, 32, 105, 105, 32, 86, 86, 32, 182, 182, 54, 384, 384, 54, 153, 153, 54, 384, 384, 54, 384, 384, 72, 374, 374, 72, 518, 518, 72, 518, 518, 152, 960, 960, 152, 816, 816, 152, 960, 960, 256, 1216]
dic = {}
algorithm = build_algorithm(cfg.algorithm)
i = 0
for name, module in algorithm.architecture.model.named_modules():
    if isinstance(module, nn.Conv2d):
        dic[name] = a[i]
        i += 1

subnet_dict = {}
for name, module in algorithm.architecture.model.named_modules():
    if isinstance(module, nn.Conv2d):
        if name in algorithm.pruner.module2group:
            space_id = algorithm.pruner.module2group[name]
            if space_id in subnet_dict:
                assert subnet_dict[space_id] == dic[name]
            else:
                subnet_dict[space_id] = dic[name]
        else:
            assert name not in subnet_dict
            subnet_dict[name] = dic[name]
channel_spaces = algorithm.pruner.channel_spaces
for key, val in subnet_dict.items():
    v = torch.zeros_like(channel_spaces[key])
    v[:, :val] = 1
    subnet_dict[key] = v
print(len(subnet_dict))
algorithm.pruner.set_subnet(subnet_dict)
flops = algorithm.get_subnet_flops()
print(flops)