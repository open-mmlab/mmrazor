# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json

import torch.nn as nn
from mmengine import MODELS
from mmengine.config import Config

from mmrazor.models import BaseAlgorithm
from mmrazor.models.mutators import BaseChannelMutator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='config of the model')

    return parser.parse_args()


def main():
    args = parse_args()
    config = Config.fromfile(args.config)
    model = MODELS.build(config['model'])
    if isinstance(model, BaseAlgorithm):
        mutator = model.mutator
    elif isinstance(model, nn.module):
        mutator = BaseChannelMutator()
        mutator.prepare_from_supernet(model)
    config = mutator.config_template(with_info=False)
    config = json.dumps(config, indent=2)
    print('=' * 100)
    print('config template')
    print('=' * 100)
    print(config)


if __name__ == '__main__':
    main()
