# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine import Config

from mmrazor.models.task_modules import ResourceEstimator
from mmrazor.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('-H', default=224, type=int)
    parser.add_argument('-W', default=224, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = Config.fromfile(args.config)
    H = args.H
    W = args.W

    default_scope = config['default_scope']
    model_config = config['model']
    model_config['_scope_'] = default_scope
    model = MODELS.build(model_config)

    estimator = ResourceEstimator(
        flops_params_cfg=dict(
            input_shape=(1, 3, H, W),
            print_per_layer_stat=True,
            input_constructor=dict(
                type='mmrazor.DefaultDemoInput', scope=default_scope)), )
    result = estimator.estimate(model)
    print(result)
