# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmengine import Config

from mmrazor.models.algorithms import ItePruneAlgorithm
from mmrazor.models.task_modules import ResourceEstimator
from mmrazor.models.task_modules.demo_inputs import DefaultDemoInput
from mmrazor.registry import MODELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('-H', default=224, type=int)
    parser.add_argument('-W', default=224, type=int)
    args = parser.parse_args()
    return args


def input_generator_wrapper(model, shape, training, scope=None):

    def input_generator(input_shape):
        inputs = DefaultDemoInput(scope=scope).get_data(
            model, input_shape=input_shape, training=training)
        if isinstance(input, dict) and 'mode' in inputs:
            inputs['mode'] = 'tensor'
        return inputs

    return input_generator


if __name__ == '__main__':
    args = parse_args()
    config = Config.fromfile(args.config)
    H = args.H
    W = args.W

    default_scope = config['default_scope']
    model_config = config['model']
    # model_config['_scope_'] = default_scope
    model: ItePruneAlgorithm = MODELS.build(model_config)

    estimator = ResourceEstimator(
        flops_params_cfg=dict(
            input_shape=(1, 3, H, W),
            print_per_layer_stat=False,
            input_constructor=input_generator_wrapper(
                model,
                (1, 3, H, W),
                training=False,
                scope=default_scope,
            )))
    result = estimator.estimate(model)
    print(result)
