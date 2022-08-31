_base_ = [
    'mmcls::_base_/datasets/cifar10_bs16.py',
    '../../../_base_/vanilla_models/wrn16_2_cifar10.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py',
]
test_evaluator = dict(topk=(1, 5))
