_base_ = [
    'mmcls::_base_/datasets/imagenet_bs32.py',
    'mmcls::resnet/resnet34_8xb32_in1k.py',
    'mmcls::_base_/schedules/imagenet_bs256.py',
    'mmcls::_base_/default_runtime.py'
]

