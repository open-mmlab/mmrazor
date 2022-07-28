_base_ = [
    'mmrazor::_base_/settings/imagenet_bs1024_spos.py',
    'mmcls::_base_/default_runtime.py',
]

model = dict(_scope_='mmrazor', type='BigNASMobileNet')
