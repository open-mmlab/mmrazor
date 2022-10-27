_base_ = [
    'mmcls::_base_/models/resnet50_cifar.py',
    'mmcls::_base_/datasets/cifar10_bs16.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='FXModelWrapper',
    model={{_base_.model}},
    customed_skipped_method=[
        'mmcls.models.ClsHead._get_predictions',
        'mmcls.models.ClsHead._get_loss'
    ])

custom_hooks = [dict(type='QuantitiveHook')]
