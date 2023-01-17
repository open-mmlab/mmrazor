_base_ = [
    'mmrazor::_base_/settings/cifar10_darts_supernet.py',
    'mmrazor::_base_/nas_backbones/darts_supernet.py',
    'mmcls::_base_/default_runtime.py',
]

custom_hooks = [
    dict(type='mmrazor.DumpSubnetHook', interval=10, by_epoch=True)
]

# model
model = dict(
    type='mmrazor.Darts',
    architecture=dict(
        type='ImageClassifier',
        backbone=_base_.nas_backbone,
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='LinearClsHead',
            num_classes=10,
            in_channels=256,
            loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
            topk=(1, 5),
            cal_acc=True)),
    mutator=dict(type='mmrazor.NasMutator'),
    unroll=True)

model_wrapper_cfg = dict(
    type='mmrazor.DartsDDP',
    broadcast_buffers=False,
    find_unused_parameters=False)

find_unused_parameter = False
