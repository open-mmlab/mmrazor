_base_ = [
    'mmrazor::_base_/settings/cifar10_darts_supernet.py',
    'mmrazor::_base_/nas_backbones/darts_supernet.py',
    'mmcls::_base_/default_runtime.py',
]

# model
mutator = dict(type='mmrazor.DiffModuleMutator')

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
    mutator=dict(type='mmrazor.DiffModuleMutator'),
    unroll=True)

model_wrapper_cfg = dict(
    type='mmrazor.DartsDDP',
    broadcast_buffers=False,
    find_unused_parameters=False)

# TRAINING
optim_wrapper = dict(
    _delete_=True,
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(
        type='OptimWrapper',
        optimizer=dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=3e-4),
        clip_grad=dict(max_norm=5, norm_type=2)),
    mutator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=3e-4, weight_decay=1e-3)))

find_unused_parameter = False
