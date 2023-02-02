_base_ = [
    'mmcls::_base_/default_runtime.py',
    'mmcls::_base_/schedules/imagenet_bs2048.py',
    'mmrazor::_base_/settings/cifar10_bs96_nsga.py',
    'mmrazor::_base_/nas_backbones/nsga_mobilenetv3_supernet.py',
]

supernet = dict(
    _scope_='mmrazor',
    type='SearchableImageClassifier',
    backbone=_base_.nas_backbone,
    neck=dict(type='SqueezeMeanPoolingWithDropout', drop_ratio=0.2),
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)),
    input_resizer_cfg=_base_.input_resizer_cfg,
    connect_head=dict(connect_with_backbone='backbone.last_mutable_channels'),
)

model = dict(
    _scope_='mmrazor',
    type='NSGANetV2',
    architecture=supernet,
    data_preprocessor=_base_.data_preprocessor,
    mutator=dict(type='mmrazor.NasMutator'))

find_unused_parameters = True

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=1, save_best='auto'))
