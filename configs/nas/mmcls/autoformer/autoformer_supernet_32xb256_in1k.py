_base_ = [
    'mmrazor::_base_/settings/imagenet_bs2048_AdamW.py',
    'mmcls::_base_/default_runtime.py',
]

# data preprocessor
data_preprocessor = dict(
    type='mmcls.ClsDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
    batch_augments=dict(
        augments=[
            dict(type='Mixup', alpha=0.2, num_classes=1000),
            dict(type='CutMix', alpha=1.0, num_classes=1000)
        ],
        probs=[0.5, 0.5]))

supernet = dict(
    _scope_='mmrazor',
    type='SearchableImageClassifier',
    data_preprocessor=data_preprocessor,
    backbone=dict(_scope_='mmrazor', type='AutoformerBackbone'),
    neck=None,
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=624,
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            mode='original',
            num_classes=1000,
            label_smooth_val=0.1,
            loss_weight=1.0),
        topk=(1, 5)),
)

model = dict(
    type='mmrazor.Autoformer',
    architecture=supernet,
    fix_subnet=None,
    mutators=dict(
        channel_mutator=dict(type='mmrazor.BigNASChannelMutator'),
        value_mutator=dict(type='mmrazor.DynamicValueMutator')))

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# checkpoint saving
_base_.default_hooks.checkpoint = dict(
    type='CheckpointHook',
    interval=2,
    by_epoch=True,
    save_best='accuracy/top1',
    max_keep_ckpts=3)

find_unused_parameters = True
