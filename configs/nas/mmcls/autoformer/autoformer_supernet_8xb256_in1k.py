_base_ = [
    'mmrazor::_base_/settings/imagenet_bs2048_AdamW.py',
    'mmcls::_base_/default_runtime.py',
]

# data preprocessor
data_preprocessor = dict(
    type='ImgDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    bgr_to_rgb=True,
)

supernet = dict(
    _scope_='mmrazor',
    type='SearchableImageClassifier',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='AutoformerBackbone'),
    neck=None,
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=624,
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)),
)

model = dict(
    type='mmrazor.Autoformer',
    architecture=supernet,
    fix_subnet=None,
    mutators=dict(
        channel_mutator=dict(type='mmrazor.BigNASChannelMutator'),
        value_mutator=dict(type='mmrazor.DynamicValueMutator')),
)

# runtime setting
custom_hooks = [dict(type='EMAHook', momentum=4e-5, priority='ABOVE_NORMAL')]

# checkpoint saving
_base_.default_hooks.checkpoint = dict(
    type='CheckpointHook',
    interval=2,
    by_epoch=True,
    save_best='acc',
    max_keep_ckpts=2)

find_unused_parameters = True
