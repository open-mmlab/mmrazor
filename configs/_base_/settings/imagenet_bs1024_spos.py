# dataset settings
dataset_type = 'mmcls.ImageNet'
preprocess_cfg = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(_scope_='mmcls', type='LoadImageFromFile'),
    dict(_scope_='mmcls', type='RandomResizedCrop', scale=224),
    dict(
        _scope_='mmcls',
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4),
    dict(_scope_='mmcls', type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(_scope_='mmcls', type='PackClsInputs'),
]

test_pipeline = [
    dict(_scope_='mmcls', type='LoadImageFromFile'),
    dict(_scope_='mmcls', type='ResizeEdge', scale=256, edge='short'),
    dict(_scope_='mmcls', type='CenterCrop', crop_size=224),
    dict(_scope_='mmcls', type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, 5), _scope_='mmcls')

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
paramwise_cfg = dict(
    bias_decay_mult=0.0, norm_decay_mult=0.0, dwconv_decay_mult=0.0)

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=4e-5),
    paramwise_cfg=paramwise_cfg,
    clip_grad=None)

# leanring policy
param_scheduler = dict(
    type='PolyLR',
    power=1.0,
    eta_min=0.0,
    by_epoch=True,
    end=300,
    convert_to_iter_based=True)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300)
val_cfg = dict()
test_cfg = dict()
