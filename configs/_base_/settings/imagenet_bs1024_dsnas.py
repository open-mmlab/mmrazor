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
    dict(type='mmcls.LoadImageFromFile'),
    dict(type='mmcls.RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='mmcls.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmcls.PackClsInputs'),
]

test_pipeline = [
    dict(type='mmcls.LoadImageFromFile'),
    dict(type='mmcls.ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='mmcls.CenterCrop', crop_size=224),
    dict(type='mmcls.PackClsInputs'),
]

train_dataloader = dict(
    batch_size=512,
    num_workers=15,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='mmcls.DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=100,
    num_workers=15,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='mmcls.DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='mmcls.Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
paramwise_cfg = dict(
    bias_decay_mult=0.0, norm_decay_mult=0.0, dwconv_decay_mult=0.0)

# optimizer
optim_wrapper = dict(
    architecture=dict(
        type='mmcls.SGD', lr=0.025, momentum=0.9, weight_decay=3e-4),
    mutator=dict(type='mmcls.Adam', lr=3e-4, weight_decay=1e-3),
    clip_grad=dict(max_norm=5, norm_type=2))

# leanring policy
param_scheduler = [
    dict(
        type='mmcls.CosineAnnealingLR',
        T_max=50,
        eta_min=0.0,
        by_epoch=True,
    ),
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=240)
val_cfg = dict(
    type='mmrazor.EvaluatorLoop',
    dataloader=val_dataloader,
    evaluator=dict(
        type='mmrazor.NaiveEvaluator',
        metrics=dict(type='mmcls.Accuracy', topk=(1, 5)),
    ))
test_cfg = dict()
