# dataset settings
dataset_type = 'mmcls.ImageNet'
data_preprocessor = dict(
    type='mmcls.ClsDataPreprocessor',
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
    batch_size=1024,
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

# leanring policy
param_scheduler = dict(
    architecture=[
        dict(type='mmcls.LinearLR', end=5, by_epoch=True, start_factor=0.0001),
        dict(
            type='mmcls.CosineAnnealingLR',
            T_max=240,
            begin=5,
            eta_min=0.0,
            by_epoch=True,
        ),
    ],
    mutator=[])

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=240)
val_cfg = dict()
test_cfg = dict()