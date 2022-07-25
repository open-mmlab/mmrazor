# dataset settings
dataset_type = 'mmcls.CIFAR10'
preprocess_cfg = dict(
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    # loaded images are already RGB format
    to_rgb=False)

train_pipeline = [
    dict(type='mmcls.RandomCrop', crop_size=32, padding=4),
    dict(type='mmcls.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmcls.Cutout', shape=16, pad_val=0, prob=1.0),
    dict(type='mmcls.PackClsInputs'),
]

test_pipeline = [
    dict(type='mmcls.PackClsInputs'),
]

train_dataloader = dict(
    batch_size=96,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='mmcls.DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/cifar10/',
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='mmcls.Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=3e-4),
    clip_grad=dict(max_norm=5, norm_type=2))

# leanring policy
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=600,
        by_epoch=True,
        begin=0,
        end=600,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=600)
val_cfg = dict()  # validate each epoch
test_cfg = dict()  # dataset settings
