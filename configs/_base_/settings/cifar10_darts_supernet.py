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
    dict(type='mmcls.PackClsInputs'),
]

test_pipeline = [
    dict(type='mmcls.PackClsInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/cifar10',
        indices=-25000,
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='mmcls.DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_prefix='data/cifar10/',
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='mmcls.DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='mmcls.Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    architecture=dict(
        type='mmcls.SGD', lr=0.025, momentum=0.9, weight_decay=3e-4),
    mutator=dict(type='mmcls.Adam', lr=3e-4, weight_decay=1e-3),
    clip_grad=dict(max_norm=5, norm_type=2))

# leanring policy
# TODO support different optim use different scheduler (wait mmengine)
param_scheduler = [
    dict(
        type='mmcls.CosineAnnealingLR',
        T_max=50,
        eta_min=1e-3,
        begin=0,
        end=50),
]
# param_scheduler = dict(
#     architecture = dict(
#         type='mmcls.CosineAnnealingLR',
#         T_max=50,
#         eta_min=1e-3,
#         begin=0,
#         end=50),
#     mutator = dict(
#         type='mmcls.ConstantLR',
#         factor=1,
#         begin=0,
#         end=50))

# train, val, test setting
# TODO split cifar dataset
train_cfg = dict(
    type='mmrazor.DartsEpochBasedTrainLoop',
    mutator_dataloader=dict(
        batch_size=64,
        num_workers=4,
        dataset=dict(
            type=dataset_type,
            data_prefix='data/cifar10',
            indices=25000,
            test_mode=False,
            pipeline=train_pipeline),
        sampler=dict(type='mmcls.DefaultSampler', shuffle=True),
        persistent_workers=True,
    ),
    max_epochs=50)

val_cfg = dict()  # validate each epoch
test_cfg = dict()  # dataset settings
