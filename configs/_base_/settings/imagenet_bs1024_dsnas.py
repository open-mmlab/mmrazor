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
    dict(type='mmcls.RandomResizedCrop', scale=224),
    dict(type='mmcls.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmcls.PackClsInputs'),
]

test_pipeline = [
    dict(type='mmcls.LoadImageFromFile'),
    dict(type='mmcls.ResizeEdge', scale=256, edge='short'),
    dict(type='mmcls.CenterCrop', crop_size=224),
    dict(type='mmcls.PackClsInputs'),
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
    sampler=dict(type='mmcls.DefaultSampler', shuffle=True),
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
    sampler=dict(type='mmcls.DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='mmcls.Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
paramwise_cfg = dict(bias_decay_mult=0.0, norm_decay_mult=0.0)

optim_wrapper = dict(
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(
        optimizer=dict(
            type='mmcls.SGD', lr=0.5, momentum=0.9, weight_decay=4e-5),
        paramwise_cfg=paramwise_cfg),
    mutator=dict(
        optimizer=dict(
            type='mmcls.Adam', lr=0.001, weight_decay=0.0, betas=(0.5,
                                                                  0.999))))

search_epochs = 85
# leanring policy
param_scheduler = dict(
    architecture=[
        dict(
            type='mmcls.LinearLR',
            end=5,
            start_factor=0.2,
            by_epoch=True,
            convert_to_iter_based=True),
        dict(
            type='mmcls.CosineAnnealingLR',
            T_max=240,
            begin=5,
            end=search_epochs,
            by_epoch=True,
            convert_to_iter_based=True),
        dict(
            type='mmcls.CosineAnnealingLR',
            T_max=160,
            begin=search_epochs,
            end=240,
            eta_min=0.0,
            by_epoch=True,
            convert_to_iter_based=True)
    ],
    mutator=[])

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=240)
val_cfg = dict()
test_cfg = dict()
