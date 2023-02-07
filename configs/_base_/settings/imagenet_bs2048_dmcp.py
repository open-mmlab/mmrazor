# dataset settings
dataset_type = 'mmcls.ImageNet'

max_search_epochs = 100
# learning rate setting
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.5,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=max_search_epochs,
        eta_min=0.08,
        by_epoch=True,
        begin=10,
        end=max_search_epochs,
        convert_to_iter_based=True),
]

# optimizer setting
paramwise_cfg = dict(norm_decay_mult=0.0, bias_decay_mult=0.0)

optim_wrapper = dict(
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(
        type='OptimWrapper',
        optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=3e-4),
        paramwise_cfg=paramwise_cfg),
    mutator=dict(
        type='OptimWrapper',
        optimizer=dict(type='Adam', lr=0.5, weight_decay=1e-3)))

# data preprocessor
data_preprocessor = dict(
    type='mmcls.ClsDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True, _scope_='mmcls'),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True, _scope_='mmcls'),
    persistent_workers=True,
)
val_evaluator = dict(type='mmcls.Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

evaluation = dict(interval=1, metric='accuracy')

train_cfg = dict(by_epoch=True, max_epochs=max_search_epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()
custom_hooks = [dict(type='DMCPSubnetHook')]
