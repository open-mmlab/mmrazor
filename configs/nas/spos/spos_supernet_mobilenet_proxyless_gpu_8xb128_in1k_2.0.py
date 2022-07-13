# dataset settings
dataset_type = 'ImageNet'
preprocess_cfg = dict(
    # RGB format normalization parameters
    mean=[0., 0., 0.],
    std=[1., 1., 1.],
    # convert image from BGR to RGB
    to_rgb=False,
)

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/imagenet':
        'sproject:s3://openmmlab/datasets/classification/imagenet',
        'data/imagenet':
        'sproject:s3://openmmlab/datasets/classification/imagenet'
    }))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/cache/share/images',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

# /mnt/lustre/share_data/wangjiaqi/data/imagenet',

val_dataloader = dict(
    batch_size=128,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        data_root='/mnt/cache/share/images',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# scheduler

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=4e-5),
    clip_grad=None)

# leanring policy
param_scheduler = [
    dict(type='PolyLR', power=1.0, eta_min=0.0, by_epoch=False, end=300000),
]

# train, val, test setting
train_cfg = dict(by_epoch=False, max_iters=300000)
val_cfg = dict()
test_cfg = dict()

# runtime

# defaults to use registries in mmrazor
default_scope = 'mmcls'

log_processor = dict(
    window_size=100,
    by_epoch=False,
    custom_cfg=[
        dict(
            data_src='loss',
            log_name='loss_large_window',
            method_name='mean',
            window_size=100)
    ])

# configure default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10000,
        save_last=True,
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None
# dict(type='ClsVisualizer', vis_backends=vis_backends, name='visualizer')
# vis_backends = [dict(type='LocalVisBackend')]

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# model
norm_cfg = dict(type='BN')
_STAGE_MUTABLE = dict(
    _scope_='mmrazor',
    type='OneShotMutableOP',
    candidates=dict(
        mb_k3e3=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=3,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU6')),
        mb_k5e3=dict(
            type='MBBlock',
            kernel_size=5,
            expand_ratio=3,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU6')),
        mb_k7e3=dict(
            type='MBBlock',
            kernel_size=7,
            expand_ratio=3,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU6')),
        mb_k3e6=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=6,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU6')),
        mb_k5e6=dict(
            type='MBBlock',
            kernel_size=5,
            expand_ratio=6,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU6')),
        mb_k7e6=dict(
            type='MBBlock',
            kernel_size=7,
            expand_ratio=6,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU6')),
        identity=dict(type='Identity'),
    ))

_FIRST_MUTABLE = dict(
    _scope_='mmrazor',
    type='OneShotMutableOP',
    candidates=dict(
        mb_k3e1=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU6')), ))

arch_setting = [
    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, mutable_cfg.
    [24, 1, 1, _FIRST_MUTABLE],
    [32, 4, 2, _STAGE_MUTABLE],
    [56, 4, 2, _STAGE_MUTABLE],
    [112, 4, 2, _STAGE_MUTABLE],
    [128, 4, 1, _STAGE_MUTABLE],
    [256, 4, 2, _STAGE_MUTABLE],
    [432, 1, 1, _STAGE_MUTABLE]
]

norm_cfg = dict(type='BN')
supernet = dict(
    type='ImageClassifier',
    data_preprocessor=preprocess_cfg,
    backbone=dict(
        _scope_='mmrazor',
        type='SearchableMobileNet',
        first_channels=40,
        last_channels=1728,
        widen_factor=1.0,
        norm_cfg=norm_cfg,
        arch_setting=arch_setting),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1728,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5),
    ),
)

mutator = dict(type='mmrazor.OneShotModuleMutator')

model = dict(
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=mutator,
)

find_unused_parameters = True
