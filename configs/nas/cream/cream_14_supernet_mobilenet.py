# dataset settings
dataset_type = 'ImageNet'

preprocess_cfg = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/imagenet': 's3://openmmlab/datasets/classification/imagenet',
#         'data/imagenet': 's3://openmmlab/datasets/classification/imagenet'
#     }))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=73,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=64),
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=128,
    num_workers=5,
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
    num_workers=5,
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
    dict(type='PolyLR', power=1.0, eta_min=0.0, by_epoch=False),
]

# train, val, test setting
train_cfg = dict(by_epoch=False, max_iters=300000)
val_cfg = dict()
test_cfg = dict()

# runtime

# defaults to use registries in mmrazor
default_scope = 'mmcls'

# configure default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=1000),
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
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ClsVisualizer', vis_backends=vis_backends, name='visualizer')

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

se_cfg = dict(
    ratio=4,
    divisor=8,
    act_cfg=(dict(type='ReLU'),
             dict(
                 type='HSigmoid', bias=3, divisor=6, min_value=0,
                 max_value=1)))

_FIRST_STAGE_MUTABLE = dict(  # DepthwiseSep
    type='OneShotMutableOP',
    candidates=dict(
        depthsepconv=dict(
            type='DepthwiseSeparableConv',
            dw_kernel_size=3,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish'))))

_MIDDLE_STAGE_MUTABLE = dict(
    type='OneShotMutableOP',
    candidates=dict(
        mb_k3e4_se=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=4,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish')),
        mb_k3e6_se=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=6,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish')),
        mb_k5e4_se=dict(
            type='MBBlock',
            kernel_size=5,
            expand_ratio=4,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish')),
        mb_k5e6_se=dict(
            type='MBBlock',
            kernel_size=5,
            expand_ratio=6,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish')),
        mb_k7e4_se=dict(
            type='MBBlock',
            kernel_size=7,
            expand_ratio=4,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish')),
        mb_k7e6_se=dict(
            type='MBBlock',
            kernel_size=7,
            expand_ratio=6,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='Swish'))))

arch_setting = [
    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, stride, mutable cfg.
    [16, 1, 1, _FIRST_STAGE_MUTABLE],
    [24, 1, 2, _MIDDLE_STAGE_MUTABLE],
    [40, 2, 2, _MIDDLE_STAGE_MUTABLE],
    [80, 2, 2, _MIDDLE_STAGE_MUTABLE],
    [96, 1, 1, _MIDDLE_STAGE_MUTABLE],
    [192, 1, 2, _MIDDLE_STAGE_MUTABLE],
]

norm_cfg = dict(type='BN')
supernet = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    data_preprocessor=preprocess_cfg,
    backbone=dict(
        _scope_='mmrazor',
        type='SearchableMobileNet',
        arch_setting=arch_setting,
        first_channels=16,
        last_channels=320,
        widen_factor=1.0,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='Swish'),
        out_indices=(6, ),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='mmrazor.CreamClsHead',
        num_classes=1000,
        in_channels=320,
        num_features=1280,
        act_cfg=dict(type='Swish'),
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
