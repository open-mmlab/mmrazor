# dataset settings
dataset_type = 'CIFAR10'
preprocess_cfg = dict(
    # RGB format normalization parameters
    mean=[125.307, 122.961, 113.8575],
    std=[51.5865, 50.847, 51.255],
    # loaded images are already RGB format
    to_rgb=False)

train_pipeline = [
    dict(type='RandomCrop', crop_size=32, padding=4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/mnt/cache/share_data/dongpeijie/data/cifar10',
        test_mode=False,
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/mnt/cache/share_data/dongpeijie/data/cifar10/',
        test_mode=True,
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(type='Accuracy', topk=(1, ))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    architecture=dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=3e-4),
    mutator=dict(type='Adam', lr=3e-4, weight_decay=1e-3),
    clip_grad=None)

# leanring policy
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=50,
        by_epoch=True,
        min_lr=1e-3,
        begin=0,
        end=50,
    )
]
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=50)
val_cfg = dict(interval=1)  # validate each epoch
test_cfg = dict()

# defaults to use registries in mmcls
default_scope = 'mmcls'

# configure default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', interval=1, save_last=True, max_keep_ckpts=3),
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

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# model
norm_cfg = dict(type='BN', affine=False)
mutable_cfg = dict(
    _scope_='mmrazor',
    type='mmrazor.DiffMutableOP',
    candidates=dict(
        zero=dict(type='mmrazor.DartsZero'),
        skip_connect=dict(type='mmrazor.DartsSkipConnect', norm_cfg=norm_cfg),
        max_pool_3x3=dict(
            type='mmrazor.DartsPoolBN', pool_type='max', norm_cfg=norm_cfg),
        avg_pool_3x3=dict(
            type='mmrazor.DartsPoolBN', pool_type='avg', norm_cfg=norm_cfg),
        sep_conv_3x3=dict(
            type='mmrazor.DartsSepConv', kernel_size=3, norm_cfg=norm_cfg),
        sep_conv_5x5=dict(
            type='mmrazor.DartsSepConv', kernel_size=5, norm_cfg=norm_cfg),
        dil_conv_3x3=dict(
            type='mmrazor.DartsDilConv', kernel_size=3, norm_cfg=norm_cfg),
        dil_conv_5x5=dict(
            type='mmrazor.DartsDilConv', kernel_size=5, norm_cfg=norm_cfg),
    ))

route_cfg = dict(
    type='mmrazor.DiffChoiceRoute',
    with_arch_param=True,
)

supernet = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='mmrazor.DartsBackbone',
        in_channels=3,
        base_channels=36,
        num_layers=20,
        num_nodes=4,
        stem_multiplier=3,
        auxliary=False,
        out_indices=(19, ),
        mutable_cfg=mutable_cfg,
        route_cfg=route_cfg),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmrazor.DartsSubnetClsHead',
        num_classes=10,
        in_channels=576,
        aux_in_channels=768,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        aux_loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=0.4),
        topk=(1, 5),
        cal_acc=True),
)

mutator = dict(type='mmrazor.DiffModuleMutator')

model = dict(
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=mutator,
)

find_unused_parameter = True
