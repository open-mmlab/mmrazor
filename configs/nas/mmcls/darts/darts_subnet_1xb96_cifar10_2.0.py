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
    dict(
        type='Cutout',
        magnitude_key='shape',
        magnitude_range=(1, 16),
        pad_val=0,
        prob=0.5),
]

test_pipeline = [
    dict(type='PackClsInputs'),
]

train_dataloader = dict(
    batch_size=96,
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
norm_cfg = dict(type='BN', affine=True)
mutable_cfg = dict(
    _scope_='mmrazor',
    type='mmrazor.DiffMutableOP',
    candidates=dict(
        zero=dict(type='mmrazor.DartsZero'),
        skip_connect=dict(
            type='mmrazor.DartsSkipConnect',
            norm_cfg=norm_cfg,
            use_drop_path=True),
        max_pool_3x3=dict(
            type='mmrazor.DartsPoolBN',
            pool_type='max',
            norm_cfg=norm_cfg,
            use_drop_path=True),
        avg_pool_3x3=dict(
            type='mmrazor.DartsPoolBN',
            pool_type='avg',
            norm_cfg=norm_cfg,
            use_drop_path=True),
        sep_conv_3x3=dict(
            type='mmrazor.DartsSepConv',
            kernel_size=3,
            norm_cfg=norm_cfg,
            use_drop_path=True),
        sep_conv_5x5=dict(
            type='mmrazor.DartsSepConv',
            kernel_size=5,
            norm_cfg=norm_cfg,
            use_drop_path=True),
        dil_conv_3x3=dict(
            type='mmrazor.DartsDilConv',
            kernel_size=3,
            norm_cfg=norm_cfg,
            use_drop_path=True),
        dil_conv_5x5=dict(
            type='mmrazor.DartsDilConv',
            kernel_size=5,
            norm_cfg=norm_cfg,
            use_drop_path=True),
    ))

route_cfg = dict(
    type='mmrazor.DiffChoiceRoute',
    with_arch_param=True,
)

supernet = dict(
    type='mmcls.ImageClassifier',
    data_preprocessor=preprocess_cfg,
    backbone=dict(
        type='mmrazor.DartsBackbone',
        in_channels=3,
        base_channels=36,
        num_layers=20,
        num_nodes=4,
        stem_multiplier=3,
        auxliary=True,
        aux_channels=128,
        aux_out_channels=768,
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

fix_subnet = 'configs/nas/darts/DARTS_SUBNET_CIFAR_PAPER_ALIAS.yaml'

model = dict(
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=mutator,
    fix_subnet=fix_subnet,
)

find_unused_parameter = False
