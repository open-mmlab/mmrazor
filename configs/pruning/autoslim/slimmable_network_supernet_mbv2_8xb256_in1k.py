# defaults to use registries in mmcls
default_scope = 'mmcls'

# !architecture config
# ==========================================================================
architecture = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(type='MobileNetV2', widen_factor=1.5),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1920,
        loss=dict(
            type='LabelSmoothLoss',
            mode='original',
            label_smooth_val=0.1,
            loss_weight=1.0),
        topk=(1, 5),
    ))
# ==========================================================================

# !dataset config
# ==========================================================================
# data preprocessor
data_preprocessor = dict(
    type='ImgDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    bgr_to_rgb=True,
)

dataset_type = 'ImageNet'

# ceph config
use_ceph = True

ceph_file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        './data/imagenet':
        'sproject:s3://openmmlab/datasets/classification/imagenet',
        'data/imagenet':
        'sproject:s3://openmmlab/datasets/classification/imagenet'
    }))
disk_file_client_args = dict(backend='disk')

if use_ceph:
    file_client_args = ceph_file_client_args
else:
    file_client_args = disk_file_client_args

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        scale=224,
        crop_ratio_range=(0.25, 1.0),
        backend='pillow'),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs'),
]

_batch_size_per_gpu = 256

train_dataloader = dict(
    batch_size=_batch_size_per_gpu,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    batch_size=_batch_size_per_gpu,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
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

# !runtime config
# ==========================================================================
# configure log processor
log_processor = dict(window_size=100)

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(
        type='CheckpointHook', max_keep_ckpts=50, save_best='auto',
        interval=1),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ClsVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False
# ==========================================================================

# !autoslim algorithm config
# ==========================================================================
num_samples = 2
model = dict(
    _scope_='mmrazor',
    type='AutoSlim',
    num_samples=num_samples,
    architecture=architecture,
    data_preprocessor=data_preprocessor,
    distiller=dict(
        type='ConfigurableDistiller',
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(type='KLDivergence', tau=1, loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(recorder='fc', from_student=True),
                preds_T=dict(recorder='fc', from_student=False)))),
    mutator=dict(
        type='OneShotChannelMutator',
        mutable_cfg=dict(
            type='OneShotMutableChannel',
            candidate_choices=list(i / 12 for i in range(2, 13)),
            candidate_mode='ratio'),
        tracer_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='ImageClassifierPseudoLoss'))))
# ==========================================================================

# !model wrapper config
# ==========================================================================
model_wrapper_cfg = dict(
    type='mmrazor.AutoSlimDDP',
    broadcast_buffers=False,
    find_unused_parameters=False)
# ==========================================================================

# !scheduler config
# ==========================================================================
paramwise_cfg = dict(
    bias_decay_mult=0.0, norm_decay_mult=0.0, dwconv_decay_mult=0.0)
optimizer = dict(
    type='SGD', lr=0.5, momentum=0.9, nesterov=True, weight_decay=0.0001)
optim_wrapper = dict(
    optimizer=optimizer,
    paramwise_cfg=paramwise_cfg,
    accumulative_counts=num_samples + 2)

# learning policy
max_epochs = 50

param_scheduler = dict(
    type='PolyLR',
    power=1.0,
    eta_min=0.0,
    by_epoch=True,
    end=max_epochs,
    convert_to_iter_based=True)

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='mmrazor.AutoSlimValLoop')
test_cfg = dict()
# ==========================================================================
