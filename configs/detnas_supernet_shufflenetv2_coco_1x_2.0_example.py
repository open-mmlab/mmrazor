# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# file_client_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox')
test_evaluator = val_evaluator

# training schedule for 1x
train_cfg = dict(by_epoch=True, max_epochs=12)
val_cfg = dict(interval=1)
test_cfg = dict()  # type: ignore

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)

default_scope = 'mmdet'

default_hooks = dict(
    optimizer=dict(type='OptimizerHook', grad_clip=None),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# TODO: Waiting for DetLocalVisualizer to implement
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
visualizer = None
# custom_hooks = [dict(type='DetVisualizationHook', interval=10)]

log_level = 'INFO'
load_from = None
resume = False

# TODO: support auto scaling lr

# model settings
preprocess_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
    pad_size_divisor=32)

norm_cfg = dict(type='BN', requires_grad=True)
supernet = dict(
    type='RetinaNet',
    preprocess_cfg=preprocess_cfg,
    backbone=dict(
        type='mmrazor.SearchableShuffleNetV2',
        norm_cfg=norm_cfg,
        out_indices=(0, 1, 2, 3),
        widen_factor=1.0,
        with_last_layer=False),
    neck=dict(
        type='FPN',
        in_channels=[64, 160, 320, 640],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

mutator = dict(
    type='mmrazor.OneShotModuleMutator',
    placeholder_mapping=dict(
        all_blocks=dict(
            type='mmrazor.OneShotMutableOP',
            choices=dict(
                shuffle_3x3=dict(
                    type='mmrazor.ShuffleBlock',
                    norm_cfg=norm_cfg,
                    kernel_size=3),
                shuffle_5x5=dict(
                    type='mmrazor.ShuffleBlock',
                    norm_cfg=norm_cfg,
                    kernel_size=5),
                shuffle_7x7=dict(
                    type='mmrazor.ShuffleBlock',
                    norm_cfg=norm_cfg,
                    kernel_size=7),
                shuffle_xception=dict(
                    type='mmrazor.ShuffleXception',
                    norm_cfg=norm_cfg,
                ),
            ))))

model = dict(
    type='mmrazor.DetNAS',
    architecture=dict(
        type='mmrazor.MMDetArchitecture',
        model=supernet,
    ),
    mutator=mutator,
    pruner=None,
    distiller=None,
    retraining=False,
)

find_unused_parameters = True
