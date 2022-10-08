_base_ = [
    'mmpose::_base_/default_runtime.py',
]
train_cfg = dict(max_epochs=300, val_interval=10)

optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4), clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=300,
        milestones=[170, 220, 280],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
architecture = dict(
    type='mmpose.TopdownPoseEstimator',
    data_preprocessor=dict(
        type='mmpose.PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='mmpose.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
    ),
    head=dict(
        type='mmpose.HeatmapHead',
        in_channels=1843,
        out_channels=17,
        loss=dict(type='mmpose.KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

model = dict(
    _scope_='mmrazor',
    type='DCFF',
    channel_cfgs='configs/pruning/mmpose/dcff/resnet_pose.json',
    architecture=architecture,
    fuse_count=1,
    mutator=dict(
        type='DCFFChannelMutator',
        channl_unit_cfg=dict(
            type='DCFFChannelUnit',
            candidate_choices=0.5,
            candidate_mode='ratio'),
        tracer_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='TopdownPoseEstimatorPseudoLoss'))))

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '/mnt/lustre/zengyi.vendor/mmpose/data/coco/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', target_type='heatmap', encoder=codec),
    dict(type='PackPoseInputs')
]

test_pipeline = [
    dict(type='LoadImage', file_client_args=file_client_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        bbox_file='data/coco/person_detection_results/'
        'COCO_val2017_detections_AP_H_56_person.json',
        pipeline=test_pipeline,
    ))
test_dataloader = val_dataloader

find_unused_parameters = True

val_evaluator = dict(
    type='mmpose.CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator
