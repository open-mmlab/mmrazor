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

stage_ratio_1 = 0.8
stage_ratio_2 = 0.8
stage_ratio_3 = 0.9
stage_ratio_4 = 0.85

# the config template of target_pruning_ratio can be got by
# python ./tools/pruning/get_channel_units.py {config_file} --choice
target_pruning_ratio = {
    'backbone.layer1.0.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.0.conv2_(0, 64)_64': stage_ratio_2,
    'backbone.layer1.0.conv3_(0, 256)_256': stage_ratio_3,
    'backbone.layer1.1.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.1.conv2_(0, 64)_64': stage_ratio_2,
    'backbone.layer1.2.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.2.conv2_(0, 64)_64': stage_ratio_2,
    # block 1 [0.8, 0.8] downsample=[0.9]
    'backbone.layer2.0.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.0.conv2_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.0.conv3_(0, 512)_512': stage_ratio_3,
    'backbone.layer2.1.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.1.conv2_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.2.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.2.conv2_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.3.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.3.conv2_(0, 128)_128': stage_ratio_2,
    # block 2 [0.8, 0.8] downsample=[0.9]
    'backbone.layer3.0.conv1_(0, 256)_256': stage_ratio_1,
    'backbone.layer3.0.conv2_(0, 256)_256': stage_ratio_2,
    'backbone.layer3.0.conv3_(0, 1024)_1024': stage_ratio_3,
    'backbone.layer3.1.conv1_(0, 256)_256': stage_ratio_1,
    'backbone.layer3.1.conv2_(0, 256)_256': stage_ratio_2,
    'backbone.layer3.2.conv1_(0, 256)_256': stage_ratio_1,
    'backbone.layer3.2.conv2_(0, 256)_256': stage_ratio_2,
    'backbone.layer3.3.conv1_(0, 256)_256': stage_ratio_4,
    'backbone.layer3.3.conv2_(0, 256)_256': stage_ratio_4,
    'backbone.layer3.4.conv1_(0, 256)_256': stage_ratio_4,
    'backbone.layer3.4.conv2_(0, 256)_256': stage_ratio_4,
    'backbone.layer3.5.conv1_(0, 256)_256': stage_ratio_4,
    'backbone.layer3.5.conv2_(0, 256)_256': stage_ratio_4,
    # block 3 [0.8, 0.8]*2+[0.8, 0.85]*2 downsample=[0.9]
    'backbone.layer4.0.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.0.conv2_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.0.conv3_(0, 2048)_2048': stage_ratio_3,
    'backbone.layer4.1.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.1.conv2_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.2.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.2.conv2_(0, 512)_512': stage_ratio_4
    # block 4 [0.85, 0.85] downsample=[0.9]
}

model = dict(
    _scope_='mmrazor',
    type='DCFF',
    architecture=architecture,
    mutator_cfg=dict(
        type='DCFFChannelMutator',
        channel_unit_cfg=dict(
            type='DCFFChannelUnit', default_args=dict(choice_mode='ratio')),
        parse_cfg=dict(
            type='ChannelAnalyzer',
            demo_input=(1, 3, 224, 224),
            tracer_type='BackwardTracer')),
    target_pruning_ratio=target_pruning_ratio,
    step_freq=1,
    linear_schedule=False)

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

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

model_wrapper = dict(
    type='mmcv.MMDistributedDataParallel', find_unused_parameters=True)

val_evaluator = dict(
    type='mmpose.CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator

val_cfg = dict(_delete_=True, type='mmrazor.ItePruneValLoop')
