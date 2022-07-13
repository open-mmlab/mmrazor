_base_ = [
    'mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
    'mmdet::datasets/coco_detection.py', 'mmdet::schedules/schedule_1x.py',
    'mmdet::default_runtime.py'
]

data_root = '/mnt/lustre/share_data/zhangwenwei/data/coco/'

train_dataloader = dict(dataset=dict(data_root=data_root, ))

visualizer = None
# custom_hooks = [dict(type='DetVisualizationHook', interval=10)]

log_level = 'INFO'
load_from = None
resume = False

# TODO: support auto scaling lr

norm_cfg = dict(type='SyncBN', requires_grad=True)
# model settings
_STAGE_MUTABLE = dict(
    _scope_='mmrazor',
    type='mmrazor.OneShotMutableOP',
    candidates=dict(
        shuffle_3x3=dict(
            type='mmrazor.ShuffleBlock', kernel_size=3, norm_cfg=norm_cfg),
        shuffle_5x5=dict(
            type='mmrazor.ShuffleBlock', kernel_size=5, norm_cfg=norm_cfg),
        shuffle_7x7=dict(
            type='mmrazor.ShuffleBlock', kernel_size=7, norm_cfg=norm_cfg),
        shuffle_xception=dict(
            type='mmrazor.ShuffleXception', norm_cfg=norm_cfg),
    ))

arch_setting = [
    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, mutable_cfg.
    [64, 4, _STAGE_MUTABLE],
    [160, 4, _STAGE_MUTABLE],
    [320, 8, _STAGE_MUTABLE],
    [640, 4, _STAGE_MUTABLE],
]

supernet = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='mmrazor.SearchableShuffleNetV2',
        arch_setting=arch_setting,
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

mutator = dict(type='mmrazor.OneShotModuleMutator')

model = dict(
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=mutator,
)

find_unused_parameters = True
