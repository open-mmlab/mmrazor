_base_ = [
    '../../_base_/datasets/mmdet/coco_detection.py',
    '../../_base_/schedules/mmdet/schedule_2x.py',
    '../../_base_/mmdet_runtime.py'
]

# model settings
student = dict(
    type='mmdet.RetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
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

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth'  # noqa

teacher = dict(
    type='mmdet.RetinaNet',
    init_cfg=dict(type='Pretrained', checkpoint=teacher_ckpt),
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
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

# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMDetArchitecture',
        model=student,
    ),
    distiller=dict(
        type='SingleTeacherDistillerV2',
        teacher=teacher,
        teacher_trainable=False,
        student_recorders=[
            dict(
                type='FunctionOutputs',
                sources=['anchor_inside_flags'],
                import_modules=['mmdet.models.dense_heads.anchor_head']),
            dict(
                type='MethodOutputs',
                sources=['MaxIoUAssigner.assign'],
                import_modules=['mmdet.core']),
            dict(
                type='ModuleOutputs', sources=['bbox_head.retina_cls', 'neck'])
        ],
        teacher_recorders=[
            dict(
                type='FunctionOutputs',
                sources=['anchor_inside_flags'],
                import_modules=['mmdet.models.dense_heads.anchor_head']),
            dict(
                type='MethodOutputs',
                sources=['MaxIoUAssigner.assign'],
                import_modules=['mmdet.core']),
            dict(
                type='ModuleOutputs', sources=['bbox_head.retina_cls', 'neck'])
        ],
        components=[
            dict(
                student_items=[
                    dict(
                        record_type='MethodOutputs',
                        source='mmdet.core.MaxIoUAssigner.assign'),
                    dict(
                        record_type='FunctionOutputs',
                        source=  # noqa: E251
                        'mmdet.models.dense_heads.anchor_head.anchor_inside_flags'  # noqa: E501
                    ),
                    dict(
                        record_type='ModuleOutputs',
                        source='bbox_head.retina_cls'),
                ],
                teacher_items=[
                    dict(
                        record_type='MethodOutputs',
                        source='mmdet.core.MaxIoUAssigner.assign'),
                    dict(
                        record_type='FunctionOutputs',
                        source=  # noqa: E251
                        'mmdet.models.dense_heads.anchor_head.anchor_inside_flags'  # noqa: E501
                    ),
                    dict(
                        record_type='ModuleOutputs',
                        source='bbox_head.retina_cls'),
                ],
                loss=dict(
                    type='RankMimicLoss',
                    loss_weight=4,
                )),
            dict(
                student_items=[
                    dict(record_type='ModuleOutputs', source='neck'),
                    dict(record_type='ModuleOutputs', source='bbox_head.retina_cls'),
                ],
                teacher_items=[
                    dict(record_type='ModuleOutputs', source='neck'),
                    dict(record_type='ModuleOutputs', source='bbox_head.retina_cls'),
                ],
                loss=dict(
                    type='PredictionGuidedFeatureImitation',
                    loss_weight=1.5,
                ))
        ]),
)

find_unused_parameters = True

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
