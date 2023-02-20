# Copyright (c) OpenMMLab. All rights reserved.
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

teacher = dict(
    type='mmdet.RetinaNet',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
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
in_channels = 256
temp = 0.5
alpha_fgd = 0.001
beta_fgd = 0.0005
gamma_fgd = 0.0005
lambda_fgd = 0.000005
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMDetArchitecture',
        model=student,
    ),
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                student_module='neck.fpn_convs.0.conv',
                teacher_module='neck.fpn_convs.0.conv',
                losses=[
                    dict(
                        type='FGDLoss',
                        name='loss_fgd_0',
                        in_channels=in_channels,
                        alpha_fgd=alpha_fgd,
                        beta_fgd=beta_fgd,
                        gamma_fgd=gamma_fgd,
                        lambda_fgd=lambda_fgd,
                    )
                ]),
            dict(
                student_module='neck.fpn_convs.1.conv',
                teacher_module='neck.fpn_convs.1.conv',
                losses=[
                    dict(
                        type='FGDLoss',
                        name='loss_fgd_1',
                        in_channels=in_channels,
                        alpha_fgd=alpha_fgd,
                        beta_fgd=beta_fgd,
                        gamma_fgd=gamma_fgd,
                        lambda_fgd=lambda_fgd,
                    )
                ]),
            dict(
                student_module='neck.fpn_convs.2.conv',
                teacher_module='neck.fpn_convs.2.conv',
                losses=[
                    dict(
                        type='FGDLoss',
                        name='loss_fgd_2',
                        in_channels=in_channels,
                        alpha_fgd=alpha_fgd,
                        beta_fgd=beta_fgd,
                        gamma_fgd=gamma_fgd,
                        lambda_fgd=lambda_fgd,
                    )
                ]),
            dict(
                student_module='neck.fpn_convs.3.conv',
                teacher_module='neck.fpn_convs.3.conv',
                losses=[
                    dict(
                        type='FGDLoss',
                        name='loss_fgd_3',
                        in_channels=in_channels,
                        alpha_fgd=alpha_fgd,
                        beta_fgd=beta_fgd,
                        gamma_fgd=gamma_fgd,
                        lambda_fgd=lambda_fgd,
                    )
                ]),
            dict(
                student_module='neck.fpn_convs.4.conv',
                teacher_module='neck.fpn_convs.4.conv',
                losses=[
                    dict(
                        type='FGDLoss',
                        name='loss_fgd_4',
                        in_channels=in_channels,
                        alpha_fgd=alpha_fgd,
                        beta_fgd=beta_fgd,
                        gamma_fgd=gamma_fgd,
                        lambda_fgd=lambda_fgd,
                    )
                ]),
        ]),
)

find_unused_parameters = True

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
