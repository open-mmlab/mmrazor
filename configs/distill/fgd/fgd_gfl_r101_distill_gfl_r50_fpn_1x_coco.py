_base_ = [
    '../../_base_/datasets/mmdet/coco_detection.py',
    '../../_base_/schedules/mmdet/schedule_1x.py',
    '../../_base_/mmdet_runtime.py'
]

# model settings
t_weight = 'https://download.openmmlab.com/mmdetection/v2.0/' + \
           'gfl/gfl_r101_fpn_mstrain_2x_coco/' + \
           'gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'
student = dict(
    type='mmdet.GFL',
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
        add_extra_convs='on_output',
        num_outs=5,
        init_cfg=dict(type='Pretrained', prefix='neck', checkpoint=t_weight)),
    bbox_head=dict(
        type='GFLHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        init_cfg=dict(
            type='Pretrained', prefix='bbox_head', checkpoint=t_weight)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

teacher = dict(
    type='mmdet.GFL',
    init_cfg=dict(type='Pretrained', checkpoint=t_weight),
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='GFLHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# algorithm setting
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
