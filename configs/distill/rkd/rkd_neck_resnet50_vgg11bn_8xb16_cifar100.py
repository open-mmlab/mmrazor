_base_ = [
    '../../_base_/datasets/mmcls/cifar100_bs16.py',
    '../../_base_/schedules/mmcls/cifar10_bs128.py',
    '../../_base_/mmcls_runtime.py'
]

# model settings
student = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(type='VGG', depth=11, norm_cfg=dict(type='BN')),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# teacher settings
teacher_ckpt = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth'  # noqa: E501

teacher = dict(
    type='mmcls.ImageClassifier',
    init_cfg=dict(type='Pretrained', checkpoint=teacher_ckpt),
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=100,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
    ))

# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMClsArchitecture',
        model=student,
    ),
    with_student_loss=True,
    with_teacher_loss=False,
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        teacher_norm_eval=True,
        components=[
            dict(
                student_module='neck.gap',
                teacher_module='neck.gap',
                losses=[
                    dict(
                        type='Distance_wise_RKD',
                        name='distance_wise_loss',
                        loss_weight=25.0,
                        with_l2_norm=True),
                    dict(
                        type='Angle_wise_RKD',
                        name='angle_wise_loss',
                        loss_weight=50.0,
                        with_l2_norm=True),
                ])
        ]),
)

find_unused_parameters = True

optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005)
lr_config = dict(policy='step', step=[60, 120, 160], gamma=0.2)
