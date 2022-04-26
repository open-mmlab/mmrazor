_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs32.py',
    '../../_base_/schedules/mmcls/imagenet_bs256.py',
    '../../_base_/mmcls_runtime.py'
]

# model settings
student = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # noqa: E501

# teacher settings
teacher = dict(
    type='mmcls.ImageClassifier',
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=512,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMClsArchitecture',
        model=student,
    ),
    with_student_loss=True,
    # teacher_trainable and with_teacher_loss have a dependency
    # relationship, if teacher_trainable is false, then
    # with_teacher_loss must be false.
    with_teacher_loss=False,
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        teacher_norm_eval=True,
        components=[
            dict(
                student_module='head.fc',
                teacher_module='head.fc',
                losses=[
                    dict(
                        type='WSLD',
                        name='loss_wsld',
                        tau=2,
                        loss_weight=2.5,
                        num_classes=1000)
                ])
        ]),
)

find_unused_parameters = True
