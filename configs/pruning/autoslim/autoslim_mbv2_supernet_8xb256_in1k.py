_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs256_autoslim.py',
    '../../_base_/schedules/mmcls/imagenet_bs2048_autoslim.py',
    '../../_base_/mmcls_runtime.py'
]

model = dict(
    type='mmcls.ImageClassifier',
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

algorithm = dict(
    type='AutoSlim',
    architecture=dict(type='MMClsArchitecture', model=model),
    distiller=dict(
        type='SelfDistiller',
        components=[
            dict(
                student_module='head.fc',
                teacher_module='head.fc',
                losses=[
                    dict(
                        type='KLDivergence',
                        name='loss_kd',
                        tau=1,
                        loss_weight=1,
                    )
                ]),
        ]),
    pruner=dict(
        type='RatioPruner',
        ratios=(2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 7 / 12, 8 / 12, 9 / 12,
                10 / 12, 11 / 12, 1.0)),
    retraining=False,
    bn_training_mode=True,
    input_shape=None)

runner = dict(type='EpochBasedRunner', max_epochs=50)

use_ddp_wrapper = True
