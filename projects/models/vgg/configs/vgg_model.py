model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(type='mmrazor.VGGCifar', num_classes=10),
    head=dict(
        type='mmcls.LinearClsHead',
        num_classes=10,
        in_channels=512,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
    ),
)
