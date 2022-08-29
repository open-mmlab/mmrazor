model = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    backbone=dict(
        _scope_='mmrazor',
        type='WideResNet',
        depth=16,
        num_stages=3,
        widen_factor=2,
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=128,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

find_unused_parameters = True
