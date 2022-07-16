_base_ = [
    'mmrazor::_base_/settings/imagenet_bs1024_spos.py',
    'mmrazor::_base_/nas_backbones/spos_mobilenet_supernet.py',
    'mmcls::_base_/default_runtime.py',
]

# model
supernet = dict(
    type='ImageClassifier',
    # data_preprocessor=_base_.preprocess_cfg,
    backbone=_base_.nas_backbone,
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1728,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)))

model = dict(
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=dict(type='mmrazor.OneShotModuleMutator'))

find_unused_parameters = True
