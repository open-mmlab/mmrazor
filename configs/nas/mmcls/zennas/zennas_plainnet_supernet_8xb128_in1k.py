_base_ = [
    'mmrazor::_base_/settings/imagenet_bs1024_spos.py',
    # 'mmrazor::_base_/nas_backbones/spos_shufflenet_supernet.py',
    'mmcls::_base_/default_runtime.py',
]

nas_backbone = dict(
    _scope_='mmrazor',
    type='MasterNet',
    fix_subnet='./work_dirs/1ms/init_plainnet.txt',
    no_create=True,
    num_classes=1000)

# model
supernet = dict(
    type='ImageClassifier',
    data_preprocessor=_base_.preprocess_cfg,
    backbone=nas_backbone,
    # neck=dict(type='GlobalAveragePooling'),
    # head=dict(
    #     type='LinearClsHead',
    #     num_classes=1000,
    #     in_channels=1024,
    #     loss=dict(
    #         type='LabelSmoothLoss',
    #         num_classes=1000,
    #         label_smooth_val=0.1,
    #         mode='original',
    #         loss_weight=1.0),
    #     topk=(1, 5))
)

model = dict(
    type='mmrazor.ZenNAS',
    architecture=supernet,
    # mutator=dict(type='mmrazor.OneShotModuleMutator')
)

find_unused_parameters = True
