_base_ = [
    'mmrazor::_base_/settings/cifar10_darts_subnet.py',
    'mmrazor::_base_/nas_backbones/darts_supernet.py',
    'mmcls::_base_/default_runtime.py',
]

subnet_backbone = _base_.nas_backbone
subnet_backbone.base_channels = 36
subnet_backbone.num_layers = 20
subnet_backbone.auxliary = True
subnet_backbone.aux_channels = 128
subnet_backbone.aux_out_channels = 768
subnet_backbone.out_indices = (19, )
subnet_backbone.norm_cfg = norm_cfg = dict(type='BN', affine=True)

# model
supernet = dict(
    type='ImageClassifier',
    data_preprocessor=_base_.preprocess_cfg,
    backbone=subnet_backbone,
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='mmrazor.DartsSubnetClsHead',
        num_classes=10,
        in_channels=576,
        aux_in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        aux_loss=dict(type='CrossEntropyLoss', loss_weight=0.4),
        topk=(1, 5),
        cal_acc=True))

fix_subnet = 'configs/nas/mmcls/darts/DARTS_SUBNET_CIFAR_PAPER_ALIAS.yaml'

model = dict(
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=None,
    fix_subnet=fix_subnet,
)

find_unused_parameter = False
