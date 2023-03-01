_base_ = [
    'mmrazor::_base_/settings/imagenet_bs1024_dsnas.py',
    'mmrazor::_base_/nas_backbones/dsnas_shufflenet_supernet.py',
    'mmcls::_base_/default_runtime.py',
]

custom_hooks = [
    dict(type='mmrazor.DumpSubnetHook', interval=10, by_epoch=True)
]

supernet = dict(
    _scope_='mmcls',
    type='ImageClassifier',
    data_preprocessor=_base_.data_preprocessor,
    backbone=_base_.nas_backbone,
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)))

# model
model = dict(
    type='mmrazor.DSNAS',
    architecture=supernet,
    mutator=dict(type='mmrazor.NasMutator'),
    pretrain_epochs=15,
    finetune_epochs=_base_.search_epochs,
)

model_wrapper_cfg = dict(
    type='mmrazor.DSNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

randomness = dict(seed=48, diff_rank_seed=True)
