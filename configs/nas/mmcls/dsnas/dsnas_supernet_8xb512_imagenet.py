_base_ = [
    'mmrazor::_base_/settings/imagenet_bs512_dsnas.py',
    'mmrazor::_base_/nas_backbones/dsnas_shufflenet_supernet.py',
    'mmcls::_base_/default_runtime.py',
]

# model
mutator = dict(type='mmrazor.DiffModuleMutator')

model = dict(
    type='mmrazor.Dsnas',
    architecture=dict(
        type='ImageClassifier',
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
            topk=(1, 5))),
    mutator=dict(type='mmrazor.DiffModuleMutator'),
    pretrain_epochs=0,
    finetune_epochs=80,
)

model_wrapper_cfg = dict(
    type='mmrazor.DsnasDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

custom_hooks = [
    dict(type='mmrazor.DumpSubnetHook',
         interval=5,
         max_keep_subnets=2),
]

# TRAINING
optim_wrapper = dict(
    _delete_=True,
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(
        # type='mmrazor.DsnasOptimWrapper',
        optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=4e-5)),
    mutator=dict(
        type='mmrazor.DsnasOptimWrapper',
        optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0)))

randomness = dict(seed=22, diff_rank_seed=True)
