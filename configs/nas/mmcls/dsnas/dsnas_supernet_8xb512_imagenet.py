_base_ = [
    'mmrazor::_base_/settings/imagenet_bs1024_dsnas.py',
    'mmrazor::_base_/nas_backbones/dsnas_shufflenet_supernet.py',
    'mmcls::_base_/default_runtime.py',
]

# model
model = dict(
    type='mmrazor.Dsnas',
    architecture=dict(
        type='ImageClassifier',
        data_preprocessor=_base_.data_preprocessor,
        backbone=_base_.nas_backbone,
        neck=dict(
            type='mmrazor.GlobalAveragePoolingWithDropout',
            kernel_size=7,
            dropout=0.1),
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

# optimizer settings
paramwise_cfg = dict(bias_decay_mult=0.0, norm_decay_mult=0.0)

# TRAINING
optim_wrapper = dict(
    constructor='mmrazor.SeparateOptimWrapperConstructor',
    architecture=dict(
        optimizer=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=4e-5),
        paramwise_cfg=paramwise_cfg),
    mutator=dict(
        optimizer=dict(
            type='Adam', lr=0.001, betas=(0.5, 0.999), weight_decay=0.0)))

randomness = dict(seed=22, diff_rank_seed=True)
