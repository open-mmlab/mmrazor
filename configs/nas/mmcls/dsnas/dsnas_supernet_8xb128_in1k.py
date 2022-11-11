_base_ = [
    'mmrazor::_base_/settings/imagenet_bs1024_dsnas.py',
    'mmrazor::_base_/nas_backbones/dsnas_shufflenet_supernet.py',
    'mmcls::_base_/default_runtime.py',
]

# model
model = dict(
    type='mmrazor.DSNAS',
    architecture=dict(
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
            topk=(1, 5))),
    mutator=dict(type='mmrazor.DiffModuleMutator'),
    pretrain_epochs=15,
    finetune_epochs=_base_.search_epochs,
)

model_wrapper_cfg = dict(
    type='mmrazor.DSNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

randomness = dict(seed=48, diff_rank_seed=True)
