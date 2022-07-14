_base_ = [
    '../../_base_/datasets/mmcls/cifar10_bs16.py',
    '../../_base_/mmcls_runtime.py'
]

model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='DartsBackbone',
        in_channels=3,
        base_channels=16,
        num_layers=8,
        num_nodes=4,
        stem_multiplier=3,
        out_indices=(7, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=256,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        cal_acc=True),
)

algorithm = dict(
    type='Darts',
    architecture=dict(type='MMClsArchitecture', model=model),
    mutator=dict(
        type='DartsMutator',
        placeholder_mapping=dict(
            node=dict(
                type='DifferentiableOP',
                with_arch_param=True,
                choices=dict(
                    zero=dict(type='DartsZero'),
                    skip_connect=dict(
                        type='DartsSkipConnect',
                        norm_cfg=dict(type='BN', affine=False)),
                    max_pool_3x3=dict(
                        type='DartsPoolBN',
                        pool_type='max',
                        norm_cfg=dict(type='BN', affine=False)),
                    avg_pool_3x3=dict(
                        type='DartsPoolBN',
                        pool_type='avg',
                        norm_cfg=dict(type='BN', affine=False)),
                    sep_conv_3x3=dict(
                        type='DartsSepConv',
                        kernel_size=3,
                        norm_cfg=dict(type='BN', affine=False)),
                    sep_conv_5x5=dict(
                        type='DartsSepConv',
                        kernel_size=5,
                        norm_cfg=dict(type='BN', affine=False)),
                    dil_conv_3x3=dict(
                        type='DartsDilConv',
                        kernel_size=3,
                        norm_cfg=dict(type='BN', affine=False)),
                    dil_conv_5x5=dict(
                        type='DartsDilConv',
                        kernel_size=5,
                        norm_cfg=dict(type='BN', affine=False)),
                )),
            node_edge=dict(
                type='DifferentiableEdge',
                num_chosen=2,
                with_arch_param=False,
            )),
    ),
    retraining=False,
    unroll=True)

data = dict(samples_per_gpu=64, workers_per_gpu=8, split=True)

# optimizer
optimizer = dict(
    architecture=dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=3e-4),
    mutator=dict(type='Adam', lr=3e-4, weight_decay=1e-3))
optimizer_config = None

# learning policy
lr_config = dict(
    policy='CosineAnnealing', min_lr=1e-3, freeze_optimizers=['mutator'])
runner = dict(type='MultiLoaderEpochBasedRunner', max_epochs=50)

custom_hooks = [dict(
    type='SearchSubnetHook',
    interval=1,
    priority=70,
)]

find_unused_parameter = True
use_ddp_wrapper = True
