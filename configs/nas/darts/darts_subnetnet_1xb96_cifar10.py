_base_ = [
    '../../_base_/datasets/mmcls/cifar10_bs96_cutout.py',
    '../../_base_/mmcls_runtime.py'
]

model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='DartsBackbone',
        in_channels=3,
        base_channels=36,
        num_layers=20,
        num_nodes=4,
        stem_multiplier=3,
        auxliary=True,
        aux_channels=128,
        aux_out_channels=768,
        out_indices=(19, )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='DartsSubnetClsHead',
        num_classes=10,
        in_channels=576,
        aux_in_channels=768,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        aux_loss=dict(type='CrossEntropyLoss', loss_weight=0.4),
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
                with_arch_param=False,
                choices=dict(
                    zero=dict(type='DartsZero'),
                    skip_connect=dict(
                        type='DartsSkipConnect', use_drop_path=True),
                    max_pool_3x3=dict(
                        type='DartsPoolBN',
                        pool_type='max',
                        use_drop_path=True),
                    avg_pool_3x3=dict(
                        type='DartsPoolBN',
                        pool_type='avg',
                        use_drop_path=True),
                    sep_conv_3x3=dict(
                        type='DartsSepConv', kernel_size=3,
                        use_drop_path=True),
                    sep_conv_5x5=dict(
                        type='DartsSepConv', kernel_size=5,
                        use_drop_path=True),
                    dil_conv_3x3=dict(
                        type='DartsDilConv', kernel_size=3,
                        use_drop_path=True),
                    dil_conv_5x5=dict(
                        type='DartsDilConv', kernel_size=5,
                        use_drop_path=True),
                )),
            node_edge=dict(
                type='DifferentiableEdge',
                num_chosen=2,
                with_arch_param=False,
            )),
    ),
    retraining=True,
    unroll=False)

data = dict(workers_per_gpu=8)

# optimizer
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0003)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0)
runner = dict(type='EpochBasedRunner', max_epochs=600)

custom_hooks = [
    dict(
        type='DropPathProbHook',
        max_prob=0.2,
        interval=1,
        priority=70,
    )
]
