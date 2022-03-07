_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs128_colorjittor.py',
    '../../_base_/schedules/mmcls/imagenet_bs1024_spos.py',
    '../../_base_/mmcls_runtime.py'
]
norm_cfg = dict(type='BN')
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='SearchableMobileNet', widen_factor=1.0, norm_cfg=norm_cfg),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1280,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5),
    ),
)

mutator = dict(
    type='OneShotMutator',
    placeholder_mapping=dict(
        searchable_blocks=dict(
            type='OneShotOP',
            choices=dict(
                mbv2_k3e3=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k5e3=dict(
                    type='MBBlock',
                    kernel_size=5,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k7e3=dict(
                    type='MBBlock',
                    kernel_size=7,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k3e6=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k5e6=dict(
                    type='MBBlock',
                    kernel_size=5,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k7e6=dict(
                    type='MBBlock',
                    kernel_size=7,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                identity=dict(type='Identity'))),
        first_blocks=dict(
            type='OneShotOP',
            choices=dict(
                mbv2_k3e1=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=1,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')), ))))

algorithm = dict(
    type='SPOS',
    architecture=dict(
        type='MMClsArchitecture',
        model=model,
    ),
    mutator=mutator,
    distiller=None,
    retraining=False,
)

runner = dict(max_iters=150000)
evaluation = dict(interval=1000, metric='accuracy')

# checkpoint saving
checkpoint_config = dict(interval=1000)

find_unused_parameters = True
