_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs128_colorjittor_aa_greedynas.py',
    '../../_base_/schedules/mmcls/imagenet_bs1024_aa_greedynas.py',
    '../../_base_/mmcls_runtime.py'
]

data = dict(samples_per_gpu=96)
optimizer = dict(lr=0.048)
lr_config = dict(
    step=int(1666 * 2.4), warmup_iters=1666 * 3, warmup_ratio=1e-6 / 0.048)
runner = dict(max_iters=450 * 1666)

norm_cfg = dict(type='BN')
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='SearchableMobileNet',
        arch_setting_type='greedynas',
        widen_factor=1.0,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU')),
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

se_cfg = dict(
    ratio=4,
    act_cfg=(dict(type='HSwish'),
             dict(
                 type='HSigmoid', bias=3, divisor=6, min_value=0,
                 max_value=1)))
mutator = dict(
    type='OneShotMutator',
    placeholder_mapping=dict(
        searchable_blocks=dict(
            type='OneShotOP',
            choices=dict(
                mb_k3e3=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                mb_k5e3=dict(
                    type='MBBlock',
                    kernel_size=5,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                mb_k7e3=dict(
                    type='MBBlock',
                    kernel_size=7,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                mb_k3e6=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                mb_k5e6=dict(
                    type='MBBlock',
                    kernel_size=5,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                mb_k7e6=dict(
                    type='MBBlock',
                    kernel_size=7,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                mb_k3e3_se=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=3,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                mb_k5e3_se=dict(
                    type='MBBlock',
                    kernel_size=5,
                    expand_ratio=3,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                mb_k7e3_se=dict(
                    type='MBBlock',
                    kernel_size=7,
                    expand_ratio=3,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                mb_k3e6_se=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=6,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                mb_k5e6_se=dict(
                    type='MBBlock',
                    kernel_size=5,
                    expand_ratio=6,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                mb_k7e6_se=dict(
                    type='MBBlock',
                    kernel_size=7,
                    expand_ratio=6,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish')),
                identity=dict(type='Identity'))),
        first_blocks=dict(
            type='OneShotOP',
            choices=dict(
                mb_k3e1=dict(
                    type='MBBlock',
                    kernel_size=3,
                    expand_ratio=1,
                    norm_cfg=norm_cfg,
                    drop_path_rate=0.2,
                    act_cfg=dict(type='HSwish'))))))

algorithm = dict(
    type='GreedyNAS',
    architecture=dict(
        type='MMClsArchitecture',
        model=model,
    ),
    mutator=mutator,
    distiller=None,
    retraining=True,
)

workflow = [('train', 1)]
evaluation = dict(interval=5000, metric='accuracy')

# checkpoint saving
checkpoint_config = dict(interval=20000, max_keep_ckpts=5)

custom_hooks = [dict(type='EMAHook', momentum=0.0001, warm_up=100)]

find_unused_parameters = False
