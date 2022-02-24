_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs128_colorjittor_greedynas.py',
    '../../_base_/schedules/mmcls/imagenet_bs1024_spos.py',
    '../../_base_/mmcls_runtime.py'
]

data = dict()

norm_cfg = dict(type='BN')
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='SearchableMobileNetV2', widen_factor=1.0, norm_cfg=norm_cfg),
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
        all_blocks=dict(
            type='OneShotOP',
            choices=dict(
                mbv2_k3e3=dict(
                    type='MBV2Block',
                    kernel_size=3,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k5e3=dict(
                    type='MBV2Block',
                    kernel_size=5,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k7e3=dict(
                    type='MBV2Block',
                    kernel_size=7,
                    expand_ratio=3,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k3e6=dict(
                    type='MBV2Block',
                    kernel_size=3,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k5e6=dict(
                    type='MBV2Block',
                    kernel_size=5,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k7e6=dict(
                    type='MBV2Block',
                    kernel_size=7,
                    expand_ratio=6,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k3e3_se=dict(
                    type='MBV2Block',
                    kernel_size=3,
                    expand_ratio=3,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k5e3_se=dict(
                    type='MBV2Block',
                    kernel_size=5,
                    expand_ratio=3,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k7e3_se=dict(
                    type='MBV2Block',
                    kernel_size=7,
                    expand_ratio=3,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k3e6_se=dict(
                    type='MBV2Block',
                    kernel_size=3,
                    expand_ratio=6,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k5e6_se=dict(
                    type='MBV2Block',
                    kernel_size=5,
                    expand_ratio=6,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                mbv2_k7e6_se=dict(
                    type='MBV2Block',
                    kernel_size=7,
                    expand_ratio=6,
                    se_cfg=se_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type='ReLU6')),
                identity=dict(type='Identity')))))

algorithm = dict(
    type='GreedyNAS',
    architecture=dict(
        type='MMClsArchitecture',
        model=model,
    ),
    mutator=mutator,
    distiller=None,
    retraining=False,
)

sampler = dict(
    score_key='accuracy_top-1',
    constraints=dict(flops=330 * 1e6),
    pool_size=1000,
    sample_num=10,
    top_k=5,
    p_stg='linear',
    start_iter=10000,
    max_iter=144360,
    init_p=0.,
    max_p=0.8)

workflow = [('train', 1)]
runner = dict(max_iters=150000)
evaluation = dict(interval=1000, metric='accuracy')

# checkpoint saving
checkpoint_config = dict(interval=10000, max_keep_ckpts=10)

find_unused_parameters = True
