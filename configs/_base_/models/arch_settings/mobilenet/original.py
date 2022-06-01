_FIRST_STAGE_MUTABLE = dict(
    type='OneShotOP',
    candidate_ops=dict(
        mb_k3e1=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6'))))

_OTHER_STAGE_MUTABLE = dict(
    type='OneShotOP',
    candidate_ops=dict(
        mb_k3e3=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=3,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6')),
        mb_k5e3=dict(
            type='MBBlock',
            kernel_size=5,
            expand_ratio=3,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6')),
        mb_k7e3=dict(
            type='MBBlock',
            kernel_size=7,
            expand_ratio=3,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6')),
        mb_k3e6=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=6,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6')),
        mb_k5e6=dict(
            type='MBBlock',
            kernel_size=5,
            expand_ratio=6,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6')),
        mb_k7e6=dict(
            type='MBBlock',
            kernel_size=7,
            expand_ratio=6,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU6')),
        identity=dict(type='Identity')))

arch_setting = [
    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, stride, mutable cfg.
    [16, 1, 1, _FIRST_STAGE_MUTABLE],
    [24, 2, 2, _OTHER_STAGE_MUTABLE],
    [32, 3, 2, _OTHER_STAGE_MUTABLE],
    [64, 4, 2, _OTHER_STAGE_MUTABLE],
    [96, 3, 1, _OTHER_STAGE_MUTABLE],
    [160, 3, 2, _OTHER_STAGE_MUTABLE],
    [320, 1, 1, _OTHER_STAGE_MUTABLE]
]
