se_cfg = dict(
    ratio=4,
    divisor=1,
    act_cfg=(dict(type='HSwish'),
             dict(
                 type='HSigmoid', bias=3, divisor=6, min_value=0,
                 max_value=1)))

_FIRST_STAGE_MUTABLE = dict(
    type='OneShotMutableOP',
    candidates=dict(
        mb_k3e4_se=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=4,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HSwish'))))

_OTHER_STAGE_MUTABLE = dict(
    type='OneShotMutableOP',
    candidates=dict(
        mb_k3e4_se=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=4,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HSwish')),
        mb_k3e6_se=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=6,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HSwish')),
        mb_k5e4_se=dict(
            type='MBBlock',
            kernel_size=5,
            expand_ratio=4,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HSwish')),
        mb_k5e6_se=dict(
            type='MBBlock',
            kernel_size=5,
            expand_ratio=6,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HSwish')),
        mb_k7e4_se=dict(
            type='MBBlock',
            kernel_size=7,
            expand_ratio=4,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HSwish')),
        mb_k7e6_se=dict(
            type='MBBlock',
            kernel_size=7,
            expand_ratio=6,
            se_cfg=se_cfg,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='HSwish'))))

arch_setting = [
    # Parameters to build layers. 4 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, stride, mutable cfg.
    [16, 1, 1, _FIRST_STAGE_MUTABLE],
    [24, 1, 2, _OTHER_STAGE_MUTABLE],
    [40, 2, 2, _OTHER_STAGE_MUTABLE],
    [80, 3, 2, _OTHER_STAGE_MUTABLE],
    [96, 4, 1, _OTHER_STAGE_MUTABLE],
    [192, 3, 2, _OTHER_STAGE_MUTABLE],
    [320, 1, 1, _OTHER_STAGE_MUTABLE]
]
