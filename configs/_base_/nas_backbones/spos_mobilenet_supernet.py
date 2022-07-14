_STAGE_MUTABLE = dict(
    _scope_='mmrazor',
    type='OneShotMutableOP',
    candidates=dict(
        mb_k3e3=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=3,
            act_cfg=dict(type='ReLU6')),
        mb_k5e3=dict(
            type='MBBlock',
            kernel_size=5,
            expand_ratio=3,
            act_cfg=dict(type='ReLU6')),
        mb_k7e3=dict(
            type='MBBlock',
            kernel_size=7,
            expand_ratio=3,
            act_cfg=dict(type='ReLU6')),
        mb_k3e6=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=6,
            act_cfg=dict(type='ReLU6')),
        mb_k5e6=dict(
            type='MBBlock',
            kernel_size=5,
            expand_ratio=6,
            act_cfg=dict(type='ReLU6')),
        mb_k7e6=dict(
            type='MBBlock',
            kernel_size=7,
            expand_ratio=6,
            act_cfg=dict(type='ReLU6')),
        identity=dict(type='Identity')))

_FIRST_MUTABLE = dict(
    _scope_='mmrazor',
    type='OneShotMutableOP',
    candidates=dict(
        mb_k3e1=dict(
            type='MBBlock',
            kernel_size=3,
            expand_ratio=1,
            act_cfg=dict(type='ReLU6'))))

arch_setting = [
    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, mutable_cfg.
    [24, 1, 1, _FIRST_MUTABLE],
    [32, 4, 2, _STAGE_MUTABLE],
    [56, 4, 2, _STAGE_MUTABLE],
    [112, 4, 2, _STAGE_MUTABLE],
    [128, 4, 1, _STAGE_MUTABLE],
    [256, 4, 2, _STAGE_MUTABLE],
    [432, 1, 1, _STAGE_MUTABLE]
]

nas_backbone = dict(
    _scope_='mmrazor',
    type='SearchableMobileNet',
    first_channels=40,
    last_channels=1728,
    widen_factor=1.0,
    arch_setting=arch_setting)
