_STAGE_MUTABLE = dict(
    type='OneShotOP',
    candidate_ops=dict(
        shuffle_3x3=dict(
            type='ShuffleBlock', kernel_size=3, norm_cfg=dict(type='BN')),
        shuffle_5x5=dict(
            type='ShuffleBlock', kernel_size=5, norm_cfg=dict(type='BN')),
        shuffle_7x7=dict(
            type='ShuffleBlock', kernel_size=7, norm_cfg=dict(type='BN')),
        shuffle_xception=dict(
            type='ShuffleXception', norm_cfg=dict(type='BN')),
    ))

arch_setting = [
    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, mutable_cfg.
    [64, 4, _STAGE_MUTABLE],
    [160, 4, _STAGE_MUTABLE],
    [320, 8, _STAGE_MUTABLE],
    [640, 4, _STAGE_MUTABLE],
]
