norm_cfg = dict(type='BN', eps=0.01)

_STAGE_MUTABLE = dict(
    type='mmrazor.OneHotMutableOP',
    fix_threshold=0.3,
    candidates=dict(
        shuffle_3x3=dict(
            type='ShuffleBlock', kernel_size=3, norm_cfg=norm_cfg),
        shuffle_5x5=dict(
            type='ShuffleBlock', kernel_size=5, norm_cfg=norm_cfg),
        shuffle_7x7=dict(
            type='ShuffleBlock', kernel_size=7, norm_cfg=norm_cfg),
        shuffle_xception=dict(type='ShuffleXception', norm_cfg=norm_cfg)))

arch_setting = [
    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, mutable_cfg.
    [64, 4, _STAGE_MUTABLE],
    [160, 4, _STAGE_MUTABLE],
    [320, 8, _STAGE_MUTABLE],
    [640, 4, _STAGE_MUTABLE]
]

nas_backbone = dict(
    type='mmrazor.SearchableShuffleNetV2',
    widen_factor=1.0,
    arch_setting=arch_setting,
    norm_cfg=norm_cfg)
