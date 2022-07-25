mutable_cfg = dict(
    type='mmrazor.DiffMutableOP',
    candidates=dict(
        zero=dict(type='mmrazor.DartsZero'),
        skip_connect=dict(type='mmrazor.DartsSkipConnect', use_drop_path=True),
        max_pool_3x3=dict(
            type='mmrazor.DartsPoolBN', pool_type='max', use_drop_path=True),
        avg_pool_3x3=dict(
            type='mmrazor.DartsPoolBN', pool_type='avg', use_drop_path=True),
        sep_conv_3x3=dict(
            type='mmrazor.DartsSepConv', kernel_size=3, use_drop_path=True),
        sep_conv_5x5=dict(
            type='mmrazor.DartsSepConv', kernel_size=5, use_drop_path=True),
        dil_conv_3x3=dict(
            type='mmrazor.DartsDilConv', kernel_size=3, use_drop_path=True),
        dil_conv_5x5=dict(
            type='mmrazor.DartsDilConv', kernel_size=5, use_drop_path=True)))

route_cfg = dict(type='mmrazor.DiffChoiceRoute', with_arch_param=True)

nas_backbone = dict(
    type='mmrazor.DartsBackbone',
    in_channels=3,
    base_channels=16,
    num_layers=8,
    num_nodes=4,
    stem_multiplier=3,
    out_indices=(7, ),
    mutable_cfg=mutable_cfg,
    route_cfg=route_cfg,
    norm_cfg=dict(type='BN', affine=False))
