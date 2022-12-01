# search space
arch_setting = dict(
    kernel_size=[  # [min_kernel_size, max_kernel_size, step]
        [3, 3, 1],
        [3, 7, 2],
        [3, 7, 2],
        [3, 7, 2],
        [3, 7, 2],
        [3, 7, 2],
    ],
    num_blocks=[  # [min_num_blocks, max_num_blocks, step]
        [1, 1, 1],
        [2, 4, 1],
        [2, 4, 1],
        [2, 4, 1],
        [2, 4, 1],
        [2, 4, 1],
    ],
    expand_ratio=[  # [min_expand_ratio, max_expand_ratio, step]
        [1, 1, 1],
        [3, 6, 1],
        [3, 6, 1],
        [3, 6, 1],
        [3, 6, 1],
        [3, 6, 1],
        [6, 6, 1],  # last layer
    ],
    # [16, 16, 24, 40, 80, 112, 160, 960, 1280]
    num_out_channels=[  # [min_channel, max_channel, step]
        [16, 16, 8],  # first layer
        [16, 16, 8],
        [16, 24, 8],
        [24, 40, 8],
        [40, 80, 8],
        [80, 112, 8],
        [112, 160, 8],
        [1024, 1280, 1280 - 1024],  # last layer
    ])

nas_backbone = dict(
    type='mmrazor.AttentiveMobileNetV3',
    arch_setting=arch_setting,
    out_indices=(6, ),
    stride_list=[1, 2, 2, 2, 1, 2],
    act_list=['ReLU', 'ReLU', 'ReLU', 'Swish', 'Swish', 'Swish'],
    with_se_list=[False, False, True, False, True, True],
    conv_cfg=dict(type='OFAConv2d'),
    norm_cfg=dict(type='mmrazor.DynamicBatchNorm2d', momentum=0.0),
    with_attentive_shortcut=False)
