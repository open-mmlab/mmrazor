# search space
arch_setting = dict(
    kernel_size=[  # [min_kernel_size, max_kernel_size, step]
        [3, 5, 2],
        [3, 5, 2],
        [3, 5, 2],
        [3, 5, 2],
        [3, 5, 2],
        [3, 5, 2],
        [3, 5, 2],
    ],
    num_blocks=[  # [min_num_blocks, max_num_blocks, step]
        [1, 2, 1],
        [3, 5, 1],
        [3, 6, 1],
        [3, 6, 1],
        [3, 8, 1],
        [3, 8, 1],
        [1, 2, 1],
    ],
    expand_ratio=[  # [min_expand_ratio, max_expand_ratio, step]
        [1, 1, 1],
        [4, 6, 1],
        [4, 6, 1],
        [4, 6, 1],
        [4, 6, 1],
        [6, 6, 1],
        [6, 6, 1],
        [6, 6, 1],  # last layer
    ],
    num_out_channels=[  # [min_channel, max_channel, step]
        [16, 24, 8],  # first layer
        [16, 24, 8],
        [24, 32, 8],
        [32, 40, 8],
        [64, 72, 8],
        [112, 128, 8],
        [192, 216, 8],
        [216, 224, 8],
        [1792, 1984, 1984 - 1792],  # last layer
    ])

input_resizer_cfg = dict(
    input_sizes=[[192, 192], [224, 224], [256, 256], [288, 288]])

nas_backbone = dict(
    type='AttentiveMobileNetV3',
    arch_setting=arch_setting,
    norm_cfg=dict(type='DynamicBatchNorm2d', momentum=0.0))
