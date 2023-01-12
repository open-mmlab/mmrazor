# Copyright (c) OpenMMLab. All rights reserved.
supernet = dict(
    type='mmrazor.sub_model',
    cfg=dict(
        type='MockModel',
    ),
    fix_subnet = {
            'backbone.mutable1': {'chosen':'conv1'},
            'backbone.mutable2': {'chosen':'conv2'},
        },
    extra_prefix='backbone.'
)

model = dict(
    type='MockAlgorithm',
    architecture=supernet
)
