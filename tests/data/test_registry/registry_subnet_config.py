# Copyright (c) OpenMMLab. All rights reserved.
supernet = dict(
    type='MockModel',
)

model = dict(
    type='MockAlgorithm',
    architecture=supernet,
    _fix_subnet_ = dict(modules={
            'architecture.mutable1': 'conv1',
            'architecture.mutable2': 'conv2',
        })
)
