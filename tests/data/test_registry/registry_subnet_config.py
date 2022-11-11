# Copyright (c) OpenMMLab. All rights reserved.
supernet = dict(
    type='MockModel',
)

model = dict(
    type='MockAlgorithm',
    architecture=supernet,
    _fix_subnet_ = {
            'architecture.mutable1': {'chosen':'conv1'},
            'architecture.mutable2': {'chosen':'conv2'},
        }
)
