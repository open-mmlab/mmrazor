# Copyright (c) OpenMMLab. All rights reserved.
from platform import architecture


supernet = dict(
    type='MockModel',
)

model = dict(
    type='MockAlgorithm',
    architecture=supernet,
    _return_architecture_ = True,
)

