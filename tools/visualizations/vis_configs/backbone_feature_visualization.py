# Copyright (c) OpenMMLab. All rights reserved.
recorders = dict(
    backbone=dict(_scope_='mmrazor', type='ModuleOutputs', source='backbone'))
mappings = dict(
    p3=dict(recorder='backbone', data_idx=0),
    p4=dict(recorder='backbone', data_idx=1),
    p5=dict(recorder='backbone', data_idx=2),
    p6=dict(recorder='backbone', data_idx=3))
