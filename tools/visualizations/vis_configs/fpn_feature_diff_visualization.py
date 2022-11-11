# Copyright (c) OpenMMLab. All rights reserved.
# configs for the 1st model
recorders1 = dict(
    neck=dict(_scope_='mmrazor', type='ModuleOutputs', source='neck'))
mappings1 = dict(
    p3=dict(recorder='neck', data_idx=0),
    p4=dict(recorder='neck', data_idx=1),
    p5=dict(recorder='neck', data_idx=2),
    p6=dict(recorder='neck', data_idx=3))

# configs for the 2nd model
recorders2 = dict(
    neck=dict(_scope_='mmrazor', type='ModuleOutputs', source='neck'))
mappings2 = dict(
    p3=dict(recorder='neck', data_idx=0),
    p4=dict(recorder='neck', data_idx=1),
    p5=dict(recorder='neck', data_idx=2),
    p6=dict(recorder='neck', data_idx=3))
