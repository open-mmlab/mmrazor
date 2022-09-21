student_recorders = dict(
    neck=dict(_scope_='mmrazor', type='ModuleOutputs', source='neck'))
student_mappings = dict(
    p3=dict(recorder='neck', data_idx=0),
    p4=dict(recorder='neck', data_idx=1),
    p5=dict(recorder='neck', data_idx=2),
    p6=dict(recorder='neck', data_idx=3))
teacher_recorders = dict(
    neck=dict(_scope_='mmrazor', type='ModuleOutputs', source='neck'))
teacher_mappings = dict(
    p3=dict(recorder='neck', data_idx=0),
    p4=dict(recorder='neck', data_idx=1),
    p5=dict(recorder='neck', data_idx=2),
    p6=dict(recorder='neck', data_idx=3))
