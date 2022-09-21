student_recorders = dict(
    backbone=dict(_scope_='mmrazor', type='ModuleOutputs', source='backbone'))
student_mappings = dict(
    p3=dict(recorder='backbone', data_idx=0),
    p4=dict(recorder='backbone', data_idx=1),
    p5=dict(recorder='backbone', data_idx=2),
    p6=dict(recorder='backbone', data_idx=3))
teacher_recorders = dict(
    backbone=dict(_scope_='mmrazor', type='ModuleOutputs', source='backbone'))
teacher_mappings = dict(
    p3=dict(recorder='backbone', data_idx=0),
    p4=dict(recorder='backbone', data_idx=1),
    p5=dict(recorder='backbone', data_idx=2),
    p6=dict(recorder='backbone', data_idx=3))
