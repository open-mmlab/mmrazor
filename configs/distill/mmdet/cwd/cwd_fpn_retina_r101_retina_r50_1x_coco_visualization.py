_base_ = ['./cwd_fpn_retina_r101_retina_r50_1x_coco.py']

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=-1),
    visualization=dict(
        _scope_='mmrazor',
        type='RazorVisualizationHook',
        enabled=True,
        recorders=dict(
            # todo: Maybe it is hard for users to understand why to add a
            #  prefix `architecture.`
            neck=dict(
                _scope_='mmrazor',
                type='ModuleOutputs',
                source='architecture.neck')),
        mappings=dict(
            p3=dict(recorder='neck', data_idx=0),
            p4=dict(recorder='neck', data_idx=1),
            p5=dict(recorder='neck', data_idx=2),
            p6=dict(recorder='neck', data_idx=3)),
        out_dir='retina_vis'))
