_base_ = [
    # TODO: use autoaug pipeline.
    'mmseg::_base_/datasets/cityscapes.py',
    'mmseg::_base_/schedules/schedule_160k.py',
    'mmseg::_base_/default_runtime.py',
    './pointrend_resnet50.py'
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=25, norm_type=2),
    _delete_=True)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=800)

param_scheduler = [
    # warm up
    dict(type='LinearLR', by_epoch=False, start_factor=0.1, begin=0, end=200),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=200,
        end=80000,
        by_epoch=False,
    )
]

# model settings
model = dict(
    _scope_='mmrazor',
    type='DCFF',
    channel_cfgs='./resnet_seg.json',
    architecture=_base_.architecture,
    mutator=dict(
        type='DCFFChannelMutator',
        channl_group_cfg=dict(
            type='DCFFChannelGroup',
            candidate_choices=[32],
            candidate_mode='number'),
        tracer_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='CascadeEncoderDecoderPseudoLoss'))))

find_unused_parameters = True

model_wrapper = dict(
    type='mmcv.MMDistributedDataParallel', find_unused_parameters=True)

custom_hooks = [dict(type='DCFFHook', by_epoch=False, dcff_count=200)]
