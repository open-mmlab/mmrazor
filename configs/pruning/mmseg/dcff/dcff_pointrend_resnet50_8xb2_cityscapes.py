_base_ = [
    # TODO: use autoaug pipeline.
    # '../../../_base_/datasets/mmseg/cityscapes.py',
    'mmseg::_base_/datasets/cityscapes.py',
    'mmseg::_base_/schedules/schedule_160k.py',
    'mmseg::_base_/default_runtime.py',
    './pointrend_resnet50.py'
]

custom_imports = dict(imports=[
    'mmseg.models',
])

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0001),
    clip_grad=dict(max_norm=25, norm_type=2),
    _delete_=True)
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=800)

architecture = _base_.architecture

# model settings
model = dict(
    _scope_='mmrazor',
    type='DCFF',
    channel_cfgs='/mnt/lustre/zengyi.vendor/mmrazor/group_pr/mmrazor/configs/pruning/mmseg/dcff/resnet_seg.json',
    #architecture=dict(
    #    cfg_path='mmseg::_base_/models/pointrend_r50.py', pretrained=False),
    architecture=architecture,
    mutator=dict(
        type='DCFFChannelMutator',
        channl_group_cfg=dict(
            type='DCFFChannelGroup',
            candidate_choices=[32],
            candidate_mode='number'),
        tracer_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='ImageSegPseudoLossGPU'))))

find_unused_parameters = True

model_wrapper = dict(
    type='mmcv.MMDistributedDataParallel', find_unused_parameters=True)

custom_hooks = [
    dict(
        type='DCFFHook',
        by_epoch=False,
        dcff_count=200)
]

val_cfg = dict(_delete_=True)
