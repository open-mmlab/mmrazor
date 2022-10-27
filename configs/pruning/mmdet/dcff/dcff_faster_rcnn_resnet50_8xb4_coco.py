_base_ = [
    './dcff_faster_rcnn_resnet50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_2x.py',
    'mmdet::_base_/default_runtime.py'
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[60, 80, 95],
    gamma=0.1,
    _delete_=True)
train_cfg = dict(max_epochs=120, val_interval=1)

# !dataset config
# ==========================================================================
# data preprocessor

model = dict(
    _scope_='mmrazor',
    type='DCFF',
    architecture=_base_.architecture,
    fuse_freq=1,
    mutator=dict(
        type='DCFFChannelMutator',
        channel_unit_cfg=dict(
            type='DCFFChannelUnit',
            units='configs/pruning/mmdet/dcff/resnet_det.json'),
        parse_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='TwoStageDetectorPseudoLoss'))))

model_wrapper = dict(
    type='mmcv.MMDistributedDataParallel', find_unused_parameters=True)

val_cfg = dict(_delete_=True)
