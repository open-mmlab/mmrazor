_base_ = [
    'mmcls::_base_/datasets/imagenet_bs32.py',
    'mmcls::_base_/schedules/imagenet_bs256.py',
    'mmcls::_base_/default_runtime.py'
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)

# !dataset config
# ==========================================================================
# data preprocessor
data_preprocessor = dict(
    type='ImgDataPreprocessor',
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    bgr_to_rgb=True,
)

# !autoslim algorithm config
model = dict(
    _scope_='mmrazor',
    type='DCFF',
    data_preprocessor=data_preprocessor,
    channel_cfgs='./resnet_cls.json',
    architecture=dict(
        cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py', pretrained=False),
    mutator=dict(
        type='DCFFChannelMutator',
        channl_group_cfg=dict(
            type='DCFFChannelGroup',
            candidate_choices=[32],
            candidate_mode='number'),
        tracer_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='ImageClassifierPseudoLoss'))))

find_unused_parameters = True

model_wrapper = dict(
    type='mmcv.MMDistributedDataParallel', find_unused_parameters=True)

custom_hooks = [dict(type='DCFFHook')]

val_cfg = dict(_delete_=True)
