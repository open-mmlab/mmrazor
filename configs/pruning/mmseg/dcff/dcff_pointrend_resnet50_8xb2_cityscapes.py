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

stage_ratio_1 = 0.65
stage_ratio_2 = 0.6
stage_ratio_3 = 0.9
stage_ratio_4 = 0.7

# the config template of target_pruning_ratio can be got by
# python ./tools/pruning/get_channel_units.py {config_file} --choice
target_pruning_ratio = {
    'backbone.layer1.0.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.0.conv2_(0, 64)_64': stage_ratio_2,
    'backbone.layer1.0.conv3_(0, 256)_256': stage_ratio_3,
    'backbone.layer1.1.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.1.conv2_(0, 64)_64': stage_ratio_2,
    'backbone.layer1.2.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.2.conv2_(0, 64)_64': stage_ratio_2,
    # block 1 [0.8, 0.8] downsample=[0.9]
    'backbone.layer2.0.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.0.conv2_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.0.conv3_(0, 512)_512': stage_ratio_3,
    'backbone.layer2.1.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.1.conv2_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.2.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.2.conv2_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.3.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.3.conv2_(0, 128)_128': stage_ratio_2,
    # block 2 [0.8, 0.8] downsample=[0.9]
    'backbone.layer3.0.conv1_(0, 256)_256': stage_ratio_1,
    'backbone.layer3.0.conv2_(0, 256)_256': stage_ratio_2,
    'backbone.layer3.0.conv3_(0, 1024)_1024': stage_ratio_3,
    'backbone.layer3.1.conv1_(0, 256)_256': stage_ratio_1,
    'backbone.layer3.1.conv2_(0, 256)_256': stage_ratio_2,
    'backbone.layer3.2.conv1_(0, 256)_256': stage_ratio_1,
    'backbone.layer3.2.conv2_(0, 256)_256': stage_ratio_2,
    'backbone.layer3.3.conv1_(0, 256)_256': stage_ratio_4,
    'backbone.layer3.3.conv2_(0, 256)_256': stage_ratio_4,
    'backbone.layer3.4.conv1_(0, 256)_256': stage_ratio_4,
    'backbone.layer3.4.conv2_(0, 256)_256': stage_ratio_4,
    'backbone.layer3.5.conv1_(0, 256)_256': stage_ratio_4,
    'backbone.layer3.5.conv2_(0, 256)_256': stage_ratio_4,
    # block 3 [0.8, 0.8]*2+[0.8, 0.85]*2 downsample=[0.9]
    'backbone.layer4.0.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.0.conv2_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.0.conv3_(0, 2048)_2048': stage_ratio_3,
    'backbone.layer4.1.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.1.conv2_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.2.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.2.conv2_(0, 512)_512': stage_ratio_4
    # block 4 [0.85, 0.85] downsample=[0.9]
}

# model settings
model = dict(
    _scope_='mmrazor',
    type='DCFF',
    architecture=_base_.architecture,
    mutator_cfg=dict(
        type='DCFFChannelMutator',
        channel_unit_cfg=dict(
            type='DCFFChannelUnit', default_args=dict(choice_mode='ratio')),
        parse_cfg=dict(
            type='ChannelAnalyzer',
            demo_input=(1, 3, 224, 224),
            tracer_type='BackwardTracer')),
    target_pruning_ratio=target_pruning_ratio,
    step_freq=200,
    linear_schedule=False)

model_wrapper = dict(
    type='mmcv.MMDistributedDataParallel', find_unused_parameters=True)

val_cfg = dict(_delete_=True, type='mmrazor.ItePruneValLoop')
