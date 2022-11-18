_base_ = [
    'mmcls::_base_/datasets/imagenet_bs32.py',
    'mmcls::_base_/schedules/imagenet_bs256.py',
    'mmcls::_base_/default_runtime.py'
]

stage_ratio_1 = 0.65
stage_ratio_2 = 0.6
stage_ratio_3 = 0.9
stage_ratio_4 = 0.7

# the config template of target_pruning_ratio can be got by
# python ./tools/get_channel_units.py {config_file} --choice
target_pruning_ratio = {
    'backbone.layer1.0.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.0.conv2_(0, 64)_64': stage_ratio_2,
    'backbone.layer1.0.conv3_(0, 256)_256': stage_ratio_3,
    'backbone.layer1.1.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.1.conv2_(0, 64)_64': stage_ratio_2,
    'backbone.layer1.2.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.2.conv2_(0, 64)_64': stage_ratio_2,
    # block 1 [0.65, 0.6] downsample=[0.9]
    'backbone.layer2.0.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.0.conv2_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.0.conv3_(0, 512)_512': stage_ratio_3,
    'backbone.layer2.1.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.1.conv2_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.2.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.2.conv2_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.3.conv1_(0, 128)_128': stage_ratio_1,
    'backbone.layer2.3.conv2_(0, 128)_128': stage_ratio_2,
    # block 2 [0.65, 0.6] downsample=[0.9]
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
    # block 3 [0.65, 0.6]*2+[0.7, 0.7]*2 downsample=[0.9]
    'backbone.layer4.0.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.0.conv2_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.0.conv3_(0, 2048)_2048': stage_ratio_3,
    'backbone.layer4.1.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.1.conv2_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.2.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.2.conv2_(0, 512)_512': stage_ratio_4
    # block 4 [0.7, 0.7] downsample=[0.9]
}

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)
train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)

data_preprocessor = {'type': 'mmcls.ClsDataPreprocessor'}

# model settings
model = dict(
    _scope_='mmrazor',
    type='DCFF',
    architecture=dict(
        cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py', pretrained=False),
    mutator_cfg=dict(
        type='DCFFChannelMutator',
        channel_unit_cfg=dict(
            type='DCFFChannelUnit', default_args=dict(choice_mode='ratio')),
        parse_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='ImageClassifierPseudoLoss'))),
    target_pruning_ratio=target_pruning_ratio,
    step_freq=1,
    linear_schedule=False,
    is_deployed=False)
