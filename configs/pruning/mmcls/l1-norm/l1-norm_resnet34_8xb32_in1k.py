_base_ = ['mmcls::resnet/resnet34_8xb32_in1k.py']

stage_ratio_1 = 0.7
stage_ratio_2 = 0.7
stage_ratio_3 = 0.7
stage_ratio_4 = 1.0

# the config template of target_pruning_ratio can be got by
# python ./tools/get_channel_units.py {config_file} --choice
target_pruning_ratio = {
    'backbone.layer1.2.conv2_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.0.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.1.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer1.2.conv1_(0, 64)_64': stage_ratio_1,
    'backbone.layer2.0.conv1_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.3.conv2_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.1.conv1_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.2.conv1_(0, 128)_128': stage_ratio_2,
    'backbone.layer2.3.conv1_(0, 128)_128': stage_ratio_2,
    'backbone.layer3.0.conv1_(0, 256)_256': stage_ratio_3,
    'backbone.layer3.5.conv2_(0, 256)_256': stage_ratio_3,
    'backbone.layer3.1.conv1_(0, 256)_256': stage_ratio_3,
    'backbone.layer3.2.conv1_(0, 256)_256': stage_ratio_3,
    'backbone.layer3.3.conv1_(0, 256)_256': stage_ratio_3,
    'backbone.layer3.4.conv1_(0, 256)_256': stage_ratio_3,
    'backbone.layer3.5.conv1_(0, 256)_256': stage_ratio_3,
    'backbone.layer4.0.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.2.conv2_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.1.conv1_(0, 512)_512': stage_ratio_4,
    'backbone.layer4.2.conv1_(0, 512)_512': stage_ratio_4
}
data_preprocessor = {'type': 'mmcls.ClsDataPreprocessor'}
architecture = _base_.model
architecture.update({
    'init_cfg': {
        'type':
        'Pretrained',
        'checkpoint':
        'https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth'  # noqa
    }
})

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='ItePruneAlgorithm',
    architecture=architecture,
    mutator_cfg=dict(
        type='ChannelMutator',
        channel_unit_cfg=dict(
            type='L1MutableChannelUnit',
            default_args=dict(choice_mode='ratio'))),
    target_pruning_ratio=target_pruning_ratio,
    step_freq=1,
)
