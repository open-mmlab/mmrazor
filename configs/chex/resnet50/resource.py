_base_ = ['mmcls::resnet/resnet50_8xb32_in1k.py']

target_pruning_ratio = {
    'backbone.conv1_(0, 64)_64': 63,
    'backbone.layer1.0.conv1_(0, 64)_64': 55,
    'backbone.layer1.0.conv2_(0, 64)_64': 56,
    'backbone.layer1.0.conv3_(0, 256)_256': 92,
    'backbone.layer1.1.conv1_(0, 64)_64': 53,
    'backbone.layer1.1.conv2_(0, 64)_64': 55,
    'backbone.layer1.2.conv1_(0, 64)_64': 49,
    'backbone.layer1.2.conv2_(0, 64)_64': 57,
    'backbone.layer2.0.conv1_(0, 128)_128': 111,
    'backbone.layer2.0.conv2_(0, 128)_128': 124,
    'backbone.layer2.0.conv3_(0, 512)_512': 58,
    'backbone.layer2.1.conv1_(0, 128)_128': 15,
    'backbone.layer2.1.conv2_(0, 128)_128': 33,
    'backbone.layer2.2.conv1_(0, 128)_128': 65,
    'backbone.layer2.2.conv2_(0, 128)_128': 99,
    'backbone.layer2.3.conv1_(0, 128)_128': 62,
    'backbone.layer2.3.conv2_(0, 128)_128': 103,
    'backbone.layer3.0.conv1_(0, 256)_256': 233,
    'backbone.layer3.0.conv2_(0, 256)_256': 250,
    'backbone.layer3.0.conv3_(0, 1024)_1024': 79,
    'backbone.layer3.1.conv1_(0, 256)_256': 46,
    'backbone.layer3.1.conv2_(0, 256)_256': 104,
    'backbone.layer3.2.conv1_(0, 256)_256': 34,
    'backbone.layer3.2.conv2_(0, 256)_256': 57,
    'backbone.layer3.3.conv1_(0, 256)_256': 83,
    'backbone.layer3.3.conv2_(0, 256)_256': 121,
    'backbone.layer3.4.conv1_(0, 256)_256': 154,
    'backbone.layer3.4.conv2_(0, 256)_256': 232,
    'backbone.layer3.5.conv1_(0, 256)_256': 228,
    'backbone.layer3.5.conv2_(0, 256)_256': 242,
    'backbone.layer4.0.conv1_(0, 512)_512': 486,
    'backbone.layer4.0.conv2_(0, 512)_512': 497,
    'backbone.layer4.0.conv3_(0, 2048)_2048': 2048,
    'backbone.layer4.1.conv1_(0, 512)_512': 507,
    'backbone.layer4.1.conv2_(0, 512)_512': 504,
    'backbone.layer4.2.conv1_(0, 512)_512': 502,
    'backbone.layer4.2.conv2_(0, 512)_512': 508
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
            default_args=dict(choice_mode='number'))),
    target_pruning_ratio=target_pruning_ratio,
    step_freq=1,
)
