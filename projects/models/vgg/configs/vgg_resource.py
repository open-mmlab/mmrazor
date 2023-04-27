_base_ = ['./vgg_pretrain.py']

target_pruning_ratio = {
    'backbone.features.conv0_(0, 64)_64': 22,
    'backbone.features.conv1_(0, 64)_64': 43,
    'backbone.features.conv3_(0, 128)_128': 85,
    'backbone.features.conv4_(0, 128)_128': 104,
    'backbone.features.conv6_(0, 256)_256': 201,
    'backbone.features.conv7_(0, 256)_256': 166,
    'backbone.features.conv8_(0, 256)_256': 144,
    'backbone.features.conv10_(0, 512)_512': 147,
    'backbone.features.conv11_(0, 512)_512': 88,
    'backbone.features.conv12_(0, 512)_512': 80,
    'backbone.features.conv14_(0, 512)_512': 146,
    'backbone.features.conv15_(0, 512)_512': 179
}
data_preprocessor = {'type': 'mmcls.ClsDataPreprocessor'}
architecture = _base_.model

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='PruneWrapper',
    architecture=architecture,
    target_pruning_ratio=target_pruning_ratio,
)
