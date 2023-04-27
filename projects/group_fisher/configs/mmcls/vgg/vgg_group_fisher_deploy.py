_base_ = '../../../../models/vgg/configs/vgg_pretrain.py'
custom_imports = dict(imports=['projects'])

architecture = _base_.model
architecture.update({'data_preprocessor': _base_.data_preprocessor})
data_preprocessor = {'_delete_': True}

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='PruneDeployWrapper2',
    architecture=architecture,
    mutable_cfg={
        'backbone.features.conv0_(0, 64)_64': 21,
        'backbone.features.conv1_(0, 64)_64': 42,
        'backbone.features.conv3_(0, 128)_128': 86,
        'backbone.features.conv4_(0, 128)_128': 110,
        'backbone.features.conv6_(0, 256)_256': 203,
        'backbone.features.conv7_(0, 256)_256': 170,
        'backbone.features.conv8_(0, 256)_256': 145,
        'backbone.features.conv10_(0, 512)_512': 138,
        'backbone.features.conv11_(0, 512)_512': 84,
        'backbone.features.conv12_(0, 512)_512': 54,
        'backbone.features.conv14_(0, 512)_512': 94,
        'backbone.features.conv15_(0, 512)_512': 108
    },
    divisor=8,
)
