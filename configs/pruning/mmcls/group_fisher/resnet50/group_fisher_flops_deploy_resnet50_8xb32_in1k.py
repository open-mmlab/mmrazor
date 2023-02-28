#############################################################################
"""You have to fill these args.

_base_(str): The path to your pretrain config file.
fix_subnet (Union[dict,str]): The dict store the pruning structure or the
    json file including it.
divisor (int): The divisor the make the channel number divisible.
"""

_base_ = 'mmcls::resnet/resnet50_8xb32_in1k.py'
fix_subnet = {
    'backbone.conv1_(0, 64)_64': 61,
    'backbone.layer1.0.conv1_(0, 64)_64': 28,
    'backbone.layer1.0.conv2_(0, 64)_64': 35,
    'backbone.layer1.0.conv3_(0, 256)_256': 242,
    'backbone.layer1.1.conv1_(0, 64)_64': 31,
    'backbone.layer1.1.conv2_(0, 64)_64': 28,
    'backbone.layer1.2.conv1_(0, 64)_64': 26,
    'backbone.layer1.2.conv2_(0, 64)_64': 41,
    'backbone.layer2.0.conv1_(0, 128)_128': 90,
    'backbone.layer2.0.conv2_(0, 128)_128': 107,
    'backbone.layer2.0.conv3_(0, 512)_512': 509,
    'backbone.layer2.1.conv1_(0, 128)_128': 42,
    'backbone.layer2.1.conv2_(0, 128)_128': 50,
    'backbone.layer2.2.conv1_(0, 128)_128': 51,
    'backbone.layer2.2.conv2_(0, 128)_128': 84,
    'backbone.layer2.3.conv1_(0, 128)_128': 49,
    'backbone.layer2.3.conv2_(0, 128)_128': 51,
    'backbone.layer3.0.conv1_(0, 256)_256': 210,
    'backbone.layer3.0.conv2_(0, 256)_256': 207,
    'backbone.layer3.0.conv3_(0, 1024)_1024': 1024,
    'backbone.layer3.1.conv1_(0, 256)_256': 103,
    'backbone.layer3.1.conv2_(0, 256)_256': 108,
    'backbone.layer3.2.conv1_(0, 256)_256': 90,
    'backbone.layer3.2.conv2_(0, 256)_256': 124,
    'backbone.layer3.3.conv1_(0, 256)_256': 94,
    'backbone.layer3.3.conv2_(0, 256)_256': 114,
    'backbone.layer3.4.conv1_(0, 256)_256': 99,
    'backbone.layer3.4.conv2_(0, 256)_256': 111,
    'backbone.layer3.5.conv1_(0, 256)_256': 108,
    'backbone.layer3.5.conv2_(0, 256)_256': 111,
    'backbone.layer4.0.conv1_(0, 512)_512': 400,
    'backbone.layer4.0.conv2_(0, 512)_512': 421,
    'backbone.layer4.1.conv1_(0, 512)_512': 377,
    'backbone.layer4.1.conv2_(0, 512)_512': 347,
    'backbone.layer4.2.conv1_(0, 512)_512': 443,
    'backbone.layer4.2.conv2_(0, 512)_512': 376
}
divisor = 16
##############################################################################

architecture = _base_.model

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherDeploySubModel',
    architecture=architecture,
    fix_subnet=fix_subnet,
    divisor=divisor,
)
