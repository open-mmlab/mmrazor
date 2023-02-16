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
    'backbone.layer1.0.conv1_(0, 64)_64': 27,
    'backbone.layer1.0.conv2_(0, 64)_64': 35,
    'backbone.layer1.0.conv3_(0, 256)_256': 241,
    'backbone.layer1.1.conv1_(0, 64)_64': 32,
    'backbone.layer1.1.conv2_(0, 64)_64': 29,
    'backbone.layer1.2.conv1_(0, 64)_64': 27,
    'backbone.layer1.2.conv2_(0, 64)_64': 42,
    'backbone.layer2.0.conv1_(0, 128)_128': 87,
    'backbone.layer2.0.conv2_(0, 128)_128': 107,
    'backbone.layer2.0.conv3_(0, 512)_512': 512,
    'backbone.layer2.1.conv1_(0, 128)_128': 44,
    'backbone.layer2.1.conv2_(0, 128)_128': 50,
    'backbone.layer2.2.conv1_(0, 128)_128': 52,
    'backbone.layer2.2.conv2_(0, 128)_128': 81,
    'backbone.layer2.3.conv1_(0, 128)_128': 47,
    'backbone.layer2.3.conv2_(0, 128)_128': 50,
    'backbone.layer3.0.conv1_(0, 256)_256': 210,
    'backbone.layer3.0.conv2_(0, 256)_256': 206,
    'backbone.layer3.0.conv3_(0, 1024)_1024': 1024,
    'backbone.layer3.1.conv1_(0, 256)_256': 107,
    'backbone.layer3.1.conv2_(0, 256)_256': 108,
    'backbone.layer3.2.conv1_(0, 256)_256': 86,
    'backbone.layer3.2.conv2_(0, 256)_256': 126,
    'backbone.layer3.3.conv1_(0, 256)_256': 91,
    'backbone.layer3.3.conv2_(0, 256)_256': 112,
    'backbone.layer3.4.conv1_(0, 256)_256': 98,
    'backbone.layer3.4.conv2_(0, 256)_256': 110,
    'backbone.layer3.5.conv1_(0, 256)_256': 112,
    'backbone.layer3.5.conv2_(0, 256)_256': 115,
    'backbone.layer4.0.conv1_(0, 512)_512': 397,
    'backbone.layer4.0.conv2_(0, 512)_512': 427,
    'backbone.layer4.1.conv1_(0, 512)_512': 373,
    'backbone.layer4.1.conv2_(0, 512)_512': 348,
    'backbone.layer4.2.conv1_(0, 512)_512': 433,
    'backbone.layer4.2.conv2_(0, 512)_512': 384
}
divisor = 8
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
