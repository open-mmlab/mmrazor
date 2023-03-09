#############################################################################
"""You have to fill these args.

_base_(str): The path to your pretrain config file.
fix_subnet (Union[dict,str]): The dict store the pruning structure or the
    json file including it.
divisor (int): The divisor the make the channel number divisible.
"""

_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'
fix_subnet = {
    'backbone.conv1_(0, 64)_64': 60,
    'backbone.layer1.0.conv1_(0, 64)_64': 48,
    'backbone.layer1.0.conv2_(0, 64)_64': 44,
    'backbone.layer1.0.conv3_(0, 256)_256': 250,
    'backbone.layer1.1.conv1_(0, 64)_64': 40,
    'backbone.layer1.1.conv2_(0, 64)_64': 41,
    'backbone.layer1.2.conv1_(0, 64)_64': 48,
    'backbone.layer1.2.conv2_(0, 64)_64': 62,
    'backbone.layer2.0.conv1_(0, 128)_128': 115,
    'backbone.layer2.0.conv2_(0, 128)_128': 127,
    'backbone.layer2.0.conv3_(0, 512)_512': 511,
    'backbone.layer2.1.conv1_(0, 128)_128': 69,
    'backbone.layer2.1.conv2_(0, 128)_128': 83,
    'backbone.layer2.2.conv1_(0, 128)_128': 111,
    'backbone.layer2.2.conv2_(0, 128)_128': 121,
    'backbone.layer2.3.conv1_(0, 128)_128': 122,
    'backbone.layer2.3.conv2_(0, 128)_128': 128,
    'backbone.layer3.0.conv1_(0, 256)_256': 255,
    'backbone.layer3.0.conv2_(0, 256)_256': 256,
    'backbone.layer3.0.conv3_(0, 1024)_1024': 1024,
    'backbone.layer3.1.conv1_(0, 256)_256': 216,
    'backbone.layer3.1.conv2_(0, 256)_256': 223,
    'backbone.layer3.2.conv1_(0, 256)_256': 229,
    'backbone.layer3.2.conv2_(0, 256)_256': 247,
    'backbone.layer3.3.conv1_(0, 256)_256': 239,
    'backbone.layer3.3.conv2_(0, 256)_256': 246,
    'backbone.layer3.4.conv1_(0, 256)_256': 237,
    'backbone.layer3.4.conv2_(0, 256)_256': 239,
    'backbone.layer3.5.conv1_(0, 256)_256': 233,
    'backbone.layer3.5.conv2_(0, 256)_256': 221,
    'backbone.layer4.0.conv1_(0, 512)_512': 499,
    'backbone.layer4.0.conv2_(0, 512)_512': 494,
    'backbone.layer4.0.conv3_(0, 2048)_2048': 2031,
    'backbone.layer4.1.conv1_(0, 512)_512': 451,
    'backbone.layer4.1.conv2_(0, 512)_512': 401,
    'backbone.layer4.2.conv1_(0, 512)_512': 396,
    'backbone.layer4.2.conv2_(0, 512)_512': 237,
    'neck.lateral_convs.0.conv_(0, 256)_256': 237,
    'neck.fpn_convs.0.conv_(0, 256)_256': 241,
    'bbox_head.cls_convs.0.conv_(0, 256)_256': 133,
    'bbox_head.cls_convs.1.conv_(0, 256)_256': 134,
    'bbox_head.cls_convs.2.conv_(0, 256)_256': 139,
    'bbox_head.cls_convs.3.conv_(0, 256)_256': 79,
    'bbox_head.reg_convs.0.conv_(0, 256)_256': 89,
    'bbox_head.reg_convs.1.conv_(0, 256)_256': 92,
    'bbox_head.reg_convs.2.conv_(0, 256)_256': 82,
    'bbox_head.reg_convs.3.conv_(0, 256)_256': 117
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
