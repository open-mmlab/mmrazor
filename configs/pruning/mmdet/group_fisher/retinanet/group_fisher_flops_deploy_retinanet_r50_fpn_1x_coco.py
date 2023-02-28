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
    'backbone.layer1.0.conv1_(0, 64)_64': 47,
    'backbone.layer1.0.conv2_(0, 64)_64': 44,
    'backbone.layer1.0.conv3_(0, 256)_256': 249,
    'backbone.layer1.1.conv1_(0, 64)_64': 37,
    'backbone.layer1.1.conv2_(0, 64)_64': 37,
    'backbone.layer1.2.conv1_(0, 64)_64': 44,
    'backbone.layer1.2.conv2_(0, 64)_64': 62,
    'backbone.layer2.0.conv1_(0, 128)_128': 114,
    'backbone.layer2.0.conv2_(0, 128)_128': 127,
    'backbone.layer2.0.conv3_(0, 512)_512': 511,
    'backbone.layer2.1.conv1_(0, 128)_128': 65,
    'backbone.layer2.1.conv2_(0, 128)_128': 83,
    'backbone.layer2.2.conv1_(0, 128)_128': 106,
    'backbone.layer2.2.conv2_(0, 128)_128': 118,
    'backbone.layer2.3.conv1_(0, 128)_128': 118,
    'backbone.layer2.3.conv2_(0, 128)_128': 127,
    'backbone.layer3.0.conv1_(0, 256)_256': 255,
    'backbone.layer3.0.conv2_(0, 256)_256': 256,
    'backbone.layer3.0.conv3_(0, 1024)_1024': 1024,
    'backbone.layer3.1.conv1_(0, 256)_256': 214,
    'backbone.layer3.1.conv2_(0, 256)_256': 232,
    'backbone.layer3.2.conv1_(0, 256)_256': 224,
    'backbone.layer3.2.conv2_(0, 256)_256': 247,
    'backbone.layer3.3.conv1_(0, 256)_256': 240,
    'backbone.layer3.3.conv2_(0, 256)_256': 246,
    'backbone.layer3.4.conv1_(0, 256)_256': 240,
    'backbone.layer3.4.conv2_(0, 256)_256': 243,
    'backbone.layer3.5.conv1_(0, 256)_256': 238,
    'backbone.layer3.5.conv2_(0, 256)_256': 232,
    'backbone.layer4.0.conv1_(0, 512)_512': 503,
    'backbone.layer4.0.conv2_(0, 512)_512': 500,
    'backbone.layer4.0.conv3_(0, 2048)_2048': 2041,
    'backbone.layer4.1.conv1_(0, 512)_512': 466,
    'backbone.layer4.1.conv2_(0, 512)_512': 430,
    'backbone.layer4.2.conv1_(0, 512)_512': 406,
    'backbone.layer4.2.conv2_(0, 512)_512': 274,
    'neck.lateral_convs.0.conv_(0, 256)_256': 236,
    'neck.fpn_convs.0.conv_(0, 256)_256': 225,
    'bbox_head.cls_convs.0.conv_(0, 256)_256': 140,
    'bbox_head.cls_convs.1.conv_(0, 256)_256': 133,
    'bbox_head.cls_convs.2.conv_(0, 256)_256': 139,
    'bbox_head.cls_convs.3.conv_(0, 256)_256': 86,
    'bbox_head.reg_convs.0.conv_(0, 256)_256': 89,
    'bbox_head.reg_convs.1.conv_(0, 256)_256': 89,
    'bbox_head.reg_convs.2.conv_(0, 256)_256': 76,
    'bbox_head.reg_convs.3.conv_(0, 256)_256': 122,
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
