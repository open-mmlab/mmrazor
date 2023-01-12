# Suppose you are in mmdet and want to use the searched subnet
# as backbone for faster-rcnn, then you can just use this config.

_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    'mmrazor::_base_/nas_backbones/spos_shufflenet_supernet.py'
]

_base_.nas_backbone.out_indices = (0, 1, 2, 3)
_base_.nas_backbone.with_last_layer = False
nas_backbone = dict(
    # use mmrazor's build_func
    type='mmrazor.sub_model',
    cfg=_base_.nas_backbone,
    fix_subnet='/path/to/your/mmrazor/configs/nas/mmcls/spos/SPOS_SUBNET.yaml',
    extra_prefix='backbone.')

_base_.model.backbone = nas_backbone
_base_.model.neck.in_channels = [64, 160, 320, 640]
