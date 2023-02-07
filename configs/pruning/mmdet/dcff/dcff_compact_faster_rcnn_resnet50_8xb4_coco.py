_base_ = ['dcff_faster_rcnn_resnet50_8xb4_coco.py']

# model settings
_base_.model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.architecture,
    fix_subnet='configs/pruning/mmdet/dcff/fix_subnet.json',
    mode='mutator',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='configs/pruning/mmdet/dcff/fix_subnet_weight.pth'))
