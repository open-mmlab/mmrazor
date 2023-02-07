_base_ = ['dcff_topdown_heatmap_resnet50_coco.py']

# model settings
_base_.model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.architecture,
    fix_subnet='configs/pruning/mmpose/dcff/fix_subnet.json',
    mode='mutator',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='configs/pruning/mmpose/dcff/fix_subnet_weight.pth'))
