_base_ = ['dcff_topdown_heatmap_resnet50_coco.py']

# model settings
model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.architecture,
    mode='mutator',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='configs/pruning/mmpose/dcff/fix_subnet_weight.pth'))
