_base_ = ['dcff_pointrend_resnet50_8xb2_cityscapes.py']

# model settings
_base_.model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.architecture,
    fix_subnet='configs/pruning/mmseg/dcff/fix_subnet.json',
    mode='mutator',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='configs/pruning/mmseg/dcff/fix_subnet_weight.pth'))
