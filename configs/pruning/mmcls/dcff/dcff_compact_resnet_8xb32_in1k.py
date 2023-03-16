_base_ = ['dcff_resnet_8xb32_in1k.py']

# model settings
_base_.model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.architecture,
    fix_subnet='configs/pruning/mmcls/dcff/fix_subnet.json',
    mode='mutator',
    init_cfg=dict(
        type='Pretrained',
        prefix='architecture',
        checkpoint='configs/pruning/mmcls/dcff/fix_subnet_weight.pth'))

_base_.val_cfg = dict()
