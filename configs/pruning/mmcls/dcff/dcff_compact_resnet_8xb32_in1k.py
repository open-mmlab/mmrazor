_base_ = ['dcff_resnet_8xb32_in1k.py']

# model settings
_base_.model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=dict(
        cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py', pretrained=False),
    fix_subnet='configs/pruning/mmcls/dcff/fix_subnet.json',
    mode='mutator',
    init_cfg=dict(
        type='Pretrained',
        checkpoint='configs/pruning/mmcls/dcff/fix_subnet_weight.pth'))
