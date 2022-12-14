_base_ = ['dcff_resnet_8xb32_in1k.py']

# model settings
model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model_prune',
    architecture=dict(
        cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py', pretrained=False),
    mutator_cfg='configs/pruning/mmcls/dcff/fix_subnet.json')
