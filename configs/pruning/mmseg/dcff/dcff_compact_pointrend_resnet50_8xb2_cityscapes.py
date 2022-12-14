_base_ = ['dcff_pointrend_resnet50_8xb2_cityscapes.py']

# model settings
model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model_prune',
    architecture=_base_.architecture,
    mutator_cfg='configs/pruning/mmseg/dcff/fix_subnet.json')
