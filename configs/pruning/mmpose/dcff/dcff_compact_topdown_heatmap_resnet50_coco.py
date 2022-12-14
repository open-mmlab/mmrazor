_base_ = ['dcff_topdown_heatmap_resnet50_coco.py']

# model settings
model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model_prune',
    architecture=_base_.architecture,
    mutator_cfg='configs/pruning/mmpose/dcff/fix_subnet.json')
