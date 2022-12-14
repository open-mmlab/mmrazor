_base_ = ['dcff_faster_rcnn_resnet50_8xb4_coco.py']

# model settings
model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model_prune',
    architecture=_base_.architecture,
    mutator_cfg='configs/pruning/mmdet/dcff/fix_subnet.json')
