_base_ = ['./detnas_frcnn_shufflenet_supernet_coco_1x.py']

# FIXME: you may replace this with the searched by yourself
supernet = _base_.supernet

model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=supernet,
    fix_subnet='configs/nas/mmdet/detnas/DETNAS_SUBNET.yaml')

_base_.model = model_cfg

find_unused_parameters = False
