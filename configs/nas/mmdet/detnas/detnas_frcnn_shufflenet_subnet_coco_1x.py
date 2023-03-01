_base_ = ['./detnas_frcnn_shufflenet_supernet_coco_1x.py']

model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='configs/nas/mmdet/detnas/DETNAS_SUBNET.yaml')

find_unused_parameters = False
