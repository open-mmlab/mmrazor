_base_ = ['./spos_shufflenet_supernet_8xb128_in1k.py']

# FIXME: you may replace this with the searched by yourself
supernet = _base_.supernet

model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=supernet,
    fix_subnet='configs/nas/mmcls/spos/SPOS_SUBNET.yaml')

_base_.model = model_cfg

find_unused_parameters = False
