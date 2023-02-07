_base_ = ['./spos_mobilenet_supernet_8xb128_in1k.py']

# FIXME: you may replace this with the mutable_cfg searched by yourself
supernet = _base_.supernet

model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=supernet,
    fix_subnet='configs/nas/spos/AngleNAS_SHUFFLENETV2_IN1k_2.0.yaml',
    mode='mutator')

_base_.model = model_cfg

find_unused_parameters = False
