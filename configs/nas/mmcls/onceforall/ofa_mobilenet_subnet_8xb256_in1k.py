_base_ = 'ofa_mobilenet_supernet_32xb64_in1k.py'

supernet = _base_.supernet

model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=supernet,
    fix_subnet='configs/nas/mmcls/onceforall/OFA_SUBNET_NOTE8_LAT22.yaml')

_base_.model = model_cfg

test_cfg = dict(evaluate_fixed_subnet=True)
