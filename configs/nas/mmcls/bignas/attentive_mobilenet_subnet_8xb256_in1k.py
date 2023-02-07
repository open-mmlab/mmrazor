_base_ = 'attentive_mobilenet_supernet_32xb64_in1k.py'

supernet = _base_.supernet

model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=supernet,
    fix_subnet='configs/nas/mmcls/bignas/ATTENTIVE_SUBNET_A6.yaml',
    mode='mutator')

_base_.model = model_cfg

test_cfg = dict(evaluate_fixed_subnet=True)
