_base_ = 'attentive_mobilenet_supernet_32xb64_in1k.py'

supernet = _base_.supernet

model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=supernet,
    fix_subnet='configs/nas/mmcls/bignas/ATTENTIVE_SUBNET_A6.yaml')

_base_.model = model_cfg
_base_.model_wrapper_cfg = None
find_unused_parameters = True

test_cfg = dict(evaluate_fixed_subnet=True)
