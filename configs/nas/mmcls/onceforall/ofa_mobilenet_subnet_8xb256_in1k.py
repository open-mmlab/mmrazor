_base_ = 'ofa_mobilenet_supernet_32xb64_in1k.py'

_base_.supernet.data_preprocessor = _base_.data_preprocessor

supernet = _base_.supernet

model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=supernet,
    fix_subnet='configs/nas/mmcls/onceforall/OFA_SUBNET_NOTE8_LAT31.yaml')
    # fix_subnet='configs/nas/mmcls/onceforall/OFA_SUBNET_NOTE8_LAT22.yaml')

_base_.model = model_cfg
_base_.model_wrapper_cfg = None
find_unused_parameters = True

test_cfg = dict(evaluate_fixed_subnet=True)
