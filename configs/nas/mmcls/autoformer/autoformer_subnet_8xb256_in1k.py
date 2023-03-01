_base_ = 'autoformer_supernet_32xb256_in1k.py'

_base_.model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=supernet,
    fix_subnet='configs/nas/mmcls/autoformer/AUTOFORMER_SUBNET_B.yaml')

_base_.model = model_cfg
