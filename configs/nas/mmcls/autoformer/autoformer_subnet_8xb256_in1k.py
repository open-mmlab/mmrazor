_base_ = 'autoformer_supernet_32xb256_in1k.py'

_base_.model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='STEP2_SUBNET_YAML.yaml')

test_cfg = dict(evaluate_fixed_subnet=True)
