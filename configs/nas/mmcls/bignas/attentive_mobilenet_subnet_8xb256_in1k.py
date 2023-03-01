_base_ = 'attentive_mobilenet_supernet_32xb64_in1k.py'

model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='configs/nas/mmcls/bignas/ATTENTIVE_SUBNET_A0.yaml',
)

model_wrapper_cfg = None
find_unused_parameters = True

test_cfg = dict(evaluate_fixed_subnet=True)

default_hooks = dict(checkpoint=None)
