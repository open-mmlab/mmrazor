_base_ = ['./spos_mobilenet_supernet_8xb128_in1k.py']

model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='configs/nas/spos/AngleNAS_SHUFFLENETV2_IN1k_2.0.yaml')

model_wrapper_cfg = None

find_unused_parameters = False
