_base_ = ['./spos_shufflenet_supernet_8xb128_in1k.py']

_base_.model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='configs/nas/mmcls/spos/SPOS_SUBNET.yaml')

model_wrapper_cfg = None

find_unused_parameters = False
