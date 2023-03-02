_base_ = ['./spos_mobilenet_supernet_8xb128_in1k.py']

model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='configs/nas/spos/AngleNAS_SHUFFLENETV2_IN1k_2.0.yaml',
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmrazor/v1/spos/spos_mobilenetv2_subnet_8xb128_in1k_flops_0.33M_acc_73.87_20211222-1f0a0b4d_v3.pth',  # noqa: E501
        prefix='architecture.'))

model_wrapper_cfg = None

find_unused_parameters = False
