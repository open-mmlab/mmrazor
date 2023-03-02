_base_ = 'ofa_mobilenet_supernet_32xb64_in1k.py'

model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='configs/nas/mmcls/onceforall/OFA_SUBNET_NOTE8_LAT31.yaml',
    # You can also load the checkpoint of supernet instead of the specific
    # subnet by modifying the `checkpoint`(path) in the following `init_cfg`
    # with `init_weight_from_supernet = True`.
    init_weight_from_supernet=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmrazor/v1/ofa/ofa_mobilenet_subnet_8xb256_in1k_note8_lat%4031ms_top1%4072.8_finetune%4025.py_20221214_0939-981a8b2a.pth',  # noqa: E501
        prefix='architecture.'))

model_wrapper_cfg = None
find_unused_parameters = True

test_cfg = dict(evaluate_fixed_subnet=True)
