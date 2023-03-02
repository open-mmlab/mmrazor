_base_ = 'attentive_mobilenet_supernet_32xb64_in1k.py'

model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='configs/nas/mmcls/bignas/ATTENTIVE_SUBNET_A0.yaml',
    # You can load the checkpoint of supernet instead of the specific
    # subnet by modifying the `checkpoint`(path) in the following `init_cfg`
    # with `init_weight_from_supernet = True`.
    init_weight_from_supernet=True,
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmrazor/v1/bignas/attentive_mobilenet_supernet_32xb64_in1k_flops-2G_acc-81.72_20221229_200440-954772a3.pth',  # noqa: E501
        prefix='architecture.'))

model_wrapper_cfg = None
find_unused_parameters = True

test_cfg = dict(evaluate_fixed_subnet=True)

default_hooks = dict(checkpoint=None)
