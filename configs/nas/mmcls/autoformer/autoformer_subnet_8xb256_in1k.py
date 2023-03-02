_base_ = 'autoformer_supernet_32xb256_in1k.py'

model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='configs/nas/mmcls/autoformer/AUTOFORMER_SUBNET_B.yaml',
    # You can also load the checkpoint of supernet instead of the specific
    # subnet by modifying the `checkpoint`(path) in the following `init_cfg`
    # with `init_weight_from_supernet = True`.
    init_weight_from_supernet=False,
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmrazor/v1/autoformer/autoformer_supernet_32xb256_in1k_20220919_110144-c658ce8f.pth',  # noqa: E501
        prefix='architecture.'))
