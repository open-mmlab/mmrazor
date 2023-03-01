_base_ = ['./dsnas_supernet_8xb128_in1k.py']

model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet=  # noqa: E251
    'configs/nas/mmcls/dsnas/DSNAS_SUBNET_IMAGENET_PAPER_ALIAS.yaml'
)  # noqa: E501

find_unused_parameters = False
