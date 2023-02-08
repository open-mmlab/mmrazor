_base_ = ['./dsnas_supernet_8xb128_in1k.py']

# NOTE: Replace this with the mutable_cfg searched by yourself.
supernet = _base_.model['architecture']

model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=supernet,
    fix_subnet=  # noqa: E251
    'configs/nas/mmcls/dsnas/DSNAS_SUBNET_IMAGENET_PAPER_ALIAS.yaml'
)  # noqa: E501

_base_.model = model_cfg

find_unused_parameters = False
