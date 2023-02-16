_base_ = ['./dsnas_supernet_8xb128_in1k.py']

# NOTE: Replace this with the mutable_cfg searched by yourself.
supernet = _base_.model['architecture']

paramwise_cfg = dict(norm_decay_mult=0.0, bias_decay_mult=0.0)
_base_.optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.8, momentum=0.9, weight_decay=0.00004, nesterov=True),
    paramwise_cfg=paramwise_cfg)

epochs = 200

param_scheduler = [
    dict(
        type='LinearLR',
        end=5,
        start_factor=0.2,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epochs,
        begin=5,
        end=epochs,
        by_epoch=True,
        convert_to_iter_based=True)
]

model_cfg = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=supernet,
    fix_subnet=  # noqa: E251
    'configs/nas/mmcls/dsnas/DSNAS_SUBNET_IMAGENET_PAPER_ALIAS.yaml'
)  # noqa: E501

_base_.model = model_cfg

find_unused_parameters = False
