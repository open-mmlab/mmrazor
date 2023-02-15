_base_ = ['dmcp_mbv2_supernet_32xb64.py']

paramwise_cfg = dict(norm_decay_mult=0.0, bias_decay_mult=0.0)

_base_.optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.8, momentum=0.9, weight_decay=0.00004, nesterov=True),
    paramwise_cfg=paramwise_cfg)

max_epochs = 100

_base_.param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.25,
        by_epoch=True,
        begin=0,
        end=3,
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        eta_min=1e-5,
        by_epoch=True,
        begin=3,
        end=max_epochs,
        convert_to_iter_based=True),
]

_base_.train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)

custom_hooks = None

# model settings
model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    fix_subnet='configs/pruning/mmcls/dmcp/DMCP_MBV2_100M.json',
    mode='mutator')

default_hooks = _base_.default_hooks
default_hooks['checkpoint'] = dict(type='CheckpointHook', interval=5)

_base_.model_wrapper_cfg = None

randomness = dict(seed=4872, diff_rank_seed=True)
