_base_ = './resnet_group_fisher_prune.py'
custom_imports = dict(imports=['projects'])

algorithm = _base_.model
pruned_path = './work_dirs/resnet_group_fisher_prune/flops_0.50.pth'
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='PruneDeployWrapper',
    algorithm=algorithm,
)

custom_hooks = [
    dict(type='mmrazor.PruneHook'),
]

# restore optimizer

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001,
        _scope_='mmcls'))
param_scheduler = dict(
    _delete_=True,
    type='MultiStepLR',
    by_epoch=True,
    milestones=[30, 60, 90],
    gamma=0.1,
    _scope_='mmcls')

# remove pruning related hooks
custom_hooks = _base_.custom_hooks[:-2]

# delete ddp
model_wrapper_cfg = None
